"""Ambient literature agent.

Watches PubMed and bioRxiv for the study organisms, screens new papers for relevance,
links them to research gaps already stored by the gap analysis pipeline, and sends a
digest. Designed to be run on a schedule (see the cron example in the README).
"""
import argparse
import asyncio
import json
import logging

from dotenv import load_dotenv

from aiolimiter import AsyncLimiter

from modules import db, rolling
from modules.config import load_config
from modules.llm import get_client
from modules.models import WatchedPaper
from modules.screening import screen_relevance
from modules.sources import _matched_terms as matched_terms
from modules.watch import run_watch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()


async def cmd_run(args) -> None:
    config = load_config(args.config)
    if args.model:
        config.model = args.model
    if args.deliver:
        config.delivery = args.deliver

    await run_watch(
        config,
        since=args.since,
        digest_path=args.output,
        deliver=not args.no_deliver,
        dry_run=args.dry_run,
        rolling_synthesis=not args.no_rolling,
        force_rolling=args.force_rolling,
    )


async def cmd_gaps(args) -> None:
    await db.init_db()
    gaps = await db.get_gaps(status=None if args.all else "open")
    if not gaps:
        print("No stored gaps. Run an analysis first: python main.py <pdf_folder> --subject '...'")
        return
    for gap in gaps:
        marker = "OPEN" if gap["status"] == "open" else gap["status"].upper()
        print(f"[{marker}] {gap['gap_id']}  {gap['title']}")
        origin = "abstracts" if gap.get("origin") == "abstract" else "full text"
        print(f"         subject: {gap['subject']}  |  category: {gap['category']}  |  from: {origin}")
        print(f"         {gap['description']}\n")


async def cmd_reopen(args) -> None:
    await db.init_db()
    await db.set_gap_status(args.gap_id, "open")
    print(f"Gap {args.gap_id} reopened.")


async def cmd_synthesise(args) -> None:
    """Runs rolling gap synthesis over the banked backlog without fetching anything new."""
    config = load_config(args.config)
    if args.model:
        config.model = args.model

    await db.init_db()
    backlog = await db.get_synthesis_backlog()
    if not backlog:
        print("No unsynthesised relevant papers banked. Run the watcher first.")
        return

    consumed = await db.get_synthesised_subjects([p["paper_id"] for p in backlog])
    grouped = rolling.group_backlog(backlog, config.subjects, consumed)
    unrouted = len(backlog) - len({p["paper_id"] for papers in grouped.values() for p in papers})
    for name, papers in sorted(grouped.items()):
        noun = "paper" if len(papers) == 1 else "papers"
        print(f"{name}: {len(papers)} {noun} banked (trigger at {config.rolling_min_papers})")
    if unrouted:
        print(f"{unrouted} banked papers match no subject group and will not be synthesised.")

    if args.status:
        return

    client = get_client()
    try:
        new_gaps, merged = await rolling.run_rolling_synthesis(
            client, config, AsyncLimiter(config.rate_limit, 60), force=args.force
        )
    finally:
        await client.aclose()
    print(f"Rolling synthesis complete: {new_gaps} new gaps, {merged} merged into existing ones.")


async def cmd_rescreen(args) -> None:
    """Re-screens every stored paper against the current watchlist interests.

    Use after editing `interests`: papers already screened keep their old verdict
    otherwise, and a paper wrongly kept stays in the synthesis backlog for good.
    """
    config = load_config(args.config)
    if args.model:
        config.model = args.model

    await db.init_db()
    rows = await db.get_all_papers()
    if not rows:
        print("No stored papers to re-screen.")
        return

    papers = [
        WatchedPaper(
            paper_id=row["paper_id"], source=row["source"], title=row["title"],
            abstract=row["abstract"] or "", authors=row["authors"] or "",
            published=row["published"] or "", url=row["url"] or "",
            matched_terms=json.loads(row["matched_terms"] or "[]"),
        )
        for row in rows
    ]
    # Watch terms first: routing reads matched_terms, which was frozen at fetch time.
    refreshed = [
        (paper.paper_id, matched_terms(f"{paper.title} {paper.abstract}", config.terms))
        for paper in papers
    ]
    changed = [(pid, terms) for (pid, terms), paper in zip(refreshed, papers) if terms != paper.matched_terms]
    if changed:
        await db.update_matched_terms(changed)
        print(f"Re-derived watch terms for {len(changed)} papers.")

    print(f"Re-screening {len(papers)} papers against the current interests.")

    client = get_client()
    try:
        verdicts = await screen_relevance(
            client, config.model, papers, config.interests, config.organisms,
            AsyncLimiter(config.rate_limit, 60),
        )
    finally:
        await client.aclose()

    await db.reset_screening()
    kept = 0
    for paper in papers:
        verdict = verdicts.get(paper.paper_id)
        if verdict is None:
            await db.record_screening(paper.paper_id, False, 0.0, "No verdict returned by the screener.")
            continue
        relevant = verdict.relevant and verdict.score >= config.min_relevance_score
        kept += relevant
        await db.record_screening(paper.paper_id, relevant, verdict.score, verdict.reason)

    print(f"{kept} of {len(papers)} papers are relevant under the current interests.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ambient literature agent for research_gaps")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Fetch, screen, match and digest new literature")
    run_parser.add_argument("--config", default="watchlist.json", help="Path to the watchlist JSON")
    run_parser.add_argument(
        "--since",
        help="Window start: an ISO date (2026-08-01) or a day count ('7d'). Defaults to the last successful run.",
    )
    run_parser.add_argument("--output", help="Write the digest to this Markdown file")
    run_parser.add_argument("--model", help="Override the OpenRouter model ID")
    run_parser.add_argument(
        "--deliver", nargs="*", choices=["email", "push"], help="Override the configured delivery channels"
    )
    run_parser.add_argument("--no-deliver", action="store_true", help="Generate the digest but send nothing")
    run_parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and keyword-filter only: no LLM calls, no database writes",
    )
    run_parser.add_argument(
        "--no-rolling", action="store_true",
        help="Skip rolling gap synthesis; screen and match only",
    )
    run_parser.add_argument(
        "--force-rolling", action="store_true",
        help="Synthesise every subject with a backlog, ignoring rolling_min_papers",
    )
    run_parser.set_defaults(func=cmd_run)

    gaps_parser = subparsers.add_parser("gaps", help="List the stored research gaps")
    gaps_parser.add_argument("--all", action="store_true", help="Include gaps already marked addressed")
    gaps_parser.set_defaults(func=cmd_gaps)

    synth_parser = subparsers.add_parser(
        "synthesise", help="Identify gaps from the banked watched papers, without fetching"
    )
    synth_parser.add_argument("--config", default="watchlist.json", help="Path to the watchlist JSON")
    synth_parser.add_argument("--model", help="Override the OpenRouter model ID")
    synth_parser.add_argument(
        "--force", action="store_true", help="Synthesise every subject with a backlog, ignoring rolling_min_papers"
    )
    synth_parser.add_argument(
        "--status", action="store_true", help="Report the backlog per subject and exit without calling the LLM"
    )
    synth_parser.set_defaults(func=cmd_synthesise)

    rescreen_parser = subparsers.add_parser(
        "rescreen", help="Re-screen every stored paper against the current watchlist interests"
    )
    rescreen_parser.add_argument("--config", default="watchlist.json", help="Path to the watchlist JSON")
    rescreen_parser.add_argument("--model", help="Override the OpenRouter model ID")
    rescreen_parser.set_defaults(func=cmd_rescreen)

    reopen_parser = subparsers.add_parser("reopen", help="Mark an addressed gap as open again")
    reopen_parser.add_argument("gap_id")
    reopen_parser.set_defaults(func=cmd_reopen)

    args = parser.parse_args()
    try:
        asyncio.run(args.func(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")


if __name__ == "__main__":
    main()
