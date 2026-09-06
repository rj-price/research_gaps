"""Ambient literature agent.

Watches PubMed and bioRxiv for the study organisms, screens new papers for relevance,
links them to research gaps already stored by the gap analysis pipeline, and sends a
digest. Designed to be run on a schedule (see the cron example in the README).
"""
import argparse
import asyncio
import logging

from dotenv import load_dotenv

from modules import db
from modules.config import load_config
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
        print(f"         subject: {gap['subject']}  |  category: {gap['category']}")
        print(f"         {gap['description']}\n")


async def cmd_reopen(args) -> None:
    await db.init_db()
    await db.set_gap_status(args.gap_id, "open")
    print(f"Gap {args.gap_id} reopened.")


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
    run_parser.set_defaults(func=cmd_run)

    gaps_parser = subparsers.add_parser("gaps", help="List the stored research gaps")
    gaps_parser.add_argument("--all", action="store_true", help="Include gaps already marked addressed")
    gaps_parser.set_defaults(func=cmd_gaps)

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
