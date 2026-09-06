"""Orchestration for the ambient literature watcher.

One run: fetch a date window from every source, drop papers already seen, screen the
remainder for relevance, match the survivors against stored research gaps, synthesise
new gaps from the accumulated backlog, then render and deliver a digest. Every stage is
recorded in SQLite so runs are resumable and a crashed run never re-screens the same
paper twice.
"""
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Tuple

from aiolimiter import AsyncLimiter

from modules import db, digest as digest_module, rolling, sources
from modules.config import WatchConfig
from modules.llm import get_client
from modules.models import WatchedPaper
from modules.screening import match_paper_to_gaps, screen_relevance

logger = logging.getLogger(__name__)


def resolve_window(since: str | None, last_run: str | None, default_days: int = 7) -> Tuple[date, date]:
    """Works out the fetch window: an explicit --since, else the last run, else a week."""
    end = datetime.now(timezone.utc).date()
    if since:
        if since.endswith("d") and since[:-1].isdigit():
            return end - timedelta(days=int(since[:-1])), end
        return date.fromisoformat(since), end
    if last_run:
        # Overlap by a day: PubMed entry dates settle late, so the boundary is fuzzy
        start = datetime.fromisoformat(last_run).date() - timedelta(days=1)
        return min(start, end), end
    return end - timedelta(days=default_days), end


async def collect_new_papers(config: WatchConfig, start: date, end: date) -> Tuple[List[WatchedPaper], int]:
    """Fetches the window and returns only papers not already in the database."""
    fetched = await sources.fetch_all(
        terms=config.terms,
        start=start,
        end=end,
        preprint_servers=config.preprint_servers,
        extra_query=config.pubmed_query,
        retmax=config.max_results,
        preprint_max_pages=config.preprint_max_pages,
    )
    logger.info(f"Fetched {len(fetched)} candidate papers from {start} to {end}.")

    new_ids = await db.filter_new_papers([paper.paper_id for paper in fetched])
    new_papers = [paper for paper in fetched if paper.paper_id in new_ids]
    logger.info(f"{len(new_papers)} are new since the last run.")
    return new_papers, len(fetched)


async def run_watch(
    config: WatchConfig, since: str | None = None, digest_path: str | None = None,
    deliver: bool = True, dry_run: bool = False, rolling_synthesis: bool = True,
    force_rolling: bool = False,
) -> str:
    """Runs one watch cycle and returns the rendered digest."""
    await db.init_db()

    last_run = await db.last_successful_run()
    start, end = resolve_window(since, last_run)

    if dry_run:
        # No LLM calls, no writes: used to check the sources and the keyword prefilter
        fetched = await sources.fetch_all(
            terms=config.terms, start=start, end=end,
            preprint_servers=config.preprint_servers,
            extra_query=config.pubmed_query, retmax=config.max_results,
            preprint_max_pages=config.preprint_max_pages,
        )
        lines = [f"# Dry run: {start} to {end}", "", f"{len(fetched)} candidates before screening.", ""]
        for paper in fetched:
            lines.append(f"- [{paper.source}] {paper.title} ({', '.join(paper.matched_terms)})")
        report = "\n".join(lines)
        print(report)
        return report

    run_id = await db.start_run(start.isoformat())
    relevant_count = match_count = 0
    client = None

    try:
        new_papers, fetched_count = await collect_new_papers(config, start, end)
        await db.store_papers(new_papers)

        if new_papers:
            client = get_client()
            limiter = AsyncLimiter(config.rate_limit, 60)

            verdicts = await screen_relevance(
                client, config.model, new_papers, config.interests, config.organisms, limiter
            )

            gaps = await db.get_gaps(status="open")
            # A paper that helped generate a gap must not then be reported as filling it.
            gap_sources = await db.get_gap_sources([gap["gap_id"] for gap in gaps])
            logger.info(f"Matching against {len(gaps)} stored open gaps.")

            for paper in new_papers:
                verdict = verdicts.get(paper.paper_id)
                if verdict is None:
                    await db.record_screening(paper.paper_id, False, 0.0, "No verdict returned by the screener.")
                    continue

                is_relevant = verdict.relevant and verdict.score >= config.min_relevance_score
                await db.record_screening(paper.paper_id, is_relevant, verdict.score, verdict.reason)
                if not is_relevant:
                    continue
                relevant_count += 1

                candidate_gaps = [
                    gap for gap in gaps if paper.paper_id not in gap_sources.get(gap["gap_id"], ())
                ]
                result = await match_paper_to_gaps(client, config.model, paper, candidate_gaps, limiter)
                for match in result.matches:
                    if match.confidence < config.min_match_confidence:
                        logger.info(
                            f"Discarded {match.relationship} match on {match.gap_id} "
                            f"for {paper.paper_id}: confidence {match.confidence:.2f} below threshold."
                        )
                        continue
                    await db.record_gap_match(
                        paper.paper_id, match.gap_id, match.relationship, match.confidence, match.evidence
                    )
                    match_count += 1
                    if match.relationship == "fills" and match.confidence >= 0.8:
                        await db.set_gap_status(match.gap_id, "addressed")
                        logger.info(f"Gap {match.gap_id} marked addressed by {paper.paper_id}.")

        # After matching, never before: gaps minted from this run's papers would otherwise
        # be offered straight back to those same papers as something to fill.
        # Checked before the client is built: a quiet week with nothing banked should not
        # need an API key at all, which is how the watcher behaved before this step existed.
        if rolling_synthesis and config.subjects and await db.get_synthesis_backlog(limit=1):
            if client is None:
                client = get_client()
            new_gaps, merged_gaps = await rolling.run_rolling_synthesis(
                client, config, AsyncLimiter(config.rate_limit, 60), force=force_rolling
            )
            if new_gaps or merged_gaps:
                logger.info(f"Rolling synthesis: {new_gaps} new gaps, {merged_gaps} merged.")

        pending = await db.get_undigested()
        report = digest_module.render_digest(pending, start.isoformat(), end.isoformat())

        if digest_path:
            with open(digest_path, "w") as handle:
                handle.write(report)
            logger.info(f"Digest written to {digest_path}.")

        if deliver and config.delivery and pending:
            subject = f"Literature digest {end.isoformat()}: {len(pending)} new papers"
            await digest_module.deliver(report, subject, config.delivery)

        await db.mark_digested([paper["paper_id"] for paper in pending])
        await db.finish_run(run_id, fetched_count, len(new_papers), relevant_count, match_count)
        logger.info(
            f"Run complete: {fetched_count} fetched, {len(new_papers)} new, "
            f"{relevant_count} relevant, {match_count} gap matches."
        )
        return report

    except Exception as e:
        logger.error(f"Watch run failed: {e}")
        await db.finish_run(run_id, 0, 0, relevant_count, match_count, status="failed")
        raise

    finally:
        if client is not None:
            await client.aclose()
