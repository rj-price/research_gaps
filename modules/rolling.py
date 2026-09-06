"""Rolling gap synthesis from watched abstracts.

The PDF pipeline in `main.py` produces gaps from full text: papers the group has actually
read, whose Discussion sections state their own limitations. This module produces gaps
from the abstracts the watcher already screens, so the gap store keeps growing between
those deliberate analyses.

Three things make it different from the PDF path, and each is deliberate:

  * Abstracts state findings, not limitations. `PaperSummary.limitations` is the Critic's
    richest input and there is no equivalent here, so gaps drawn this way are marked
    `origin='abstract'` and should be read as leads rather than conclusions.
  * A subject is synthesised only once enough papers have accumulated for it
    (`rolling_min_papers`), not on every weekly run. Five abstracts do not make a field.
  * Every candidate gap is checked against the gaps already stored for that subject before
    it is written. `make_gap_id` hashes the exact title, so without this step a reworded
    restatement of the same gap becomes a second row every cycle, and the matcher's prompt
    grows without bound.
"""
import json
import logging
from typing import Dict, List, Tuple

from modules import db
from modules.agents import run_critic_agent, run_synthesiser_agent
from modules.config import SubjectGroup, WatchConfig, route_to_subjects
from modules.llm import generate_structured
from modules.models import GapMergeResult, IdentifiedGap

logger = logging.getLogger(__name__)

MAX_ABSTRACT_CHARS = 2500

# Told to both agents so they treat the evidence as what it is. Without it they write as
# though they had read the papers, and produce gaps asserting limitations no abstract stated.
SOURCE_NOTE = """
    IMPORTANT: these are published abstracts, not full papers. You are seeing stated
    findings only. No Methods, Discussion or author-stated limitations are available.
    Do not assert a limitation, flaw or absence of evidence that the abstracts do not
    support. Where a gap rests on what the abstracts do not report, say that the
    abstracts do not report it rather than claiming the work was never done.

    Return few gaps, not many. Every gap you return is stored and matched against future
    literature for months, so a vague one is worse than none at all. Reject the templates
    that fit any paper in any field: "test this in another species", "validate under field
    conditions", "account for environmental variation", "extend to other crops",
    "investigate the underlying mechanism". Only raise one of those if these specific
    abstracts make it concrete — naming the gene, isolate, host or measurement that the
    follow-up study would have to use. If the abstracts do not support a gap you can state
    that precisely, return fewer gaps.
"""


def _paper_block(paper: dict) -> str:
    abstract = (paper.get("abstract") or "No abstract available.")[:MAX_ABSTRACT_CHARS]
    return (
        f"Title: {paper['title']}\n"
        f"Source: {paper.get('source', '')} ({paper.get('published') or 'date unknown'})\n"
        f"Abstract: {abstract}"
    )


def group_backlog(backlog: List[dict], subjects: List[SubjectGroup]) -> Dict[str, List[dict]]:
    """Buckets unsynthesised papers by subject group, using the terms each paper matched."""
    grouped: Dict[str, List[dict]] = {subject.name: [] for subject in subjects}
    for paper in backlog:
        raw = paper.get("matched_terms") or "[]"
        try:
            terms = json.loads(raw) if isinstance(raw, str) else list(raw)
        except (ValueError, TypeError):
            terms = []
        for name in route_to_subjects(terms, subjects):
            grouped[name].append(paper)
    return grouped


async def merge_candidates(
    client, model_id: str, subject: str, candidates: List[IdentifiedGap], stored: List[dict], limiter,
) -> GapMergeResult:
    """Decides which candidate gaps restate a gap already stored for this subject."""
    if not stored or not candidates:
        return GapMergeResult(decisions=[])

    stored_listing = "\n\n".join(
        f"gap_id: {gap['gap_id']}\ntitle: {gap['title']}\ndescription: {gap['description']}"
        for gap in stored
    )
    candidate_listing = "\n\n".join(
        f"[{i}] title: {gap.title}\n    category: {gap.category}\n    description: {gap.description}"
        for i, gap in enumerate(candidates)
    )
    valid_ids = {gap["gap_id"] for gap in stored}

    prompt = f"""
    Our lab tracks open research gaps for the subject "{subject}". A fresh analysis of
    newly published literature has proposed the candidate gaps below. Decide, for each
    candidate, whether it is genuinely new or simply restates a gap we already track.

    Already stored gaps:
    {stored_listing}

    Candidate gaps:
    {candidate_listing}

    Rules:
    - Two gaps are the same when closing one would close the other. Different wording,
      different emphasis or a different example does not make a new gap.
    - Sharing an organism or a technique is NOT enough. A gap about effector presence
      /absence variation and a gap about host resistance durability in the same species
      are two gaps.
    - When a candidate is a duplicate, set duplicate_of to the stored gap_id exactly as
      written above, and write merged_description: a single 2-3 sentence statement
      covering everything both versions say, still specific enough to be matched against.
    - When a candidate is new, leave duplicate_of empty.
    - Return exactly one decision per candidate, with new_gap_index matching the number
      in brackets.
    """

    async with limiter:
        result = await generate_structured(
            client, model_id, prompt,
            response_model=GapMergeResult,
            system_instruction="You are a meticulous research librarian deduplicating a register of open questions.",
        )

    kept = []
    for decision in result.decisions:
        if not 0 <= decision.new_gap_index < len(candidates):
            logger.warning(f"Merge step returned out-of-range index {decision.new_gap_index}; ignored.")
            continue
        if decision.duplicate_of and decision.duplicate_of not in valid_ids:
            # An invented gap_id would silently drop the candidate, so treat it as new.
            logger.warning(f"Merge step named unknown gap_id {decision.duplicate_of}; treating candidate as new.")
            decision.duplicate_of = ""
        kept.append(decision)
    return GapMergeResult(decisions=kept)


async def synthesise_subject(
    client, model_id: str, subject: str, papers: List[dict], limiter,
) -> Tuple[int, int]:
    """Runs Synthesiser, Critic and the merge step for one subject. Returns (new, merged)."""
    logger.info(f"Rolling synthesis for '{subject}' over {len(papers)} abstracts.")
    summaries = [_paper_block(paper) for paper in papers]
    paper_ids = [paper["paper_id"] for paper in papers]

    synthesis = await run_synthesiser_agent(
        client, model_id, summaries, subject, generate_structured, source_note=SOURCE_NOTE
    )
    critic = await run_critic_agent(
        client, model_id, summaries, synthesis, generate_structured, source_note=SOURCE_NOTE
    )

    if not critic.gaps:
        logger.info(f"Critic returned no discrete gaps for '{subject}'.")
        await db.mark_synthesised(paper_ids)
        return 0, 0

    stored = [gap for gap in await db.get_gaps(status="open") if gap["subject"] == subject]
    merge = await merge_candidates(client, model_id, subject, critic.gaps, stored, limiter)
    duplicates = {d.new_gap_index: d for d in merge.decisions if d.duplicate_of}

    fresh = [gap for i, gap in enumerate(critic.gaps) if i not in duplicates]
    if fresh:
        await db.store_gaps(subject, fresh, origin="abstract", source_papers=paper_ids)

    for decision in duplicates.values():
        target = next(gap for gap in stored if gap["gap_id"] == decision.duplicate_of)
        await db.merge_into_gap(
            decision.duplicate_of,
            decision.merged_description.strip() or target["description"],
            source_papers=paper_ids,
        )
        logger.info(f"Merged candidate gap into {decision.duplicate_of}: {decision.reason}")

    await db.mark_synthesised(paper_ids)
    logger.info(f"'{subject}': {len(fresh)} new gaps, {len(duplicates)} merged into existing ones.")
    return len(fresh), len(duplicates)


async def run_rolling_synthesis(
    client, config: WatchConfig, limiter, force: bool = False,
) -> Tuple[int, int]:
    """Synthesises every subject whose backlog has reached the trigger. Returns (new, merged)."""
    if not config.subjects:
        logger.info("No subject groups declared in the watchlist; skipping rolling synthesis.")
        return 0, 0

    backlog = await db.get_synthesis_backlog()
    if not backlog:
        return 0, 0

    grouped = group_backlog(backlog, config.subjects)
    total_new = total_merged = 0

    for name, papers in grouped.items():
        if not papers:
            continue
        if len(papers) < config.rolling_min_papers and not force:
            logger.info(
                f"'{name}': {len(papers)}/{config.rolling_min_papers} papers banked; "
                f"waiting for a fuller backlog."
            )
            continue
        # Oldest first, so a capped run consumes the papers that have waited longest.
        selected = papers[: config.rolling_max_papers]
        try:
            new, merged = await synthesise_subject(client, config.model, name, selected, limiter)
        except Exception as e:
            # One subject failing must not cost the run its digest or the other subjects.
            # The papers stay unsynthesised, so the next run retries them.
            logger.error(f"Rolling synthesis failed for '{name}': {e}")
            continue
        total_new += new
        total_merged += merged

    return total_new, total_merged
