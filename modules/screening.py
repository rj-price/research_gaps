"""LLM screening for the ambient watcher.

Two steps, both with structured output:
  1. Relevance — is this paper genuinely about the watched organisms and interests?
  2. Gap matching — does it fill, partially address, contradict or inform a stored gap?

The keyword prefilter in `sources` is deliberately loose (a term in the abstract is
enough), so this layer is what stops the digest filling with noise.
"""
import json
import logging
from typing import Dict, List

from modules.llm import generate_structured
from modules.models import GapMatchResult, RelevanceBatch, RelevanceVerdict, WatchedPaper

logger = logging.getLogger(__name__)

MAX_ABSTRACT_CHARS = 2500


def _paper_block(index: int, paper: WatchedPaper) -> str:
    abstract = (paper.abstract or "No abstract available.")[:MAX_ABSTRACT_CHARS]
    return (
        f"[{index}] Title: {paper.title}\n"
        f"    Source: {paper.source} ({paper.published})\n"
        f"    Matched terms: {', '.join(paper.matched_terms) or 'none'}\n"
        f"    Abstract: {abstract}"
    )


async def screen_relevance(
    client: "OpenRouterClient", model_id: str, papers: List[WatchedPaper],
    interests: str, organisms: List[str], limiter, batch_size: int = 10,
) -> Dict[str, RelevanceVerdict]:
    """Judges a batch of papers against the watchlist. Returns verdicts keyed by paper_id."""
    verdicts: Dict[str, RelevanceVerdict] = {}

    for offset in range(0, len(papers), batch_size):
        batch = papers[offset:offset + batch_size]
        listing = "\n\n".join(_paper_block(i, paper) for i, paper in enumerate(batch))

        prompt = f"""
        You are screening newly published literature for a research group.

        Study organisms: {', '.join(organisms)}
        Research interests: {interests}

        For each of the {len(batch)} papers below, decide whether it is genuinely relevant
        to this group. Be strict. A paper is NOT relevant if it merely mentions one of the
        organisms in passing, uses it only as a reagent or outgroup, or is about an
        unrelated field that shares a keyword (for example a species name reused in
        another genus, or a clinical study that cites the organism incidentally).

        Return exactly one verdict per paper, with paper_index matching the number in
        brackets.

        {listing}
        """

        async with limiter:
            result = await generate_structured(
                client, model_id, prompt,
                response_model=RelevanceBatch,
                system_instruction="You are a meticulous literature screener. You reject weak matches.",
            )
        for verdict in result.verdicts:
            if 0 <= verdict.paper_index < len(batch):
                verdicts[batch[verdict.paper_index].paper_id] = verdict
            else:
                logger.warning(f"Screener returned out-of-range index {verdict.paper_index}; ignored.")

        logger.info(f"Screened {min(offset + batch_size, len(papers))}/{len(papers)} papers.")

    missing = [p.paper_id for p in papers if p.paper_id not in verdicts]
    if missing:
        logger.warning(f"{len(missing)} papers received no verdict; treating as not relevant.")
    return verdicts


async def match_paper_to_gaps(
    client: "OpenRouterClient", model_id: str, paper: WatchedPaper, gaps: List[dict], limiter,
) -> GapMatchResult:
    """Checks one paper against every stored open gap."""
    if not gaps:
        return GapMatchResult(matches=[])

    gap_listing = "\n\n".join(
        f"gap_id: {gap['gap_id']}\n"
        f"subject: {gap['subject']}\n"
        f"category: {gap['category']}\n"
        f"title: {gap['title']}\n"
        f"description: {gap['description']}"
        for gap in gaps
    )
    valid_ids = {gap["gap_id"] for gap in gaps}

    prompt = f"""
    A new paper has appeared. Below it is a list of research gaps previously identified
    by our lab's gap analysis. Decide which, if any, of those gaps this paper speaks to.

    New paper:
    Title: {paper.title}
    Source: {paper.source} ({paper.published})
    Authors: {paper.authors}
    Abstract: {(paper.abstract or 'No abstract available.')[:MAX_ABSTRACT_CHARS]}

    Stored research gaps:
    {gap_listing}

    Rules:
    - Only return a match where the paper's actual findings bear on the gap. Shared
      subject matter alone is not a match.
    - Use 'fills' only when the paper substantially closes the gap as described.
    - Use 'partially_addresses' when it makes real progress but leaves the gap open.
    - Use 'contradicts' when its findings conflict with the premise of the gap.
    - Use 'informs' for relevant methodology or context that does not close anything.
    - gap_id must be copied exactly from the list above.
    - Returning an empty list is the correct answer when nothing genuinely matches.
    """

    async with limiter:
        result = await generate_structured(
            client, model_id, prompt,
            response_model=GapMatchResult,
            system_instruction="You are a rigorous research analyst linking new findings to open questions.",
        )
    kept = [match for match in result.matches if match.gap_id in valid_ids]
    if len(kept) != len(result.matches):
        logger.warning(f"Dropped {len(result.matches) - len(kept)} matches with unrecognised gap_ids.")
    return GapMatchResult(matches=kept)
