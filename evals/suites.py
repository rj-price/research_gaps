"""The eval suites.

Each suite drives the real production code path — `screen_relevance`,
`match_paper_to_gaps`, the Synthesiser and Critic agents — rather than a
reimplementation, so a regression in the app shows up here.

Nothing touches the application database: the gap fixtures are passed to the matcher
directly, and the gap analysis suite calls the agents rather than
`run_multi_agent_pipeline`, which would persist its gaps.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from aiolimiter import AsyncLimiter
from pydantic import BaseModel, Field

from modules.agents import run_critic_agent, run_synthesiser_agent
from modules.llm import OpenRouterClient, generate_structured
from modules.models import CriticResult
from modules.screening import match_paper_to_gaps, screen_relevance

from evals import cases as datasets
from evals import judge as judge_module
from evals.metrics import BinaryScore, Mean, SetScore

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"unexplored_territory", "methodological_limitation", "contradiction"}
MAX_TITLE_WORDS = 15
JUDGE_ATTEMPTS = 3


class SuiteResult(BaseModel):
    suite: str
    dataset: str
    model: str
    judge_model: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    cases: List[Dict[str, Any]] = Field(default_factory=list)
    started_at: str = ""
    duration_s: float = 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def _attempt(label: str, call, attempts: int = JUDGE_ATTEMPTS):
    """Runs a judge call, returning None rather than losing the whole run to one failure.

    Provider-side content filters fire intermittently on the same prompt, so a refusal is
    worth another try; `generate_structured` already retries beneath this. A case the
    judge never scored is excluded from the means and counted, so a systematically broken
    judge shows up as a threshold breach instead of a suspiciously low score.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except Exception as e:
            logger.warning(f"{label}: judge attempt {attempt}/{attempts} failed: {e}")
    logger.error(f"{label}: judge gave up after {attempts} attempts.")
    return None


# --- Suite 1: relevance screening ---

async def run_relevance(
    client: OpenRouterClient, model: str, limiter: AsyncLimiter,
    limit: int | None = None, threshold: float = 0.6, batch_size: int = 10,
) -> SuiteResult:
    """Scores the watcher's first filter: does it keep the right papers and reject the traps?"""
    dataset = datasets.load_relevance()
    selected = dataset.cases[:limit] if limit else dataset.cases
    papers = [case.to_paper() for case in selected]

    start = time.monotonic()
    started_at = _now()
    verdicts = await screen_relevance(
        client, model, papers, dataset.interests, dataset.organisms, limiter, batch_size=batch_size,
    )

    score = BinaryScore()
    hard_negatives = BinaryScore()
    # The screener's score is confidence in its own verdict, not probability of relevance:
    # a confident rejection scores high. Averaging both directions together says nothing,
    # so they are reported apart — a screener hedging on the papers it keeps is the signal.
    confidence_kept, confidence_rejected = Mean(), Mean()
    rows: List[Dict[str, Any]] = []

    for case in selected:
        verdict = verdicts.get(case.paper_id)
        # A missing verdict is a rejection in production, so it is scored as one here
        predicted = bool(verdict and verdict.relevant and verdict.score >= threshold)
        score.add(predicted, case.relevant)
        if "hard-negative" in case.tags:
            hard_negatives.add(predicted, case.relevant)
        if verdict:
            (confidence_kept if verdict.relevant else confidence_rejected).add(verdict.score)

        rows.append({
            "paper_id": case.paper_id,
            "title": case.title,
            "expected": case.relevant,
            "predicted": predicted,
            "correct": predicted == case.relevant,
            "score": verdict.score if verdict else None,
            "reason": verdict.reason if verdict else "no verdict returned",
            "tags": case.tags,
        })

    metrics = score.as_dict()
    metrics["hard_negative_rejection_rate"] = round(
        hard_negatives.tn / hard_negatives.total if hard_negatives.total else 0.0, 4
    )
    metrics["no_verdict"] = sum(1 for row in rows if row["score"] is None)
    metrics.update(confidence_kept.as_dict("mean_confidence_kept"))
    metrics.update(confidence_rejected.as_dict("mean_confidence_rejected"))

    return SuiteResult(
        suite="relevance", dataset=dataset.name, model=model, metrics=metrics, cases=rows,
        started_at=started_at, duration_s=round(time.monotonic() - start, 1),
    )


# --- Suite 2: gap matching ---

async def run_gap_matching(
    client: OpenRouterClient, model: str, limiter: AsyncLimiter,
    limit: int | None = None, confidence_threshold: float = 0.75,
) -> SuiteResult:
    """Scores the link from a new paper to a stored gap: which gaps, and what relationship."""
    dataset = datasets.load_gap_matching()
    selected = dataset.cases[:limit] if limit else dataset.cases
    gap_rows = [gap.as_row() for gap in dataset.gaps]

    start = time.monotonic()
    started_at = _now()

    link_score = SetScore()
    relationship = BinaryScore()
    quiet = BinaryScore()  # does it stay silent on papers that match nothing?
    rows: List[Dict[str, Any]] = []

    for case in selected:
        result = await match_paper_to_gaps(client, model, case.to_paper(), gap_rows, limiter)
        # Mirror the watcher: low-confidence matches never reach the database
        kept = [m for m in result.matches if m.confidence >= confidence_threshold]
        predicted_ids = {m.gap_id for m in kept}

        link_score.add(predicted_ids, case.expected_ids, case.allowed)
        quiet.add(bool(predicted_ids), bool(case.expected_ids))

        by_id = {m.gap_id: m for m in kept}
        relationship_rows = []
        for expectation in case.expected:
            match = by_id.get(expectation.gap_id)
            if match is None:
                continue  # a missed link is already counted as a false negative
            correct = match.relationship in expectation.relationships
            relationship.add(correct, True)
            relationship_rows.append({
                "gap_id": expectation.gap_id,
                "expected": expectation.relationships,
                "predicted": match.relationship,
                "correct": correct,
            })

        rows.append({
            "paper_id": case.paper_id,
            "title": case.title,
            "expected_gaps": sorted(case.expected_ids),
            "allowed_gaps": sorted(case.allowed),
            "predicted_gaps": sorted(predicted_ids),
            "dropped_below_threshold": [
                {"gap_id": m.gap_id, "confidence": m.confidence}
                for m in result.matches if m.confidence < confidence_threshold
            ],
            "relationships": relationship_rows,
            "evidence": [{"gap_id": m.gap_id, "evidence": m.evidence} for m in kept],
        })

    metrics = link_score.as_dict()
    metrics["relationship_accuracy"] = round(relationship.accuracy, 4)
    metrics["relationship_n"] = relationship.total
    # Of the papers that should match nothing, how many did the matcher leave alone?
    metrics["silence_on_unrelated"] = round(quiet.tn / (quiet.tn + quiet.fp) if (quiet.tn + quiet.fp) else 0.0, 4)

    return SuiteResult(
        suite="gap_matching", dataset=dataset.name, model=model, metrics=metrics, cases=rows,
        started_at=started_at, duration_s=round(time.monotonic() - start, 1),
    )


# --- Suite 3: gap analysis quality ---

def _render_analysis(critic: CriticResult) -> str:
    """The Critic's output as the judge sees it — prose sections plus the discrete gaps."""
    lines = [
        "### Unexplored Territories", critic.unexplored_territories, "",
        "### Methodological Limitations", critic.methodological_limitations, "",
        "### Contradictions", critic.contradictions, "", "### Identified gaps",
    ]
    for gap in critic.gaps:
        lines.append(f"- **{gap.title}** ({gap.category}): {gap.description}")
    return "\n".join(lines)


def _structural_issues(critic: CriticResult) -> List[str]:
    """Checks that need no model: schema discipline the downstream watcher depends on."""
    issues = []
    if not critic.gaps:
        issues.append("no discrete gaps returned, so nothing is trackable by the watcher")
    for gap in critic.gaps:
        if gap.category not in VALID_CATEGORIES:
            issues.append(f"invalid category '{gap.category}' on gap '{gap.title}'")
        if len(gap.title.split()) > MAX_TITLE_WORDS:
            issues.append(f"title over {MAX_TITLE_WORDS} words: '{gap.title}'")
        if not gap.description.strip():
            issues.append(f"empty description on gap '{gap.title}'")
    return issues


async def run_gap_analysis(
    client: OpenRouterClient, model: str, limiter: AsyncLimiter,
    limit: int | None = None, judge_model_id: str | None = None,
) -> SuiteResult:
    """Runs the Synthesiser and Critic over fixed summaries and has a judge grade the gaps."""
    dataset = datasets.load_gap_analysis()
    selected = dataset.cases[:limit] if limit else dataset.cases
    judge_id = judge_model_id or judge_module.judge_model()

    start = time.monotonic()
    started_at = _now()

    grounded, specific = Mean(), Mean()
    theme_recall, fabrication = Mean(), Mean()
    rows: List[Dict[str, Any]] = []
    structural_failures = 0
    judge_errors = 0

    for case in selected:
        async with limiter:
            synthesis = await run_synthesiser_agent(
                client, model, case.summaries, case.subject, generate_structured
            )
        async with limiter:
            critic = await run_critic_agent(
                client, model, case.summaries, synthesis, generate_structured
            )

        issues = _structural_issues(critic)
        structural_failures += len(issues)

        grades = await _attempt(
            case.case_id,
            lambda: judge_module.grade_gaps(client, case.summaries, critic.gaps, limiter, judge_id),
        )
        if grades is None:
            judge_errors += 1
            grades = []
        for grade in grades:
            grounded.add(grade.grounded)
            specific.add(grade.specific)

        coverage = await _attempt(
            case.case_id,
            lambda: judge_module.check_coverage(
                client, _render_analysis(critic), case.expected_themes, case.distractor_claims,
                limiter, judge_id,
            ),
        )
        themes = coverage.themes if coverage else []
        asserted = [v for v in coverage.distractors if v.asserted] if coverage else []
        if coverage is None:
            # Scored as neither pass nor fail: an unjudged case is missing data, not a zero
            judge_errors += 1
        else:
            covered = [v for v in themes if v.covered]
            theme_recall.add(len(covered) / len(case.expected_themes) if case.expected_themes else 0.0)
            fabrication.add(len(asserted) / len(case.distractor_claims) if case.distractor_claims else 0.0)

        rows.append({
            "case_id": case.case_id,
            "subject": case.subject,
            "gap_count": len(critic.gaps),
            "structural_issues": issues,
            "gaps": [
                {
                    "title": gap.title,
                    "category": gap.category,
                    "description": gap.description,
                    "grounded": next((g.grounded for g in grades if g.gap_index == i), None),
                    "specific": next((g.specific for g in grades if g.gap_index == i), None),
                    "judge_comment": next((g.comment for g in grades if g.gap_index == i), ""),
                }
                for i, gap in enumerate(critic.gaps)
            ],
            "judge_failed": coverage is None or (bool(critic.gaps) and not grades),
            "themes": [
                {"theme": case.expected_themes[v.theme_index], "covered": v.covered, "covering_gap": v.covering_gap}
                for v in themes
                if 0 <= v.theme_index < len(case.expected_themes)
            ],
            "fabrications": [
                {"claim": case.distractor_claims[v.claim_index], "quote": v.quote}
                for v in asserted
                if 0 <= v.claim_index < len(case.distractor_claims)
            ],
        })

    metrics: Dict[str, Any] = {}
    metrics.update(grounded.as_dict("mean_grounded"))
    metrics.update(specific.as_dict("mean_specific"))
    metrics.update(theme_recall.as_dict("theme_recall"))
    metrics.update(fabrication.as_dict("fabrication_rate"))
    metrics["structural_issues"] = structural_failures
    metrics["judge_errors"] = judge_errors

    return SuiteResult(
        suite="gap_analysis", dataset=dataset.name, model=model, judge_model=judge_id,
        metrics=metrics, cases=rows, started_at=started_at,
        duration_s=round(time.monotonic() - start, 1),
    )


SUITES = {
    "relevance": run_relevance,
    "gap_matching": run_gap_matching,
    "gap_analysis": run_gap_analysis,
}
