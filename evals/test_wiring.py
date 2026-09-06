"""End-to-end tests of the suites with a stubbed model. No API key, no network, no cost.

These prove the plumbing — that the suites drive the real screening, matching and agent
code, and that a perfect run scores as a perfect run. Whether the actual model is any
good is what `test_suites.py` measures.
"""
import pytest
from aiolimiter import AsyncLimiter

from modules.models import (
    CriticResult, GapMatchResult, IdentifiedGap, RelevanceBatch, RelevanceVerdict, SynthesisResult,
)

from evals import cases as datasets
from evals import judge as judge_module
from evals import suites
from evals.judge import CoverageReport, DistractorVerdict, GapGrade, GapGradeBatch, ThemeVerdict


@pytest.fixture
def limiter():
    return AsyncLimiter(1000, 60)


def _install(monkeypatch, handler):
    """Replaces the single LLM entry point everywhere the suites reach it."""
    async def fake(client, model_id, prompt, response_model, system_instruction, pdf_path=None):
        return handler(response_model, prompt, model_id)

    for module in ("modules.screening", "evals.judge", "evals.suites"):
        monkeypatch.setattr(f"{module}.generate_structured", fake)
    return fake


async def test_relevance_suite_scores_a_perfect_screener(monkeypatch, limiter):
    dataset = datasets.load_relevance()
    truth = {case.title: case.relevant for case in dataset.cases}

    def handler(response_model, prompt, model_id):
        # Recover the batch order from the prompt the screener built
        titles = [line.split("Title: ", 1)[1] for line in prompt.splitlines() if " Title: " in line]
        return RelevanceBatch(verdicts=[
            RelevanceVerdict(paper_index=i, relevant=truth[title], score=0.95, reason="stub")
            for i, title in enumerate(titles)
        ])

    _install(monkeypatch, handler)
    result = await suites.run_relevance(None, "stub", limiter)

    assert result.metrics["f1"] == 1.0
    assert result.metrics["hard_negative_rejection_rate"] == 1.0
    assert result.metrics["no_verdict"] == 0
    assert len(result.cases) == len(dataset.cases)
    # Confidence is reported per direction, not pooled across kept and rejected papers
    assert result.metrics["mean_confidence_kept_n"] == sum(c.relevant for c in dataset.cases)
    assert result.metrics["mean_confidence_rejected_n"] == sum(not c.relevant for c in dataset.cases)


async def test_relevance_suite_treats_a_missing_verdict_as_a_rejection(monkeypatch, limiter):
    _install(monkeypatch, lambda response_model, prompt, model_id: RelevanceBatch(verdicts=[]))
    result = await suites.run_relevance(None, "stub", limiter)

    assert result.metrics["no_verdict"] == len(result.cases)
    assert result.metrics["tp"] == 0 and result.metrics["fp"] == 0
    assert all(row["predicted"] is False for row in result.cases)


async def test_relevance_suite_applies_the_confidence_threshold(monkeypatch, limiter):
    def handler(response_model, prompt, model_id):
        titles = [line.split("Title: ", 1)[1] for line in prompt.splitlines() if " Title: " in line]
        return RelevanceBatch(verdicts=[
            RelevanceVerdict(paper_index=i, relevant=True, score=0.5, reason="hedged")
            for i in range(len(titles))
        ])

    _install(monkeypatch, handler)
    result = await suites.run_relevance(None, "stub", limiter, threshold=0.6)

    # Relevant but under-confident is a rejection, exactly as in the watcher
    assert all(row["predicted"] is False for row in result.cases)


async def test_gap_matching_suite_scores_a_perfect_matcher(monkeypatch, limiter):
    dataset = datasets.load_gap_matching()
    expected = {case.title: case.expected for case in dataset.cases}

    def handler(response_model, prompt, model_id):
        title = next(line.split("Title: ", 1)[1].strip() for line in prompt.splitlines() if "Title: " in line)
        return GapMatchResult(matches=[
            {"gap_id": e.gap_id, "relationship": e.relationships[0], "confidence": 0.9, "evidence": "stub"}
            for e in expected[title]
        ])

    _install(monkeypatch, handler)
    result = await suites.run_gap_matching(None, "stub", limiter)

    assert result.metrics["f1"] == 1.0
    assert result.metrics["relationship_accuracy"] == 1.0
    assert result.metrics["silence_on_unrelated"] == 1.0


async def test_gap_matching_suite_drops_low_confidence_matches_before_scoring(monkeypatch, limiter):
    dataset = datasets.load_gap_matching()
    expected = {case.title: case.expected for case in dataset.cases}

    def handler(response_model, prompt, model_id):
        title = next(line.split("Title: ", 1)[1].strip() for line in prompt.splitlines() if "Title: " in line)
        return GapMatchResult(matches=[
            {"gap_id": e.gap_id, "relationship": e.relationships[0], "confidence": 0.4, "evidence": "stub"}
            for e in expected[title]
        ])

    _install(monkeypatch, handler)
    result = await suites.run_gap_matching(None, "stub", limiter, confidence_threshold=0.75)

    assert result.metrics["tp"] == 0
    assert result.metrics["recall"] == 0.0
    assert any(row["dropped_below_threshold"] for row in result.cases)


async def test_gap_analysis_suite_reports_judge_scores_and_structural_issues(monkeypatch, limiter):
    def handler(response_model, prompt, model_id):
        if response_model is SynthesisResult:
            return SynthesisResult(narrative="n", dominant_methodologies="m")
        if response_model is CriticResult:
            return CriticResult(
                unexplored_territories="u", methodological_limitations="m", contradictions="c",
                gaps=[
                    IdentifiedGap(title="Single isolate reliance", category="methodological_limitation",
                                  description="Every study uses Fola-14."),
                    # Constructed unvalidated: the suite's structural check is the last line of defence
                    IdentifiedGap.model_construct(title="Vague", category="not_a_category",
                                                  description="More work is needed."),
                ],
            )
        if response_model is GapGradeBatch:
            return GapGradeBatch(grades=[
                GapGrade(gap_index=0, grounded=5, specific=5, comment="good"),
                GapGrade(gap_index=1, grounded=3, specific=1, comment="generic"),
            ])
        if response_model is CoverageReport:
            return CoverageReport(
                themes=[ThemeVerdict(theme_index=0, covered=True, covering_gap="Single isolate reliance")],
                distractors=[DistractorVerdict(claim_index=0, asserted=True, quote="field trials showed")],
            )
        raise AssertionError(f"unexpected response model {response_model}")

    _install(monkeypatch, handler)
    result = await suites.run_gap_analysis(None, "stub", limiter, limit=1, judge_model_id="stub-judge")

    assert result.judge_model == "stub-judge"
    assert result.metrics["mean_grounded"] == 4.0
    assert result.metrics["mean_specific"] == 3.0
    assert result.metrics["structural_issues"] == 1  # the invalid category
    assert result.metrics["fabrication_rate"] > 0
    assert result.cases[0]["fabrications"][0]["quote"] == "field trials showed"

    # Only theme 0 was reported as covered, and the case has more themes than that
    case = datasets.load_gap_analysis().cases[0]
    assert result.metrics["theme_recall"] == pytest.approx(1 / len(case.expected_themes))


async def test_gap_analysis_suite_survives_a_critic_that_returns_no_gaps(monkeypatch, limiter):
    def handler(response_model, prompt, model_id):
        if response_model is SynthesisResult:
            return SynthesisResult(narrative="n", dominant_methodologies="m")
        if response_model is CriticResult:
            return CriticResult(unexplored_territories="u", methodological_limitations="m",
                                contradictions="c", gaps=[])
        return CoverageReport(themes=[], distractors=[])

    _install(monkeypatch, handler)
    result = await suites.run_gap_analysis(None, "stub", limiter, limit=1)

    assert result.metrics["structural_issues"] == 1
    assert result.metrics["mean_grounded_n"] == 0
    assert result.metrics["theme_recall"] == 0.0


def test_judge_model_is_overridable_by_environment(monkeypatch):
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    assert judge_module.judge_model() == judge_module.DEFAULT_JUDGE_MODEL
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "openai/gpt-5")
    assert judge_module.judge_model() == "openai/gpt-5"


async def test_gap_analysis_suite_counts_a_failed_judge_instead_of_scoring_it_zero(monkeypatch, limiter):
    """A judge refused by a provider content filter must not look like a bad model."""
    calls = {"judge": 0}

    def handler(response_model, prompt, model_id):
        if response_model is SynthesisResult:
            return SynthesisResult(narrative="n", dominant_methodologies="m")
        if response_model is CriticResult:
            return CriticResult(
                unexplored_territories="u", methodological_limitations="m", contradictions="c",
                gaps=[IdentifiedGap(title="Single isolate reliance", category="contradiction",
                                    description="Every study uses Fola-14.")],
            )
        calls["judge"] += 1
        raise RuntimeError("OpenRouter returned an empty message (finish_reason: refusal).")

    _install(monkeypatch, handler)
    result = await suites.run_gap_analysis(None, "stub", limiter, limit=1)

    assert calls["judge"] == 2 * suites.JUDGE_ATTEMPTS  # both judge calls, each retried
    assert result.metrics["judge_errors"] == 2
    assert result.metrics["theme_recall_n"] == 0  # excluded, not counted as zero
    assert result.metrics["mean_grounded_n"] == 0
    assert result.cases[0]["judge_failed"] is True
    assert result.cases[0]["themes"] == [] and result.cases[0]["fabrications"] == []


async def test_gap_analysis_suite_retries_a_judge_that_recovers(monkeypatch, limiter):
    attempts = {"n": 0}

    def handler(response_model, prompt, model_id):
        if response_model is SynthesisResult:
            return SynthesisResult(narrative="n", dominant_methodologies="m")
        if response_model is CriticResult:
            return CriticResult(unexplored_territories="u", methodological_limitations="m",
                                contradictions="c",
                                gaps=[IdentifiedGap(title="t", category="contradiction", description="d")])
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("content filter")
        if response_model is GapGradeBatch:
            return GapGradeBatch(grades=[GapGrade(gap_index=0, grounded=4, specific=4, comment="ok")])
        return CoverageReport(themes=[ThemeVerdict(theme_index=0, covered=True, covering_gap="t")],
                              distractors=[])

    _install(monkeypatch, handler)
    result = await suites.run_gap_analysis(None, "stub", limiter, limit=1)

    assert result.metrics["judge_errors"] == 0
    assert result.metrics["mean_grounded"] == 4.0
