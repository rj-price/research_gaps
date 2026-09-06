"""Offline tests for the harness itself. No API key, no network, no cost."""
import pytest

from evals import cases as datasets
from evals.metrics import BinaryScore, Mean, SetScore
from evals.runner import check_thresholds, load_thresholds
from evals.suites import SUITES, VALID_CATEGORIES, SuiteResult, _structural_issues
from modules.models import CriticResult, IdentifiedGap


def test_binary_score_counts_each_quadrant():
    score = BinaryScore()
    for predicted, actual in [(True, True), (True, False), (False, True), (False, False)]:
        score.add(predicted, actual)
    assert (score.tp, score.fp, score.fn, score.tn) == (1, 1, 1, 1)
    assert score.accuracy == 0.5
    assert score.precision == 0.5
    assert score.recall == 0.5
    assert score.f1 == 0.5


def test_binary_score_is_zero_rather_than_undefined_when_nothing_is_predicted():
    score = BinaryScore()
    score.add(False, True)
    assert score.precision == 0.0
    assert score.f1 == 0.0


def test_set_score_ignores_allowed_extras_but_still_credits_expected():
    score = SetScore()
    score.add(predicted={"a", "b"}, expected={"a"}, allowed={"b"})
    assert (score.tp, score.fp, score.fn, score.ignored) == (1, 0, 0, 1)
    assert score.precision == 1.0


def test_set_score_penalises_a_link_that_is_neither_expected_nor_allowed():
    score = SetScore()
    score.add(predicted={"a", "z"}, expected={"a", "b"}, allowed={"c"})
    assert (score.tp, score.fp, score.fn) == (1, 1, 1)


def test_set_score_allowed_never_masks_an_expected_gap():
    # A gap listed as both expected and allowed must still count, not be ignored
    score = SetScore()
    score.add(predicted={"a"}, expected={"a"}, allowed={"a"})
    assert (score.tp, score.ignored) == (1, 0)


def test_mean_reports_its_sample_count():
    mean = Mean()
    for value in (4, 5, 3):
        mean.add(value)
    assert mean.as_dict("grounded") == {"grounded": 4.0, "grounded_n": 3}


def _critic(**overrides) -> CriticResult:
    defaults = dict(
        unexplored_territories="x", methodological_limitations="y", contradictions="z",
        gaps=[IdentifiedGap(title="A short title", category="contradiction", description="A description.")],
    )
    return CriticResult(**{**defaults, **overrides})


def test_structural_issues_passes_a_well_formed_critic_result():
    assert _structural_issues(_critic()) == []


@pytest.mark.parametrize("gap, fragment", [
    (IdentifiedGap.model_construct(title="t", category="not_a_category", description="d"), "invalid category"),
    (IdentifiedGap(title=" ".join(["word"] * 16), category="contradiction", description="d"), "over 15 words"),
    (IdentifiedGap(title="t", category="contradiction", description="  "), "empty description"),
])
def test_structural_issues_flags_malformed_gaps(gap, fragment):
    issues = _structural_issues(_critic(gaps=[gap]))
    assert any(fragment in issue for issue in issues)


def test_structural_issues_flags_an_empty_gap_list():
    assert "no discrete gaps returned" in _structural_issues(_critic(gaps=[]))[0]


def test_thresholds_cover_every_suite_and_only_known_directions():
    thresholds = load_thresholds()
    # Tied to SUITES rather than a hardcoded list: a new suite with no thresholds would
    # otherwise run in CI and pass unconditionally.
    assert set(thresholds) == set(SUITES)
    for bounds in thresholds.values():
        for bound in bounds.values():
            assert set(bound) <= {"min", "max"} and bound


def test_check_thresholds_reports_breaches_in_both_directions():
    thresholds = {"relevance": {"f1": {"min": 0.9}, "no_verdict": {"max": 0}}}
    result = SuiteResult(suite="relevance", dataset="d", model="m", metrics={"f1": 0.5, "no_verdict": 2})
    breaches = check_thresholds(result, thresholds)
    assert len(breaches) == 2
    assert "below minimum" in breaches[0] and "above maximum" in breaches[1]


def test_check_thresholds_flags_a_metric_the_suite_never_reported():
    result = SuiteResult(suite="relevance", dataset="d", model="m", metrics={})
    assert "not reported" in check_thresholds(result, {"relevance": {"f1": {"min": 0.9}}})[0]


def test_check_thresholds_passes_a_result_that_clears_every_bound():
    thresholds = {"relevance": {"f1": {"min": 0.5}, "no_verdict": {"max": 1}}}
    result = SuiteResult(suite="relevance", dataset="d", model="m", metrics={"f1": 0.9, "no_verdict": 0})
    assert check_thresholds(result, thresholds) == []


# --- Dataset integrity ---

def test_relevance_dataset_has_both_labels_and_some_hard_negatives():
    dataset = datasets.load_relevance()
    labels = {case.relevant for case in dataset.cases}
    assert labels == {True, False}
    assert sum("hard-negative" in case.tags for case in dataset.cases) >= 3


def test_relevance_paper_ids_are_unique():
    ids = [case.paper_id for case in datasets.load_relevance().cases]
    assert len(ids) == len(set(ids))


def test_gap_matching_expectations_reference_real_gaps_and_valid_relationships():
    dataset = datasets.load_gap_matching()
    known = {gap.gap_id for gap in dataset.gaps}
    relationships = {"fills", "partially_addresses", "contradicts", "informs"}
    for case in dataset.cases:
        assert case.expected_ids <= known, f"{case.paper_id} expects an unknown gap"
        assert set(case.allowed) <= known, f"{case.paper_id} allows an unknown gap"
        for expectation in case.expected:
            assert expectation.relationships, f"{case.paper_id} lists no accepted relationship"
            assert set(expectation.relationships) <= relationships


def test_gap_matching_includes_papers_that_should_match_nothing():
    dataset = datasets.load_gap_matching()
    assert sum(not case.expected for case in dataset.cases) >= 2


def test_gap_matching_gap_categories_match_the_critic_schema():
    for gap in datasets.load_gap_matching().gaps:
        assert gap.category in VALID_CATEGORIES


def test_gap_analysis_cases_carry_themes_and_distractors():
    dataset = datasets.load_gap_analysis()
    assert dataset.cases
    for case in dataset.cases:
        assert case.summaries and case.expected_themes and case.distractor_claims
