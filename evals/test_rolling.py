"""Offline tests for rolling gap synthesis. No API key, no network, no cost.

The expensive part of this feature is a judgement call an LLM makes (is this candidate
gap the same as one we already track?). What is testable without the model is everything
around it: that papers reach the right subject, that the trigger holds work back until
enough has accumulated, that a duplicate updates rather than inserts, and that a paper is
never offered its own gap to fill.
"""
import json

import pytest
from aiolimiter import AsyncLimiter

from modules import db, rolling
from modules.config import SubjectGroup, WatchConfig, route_to_subjects
from modules.models import (
    CriticResult, GapMergeDecision, GapMergeResult, IdentifiedGap, SynthesisResult, WatchedPaper,
)


@pytest.fixture
def limiter():
    return AsyncLimiter(1000, 60)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    return db


SUBJECTS = [
    SubjectGroup(name="rust genomics", terms=["Puccinia", "yellow rust"]),
    SubjectGroup(name="soft fruit genomics", terms=["Rubus", "Fragaria"]),
]


def _paper(paper_id: str, terms, title="A paper", abstract="Findings."):
    return {
        "paper_id": paper_id, "title": title, "abstract": abstract, "authors": "A. Author",
        "published": "2026-09-01", "url": "http://example.invalid", "source": "pubmed",
        "matched_terms": json.dumps(terms), "relevance_reason": "on topic",
    }


# --- Routing ---

def test_a_species_routes_into_a_group_declaring_the_genus():
    assert route_to_subjects(["Puccinia striiformis"], SUBJECTS) == ["rust genomics"]


def test_a_broad_match_does_not_route_into_a_narrower_group():
    """A paper matched only on the bare species cannot be claimed by a forma specialis
    group: that is what put banana, maize and soybean papers in one bucket on the first
    real run, and produced gaps general enough to fit all three."""
    narrow = [SubjectGroup(name="strawberry wilt", terms=["Fusarium oxysporum f. sp. fragariae"])]
    assert route_to_subjects(["Fusarium oxysporum"], narrow) == []
    assert route_to_subjects(["Fusarium oxysporum f. sp. fragariae"], narrow) == ["strawberry wilt"]


def test_a_paper_can_belong_to_two_subjects():
    assert route_to_subjects(["Puccinia", "Rubus"], SUBJECTS) == ["rust genomics", "soft fruit genomics"]


def test_an_unmatched_paper_is_left_out_rather_than_forced_into_a_group():
    assert route_to_subjects(["Zymoseptoria tritici"], SUBJECTS) == []


def test_grouping_survives_matched_terms_that_are_not_valid_json():
    paper = _paper("pubmed:1", ["Puccinia"])
    paper["matched_terms"] = "not json"
    assert rolling.group_backlog([paper], SUBJECTS) == {"rust genomics": [], "soft fruit genomics": []}


# --- Trigger ---

async def test_a_subject_below_the_trigger_is_left_banked(temp_db, limiter, monkeypatch):
    await db.init_db()
    await db.store_papers([
        WatchedPaper(paper_id=f"pubmed:{i}", source="pubmed", title=f"Paper {i}",
                     abstract="Findings.", matched_terms=["Puccinia"])
        for i in range(3)
    ])
    for i in range(3):
        await db.record_screening(f"pubmed:{i}", True, 0.9, "on topic")

    called = False

    async def fail(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(rolling, "synthesise_subject", fail)
    config = WatchConfig(subjects=SUBJECTS, rolling_min_papers=12)

    assert await rolling.run_rolling_synthesis(None, config, limiter) == (0, 0)
    assert not called
    assert len(await db.get_synthesis_backlog()) == 3


async def test_force_synthesises_a_backlog_below_the_trigger(temp_db, limiter, monkeypatch):
    await db.init_db()
    await db.store_papers([WatchedPaper(paper_id="pubmed:1", source="pubmed", title="Paper",
                                        abstract="Findings.", matched_terms=["Puccinia"])])
    await db.record_screening("pubmed:1", True, 0.9, "on topic")

    seen = []

    async def record(client, model_id, subject, papers, limiter):
        seen.append((subject, len(papers)))
        return 2, 0

    monkeypatch.setattr(rolling, "synthesise_subject", record)
    config = WatchConfig(subjects=SUBJECTS, rolling_min_papers=12)

    assert await rolling.run_rolling_synthesis(None, config, limiter, force=True) == (2, 0)
    assert seen == [("rust genomics", 1)]


async def test_one_failing_subject_does_not_stop_the_others(temp_db, limiter, monkeypatch):
    await db.init_db()
    papers = [
        WatchedPaper(paper_id="pubmed:1", source="pubmed", title="Rust", abstract="a", matched_terms=["Puccinia"]),
        WatchedPaper(paper_id="pubmed:2", source="pubmed", title="Fruit", abstract="b", matched_terms=["Rubus"]),
    ]
    await db.store_papers(papers)
    for paper in papers:
        await db.record_screening(paper.paper_id, True, 0.9, "on topic")

    async def flaky(client, model_id, subject, papers, limiter):
        if subject == "rust genomics":
            raise RuntimeError("provider refused")
        await db.mark_synthesised([p["paper_id"] for p in papers], subject)
        return 1, 0

    monkeypatch.setattr(rolling, "synthesise_subject", flaky)
    config = WatchConfig(subjects=SUBJECTS, rolling_min_papers=1)

    assert await rolling.run_rolling_synthesis(None, config, limiter) == (1, 0)
    # The failed subject's paper is unconsumed, so the next run retries it; the successful
    # subject's paper is retired because every subject it routes to has now used it.
    consumed = await db.get_synthesised_subjects()
    assert "pubmed:1" not in consumed
    assert consumed["pubmed:2"] == {"soft fruit genomics"}
    assert [p["paper_id"] for p in await db.get_synthesis_backlog()] == ["pubmed:1"]


async def test_the_backlog_fed_to_one_synthesis_is_capped(temp_db, limiter, monkeypatch):
    await db.init_db()
    papers = [
        WatchedPaper(paper_id=f"pubmed:{i}", source="pubmed", title=f"P{i}",
                     abstract="a", matched_terms=["Puccinia"])
        for i in range(10)
    ]
    await db.store_papers(papers)
    for paper in papers:
        await db.record_screening(paper.paper_id, True, 0.9, "on topic")

    sizes = []

    async def record(client, model_id, subject, papers, limiter):
        sizes.append(len(papers))
        return 0, 0

    monkeypatch.setattr(rolling, "synthesise_subject", record)
    config = WatchConfig(subjects=SUBJECTS, rolling_min_papers=1, rolling_max_papers=4)

    await rolling.run_rolling_synthesis(None, config, limiter)
    assert sizes == [4]


# --- Synthesis, merging and provenance ---

def _install(monkeypatch, handler):
    async def fake(client, model_id, prompt, response_model, system_instruction, pdf_path=None):
        return handler(response_model, prompt)

    for module in ("modules.agents", "modules.rolling"):
        monkeypatch.setattr(f"{module}.generate_structured", fake, raising=False)
    monkeypatch.setattr("modules.rolling.generate_structured", fake)


CANDIDATE = IdentifiedGap(
    title="Effector presence/absence variation across Puccinia striiformis races is unquantified",
    category="unexplored_territory",
    description="No study reports PAV of candidate effectors across field races.",
)


def _agent_handler(candidates, merge_result=None):
    def handler(response_model, prompt):
        if response_model is SynthesisResult:
            return SynthesisResult(narrative="n", dominant_methodologies="m")
        if response_model is CriticResult:
            return CriticResult(unexplored_territories="u", methodological_limitations="m",
                                contradictions="c", gaps=candidates)
        if response_model is GapMergeResult:
            return merge_result or GapMergeResult(decisions=[])
        raise AssertionError(f"unexpected model {response_model}")
    return handler


async def test_a_new_gap_is_stored_with_its_abstract_origin_and_sources(temp_db, limiter, monkeypatch):
    await db.init_db()
    _install(monkeypatch, _agent_handler([CANDIDATE]))

    new, merged = await rolling.synthesise_subject(
        None, "model", "rust genomics", [_paper("pubmed:1", ["Puccinia"])], limiter
    )
    assert (new, merged) == (1, 0)

    gaps = await db.get_gaps(status="open")
    assert len(gaps) == 1
    assert gaps[0]["origin"] == "abstract"
    assert await db.get_gap_sources() == {gaps[0]["gap_id"]: {"pubmed:1"}}
    # Consumed for this subject, so a later run does not synthesise it again here.
    assert await db.get_synthesised_subjects() == {"pubmed:1": {"rust genomics"}}


async def test_a_restated_gap_updates_the_stored_one_instead_of_adding_a_row(temp_db, limiter, monkeypatch):
    await db.init_db()
    _install(monkeypatch, _agent_handler([CANDIDATE]))
    await rolling.synthesise_subject(
        None, "model", "rust genomics", [_paper("pubmed:1", ["Puccinia"])], limiter
    )
    stored_id = (await db.get_gaps(status="open"))[0]["gap_id"]

    reworded = IdentifiedGap(
        title="Nobody has measured effector PAV between yellow rust field isolates",
        category="unexplored_territory",
        description="The same gap, said differently.",
    )
    merge = GapMergeResult(decisions=[GapMergeDecision(
        new_gap_index=0, duplicate_of=stored_id,
        merged_description="Effector PAV across P. striiformis field races is unquantified.",
        reason="same question",
    )])
    _install(monkeypatch, _agent_handler([reworded], merge))

    new, merged = await rolling.synthesise_subject(
        None, "model", "rust genomics", [_paper("pubmed:2", ["Puccinia"])], limiter
    )
    assert (new, merged) == (0, 1)

    gaps = await db.get_gaps(status="open")
    assert len(gaps) == 1
    assert gaps[0]["description"] == "Effector PAV across P. striiformis field races is unquantified."
    # Both papers are now credited as sources, so neither can later fill this gap.
    assert await db.get_gap_sources() == {stored_id: {"pubmed:1", "pubmed:2"}}


async def test_a_merge_naming_an_unknown_gap_id_stores_the_candidate_as_new(temp_db, limiter, monkeypatch):
    await db.init_db()
    _install(monkeypatch, _agent_handler([CANDIDATE]))
    await rolling.synthesise_subject(
        None, "model", "rust genomics", [_paper("pubmed:1", ["Puccinia"])], limiter
    )

    other = IdentifiedGap(title="A different gap entirely", category="contradiction", description="d")
    merge = GapMergeResult(decisions=[GapMergeDecision(
        new_gap_index=0, duplicate_of="deadbeefdeadbeef", merged_description="x", reason="hallucinated",
    )])
    _install(monkeypatch, _agent_handler([other], merge))

    new, merged = await rolling.synthesise_subject(
        None, "model", "rust genomics", [_paper("pubmed:2", ["Puccinia"])], limiter
    )
    assert (new, merged) == (1, 0)
    assert len(await db.get_gaps(status="open")) == 2


async def test_a_critic_returning_no_gaps_still_consumes_the_backlog(temp_db, limiter, monkeypatch):
    await db.init_db()
    await db.store_papers([WatchedPaper(paper_id="pubmed:1", source="pubmed", title="P",
                                        abstract="a", matched_terms=["Puccinia"])])
    await db.record_screening("pubmed:1", True, 0.9, "on topic")
    _install(monkeypatch, _agent_handler([]))

    assert await rolling.synthesise_subject(
        None, "model", "rust genomics", [_paper("pubmed:1", ["Puccinia"])], limiter
    ) == (0, 0)
    assert await db.get_synthesised_subjects() == {"pubmed:1": {"rust genomics"}}


async def test_the_merge_step_is_skipped_when_nothing_is_stored_yet(temp_db, limiter, monkeypatch):
    """The first synthesis for a subject has nothing to deduplicate against."""
    seen = []

    def handler(response_model, prompt):
        seen.append(response_model)
        if response_model is SynthesisResult:
            return SynthesisResult(narrative="n", dominant_methodologies="m")
        return CriticResult(unexplored_territories="u", methodological_limitations="m",
                            contradictions="c", gaps=[CANDIDATE])

    await db.init_db()
    _install(monkeypatch, handler)
    await rolling.synthesise_subject(
        None, "model", "rust genomics", [_paper("pubmed:1", ["Puccinia"])], limiter
    )
    assert GapMergeResult not in seen


async def test_the_source_note_tells_both_agents_they_are_reading_abstracts(temp_db, limiter, monkeypatch):
    prompts = []

    def handler(response_model, prompt):
        prompts.append(prompt)
        if response_model is SynthesisResult:
            return SynthesisResult(narrative="n", dominant_methodologies="m")
        return CriticResult(unexplored_territories="u", methodological_limitations="m",
                            contradictions="c", gaps=[])

    await db.init_db()
    _install(monkeypatch, handler)
    await rolling.synthesise_subject(
        None, "model", "rust genomics", [_paper("pubmed:1", ["Puccinia"])], limiter
    )
    assert len(prompts) == 2
    assert all("not full papers" in prompt for prompt in prompts)


# --- Self-reference ---

async def test_a_gaps_own_source_papers_are_excluded_from_matching(temp_db, limiter, monkeypatch):
    await db.init_db()
    _install(monkeypatch, _agent_handler([CANDIDATE]))
    await rolling.synthesise_subject(
        None, "model", "rust genomics", [_paper("pubmed:1", ["Puccinia"])], limiter
    )
    gap_id = (await db.get_gaps(status="open"))[0]["gap_id"]

    sources = await db.get_gap_sources([gap_id])
    gaps = await db.get_gaps(status="open")

    # The filter run_watch applies before calling the matcher.
    for paper_id, expected in (("pubmed:1", 0), ("pubmed:99", 1)):
        visible = [gap for gap in gaps if paper_id not in sources.get(gap["gap_id"], ())]
        assert len(visible) == expected


# --- run_watch ordering ---

async def test_run_watch_matches_before_it_synthesises(temp_db, monkeypatch, tmp_path):
    """Order is the whole self-reference guard: gaps minted from this run's papers must
    not be offered back to those same papers as something to fill."""
    from modules import watch as watch_module

    paper = WatchedPaper(paper_id="pubmed:1", source="pubmed", title="Rust effectors",
                         abstract="Findings.", published="2026-09-01", matched_terms=["Puccinia"])

    async def fake_fetch_all(**kwargs):
        return [paper]

    monkeypatch.setattr(watch_module.sources, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(watch_module, "get_client", lambda: None)

    calls = []

    async def fake_screen(client, model, papers, interests, organisms, limiter, batch_size=10):
        calls.append("screen")
        from modules.models import RelevanceVerdict
        return {p.paper_id: RelevanceVerdict(paper_index=i, relevant=True, score=0.95, reason="on topic")
                for i, p in enumerate(papers)}

    gaps_seen = []

    async def fake_match(client, model, paper, gaps, limiter):
        calls.append("match")
        gaps_seen.append([gap["gap_id"] for gap in gaps])
        from modules.models import GapMatchResult
        return GapMatchResult(matches=[])

    async def fake_rolling(client, config, limiter, force=False):
        calls.append("rolling")
        return 0, 0

    monkeypatch.setattr(watch_module, "screen_relevance", fake_screen)
    monkeypatch.setattr(watch_module, "match_paper_to_gaps", fake_match)
    monkeypatch.setattr(watch_module.rolling, "run_rolling_synthesis", fake_rolling)

    config = WatchConfig(subjects=SUBJECTS, delivery=[])
    await watch_module.run_watch(config, since="7d", digest_path=str(tmp_path / "d.md"), deliver=False)

    assert calls == ["screen", "match", "rolling"]
    assert gaps_seen == [[]]


async def test_run_watch_hides_a_gap_from_the_paper_that_created_it(temp_db, monkeypatch, tmp_path):
    from modules import watch as watch_module

    # A gap already stored, sourced from the very paper this run will screen.
    await db.init_db()
    await db.store_gaps("rust genomics", [CANDIDATE], origin="abstract", source_papers=["pubmed:1"])
    await db.store_gaps("rust genomics", [IdentifiedGap(
        title="An unrelated open question", category="contradiction", description="d")])
    own_gap = db.make_gap_id("rust genomics", CANDIDATE.title)

    paper = WatchedPaper(paper_id="pubmed:1", source="pubmed", title="Rust effectors",
                         abstract="Findings.", published="2026-09-01", matched_terms=["Puccinia"])

    async def fake_fetch_all(**kwargs):
        return [paper]

    async def fake_screen(client, model, papers, interests, organisms, limiter, batch_size=10):
        from modules.models import RelevanceVerdict
        return {papers[0].paper_id: RelevanceVerdict(paper_index=0, relevant=True, score=0.95, reason="r")}

    offered = []

    async def fake_match(client, model, paper, gaps, limiter):
        offered.extend(gap["gap_id"] for gap in gaps)
        from modules.models import GapMatchResult
        return GapMatchResult(matches=[])

    async def fake_rolling(client, config, limiter, force=False):
        return 0, 0

    monkeypatch.setattr(watch_module.sources, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(watch_module, "get_client", lambda: None)
    monkeypatch.setattr(watch_module, "screen_relevance", fake_screen)
    monkeypatch.setattr(watch_module, "match_paper_to_gaps", fake_match)
    monkeypatch.setattr(watch_module.rolling, "run_rolling_synthesis", fake_rolling)

    config = WatchConfig(subjects=SUBJECTS, delivery=[])
    await watch_module.run_watch(config, since="7d", digest_path=str(tmp_path / "d.md"), deliver=False)

    assert own_gap not in offered
    assert len(offered) == 1


async def test_overlapping_subjects_each_get_the_paper(temp_db, limiter, monkeypatch):
    """A species-complex group and a forma specialis group both cover the same paper.
    Consumption is per subject, so whichever fires first must not steal it from the other."""
    await db.init_db()
    overlapping = [
        SubjectGroup(name="species complex", terms=["Fusarium oxysporum"]),
        SubjectGroup(name="strawberry wilt", terms=["Fusarium oxysporum f. sp. fragariae"]),
    ]
    paper = _paper("pubmed:1", ["Fusarium oxysporum f. sp. fragariae"])
    assert route_to_subjects(["Fusarium oxysporum f. sp. fragariae"], overlapping) == [
        "species complex", "strawberry wilt",
    ]

    await db.mark_synthesised(["pubmed:1"], "species complex")
    consumed = await db.get_synthesised_subjects(["pubmed:1"])
    grouped = rolling.group_backlog([paper], overlapping, consumed)

    assert grouped["species complex"] == []
    assert [p["paper_id"] for p in grouped["strawberry wilt"]] == ["pubmed:1"]


async def test_a_paper_routing_nowhere_is_never_retired(temp_db, limiter, monkeypatch):
    """It may be covered by a subject group declared later, so it stays visible."""
    await db.init_db()
    unrouted = _paper("pubmed:1", ["Zymoseptoria tritici"])
    await rolling._retire_consumed_papers([unrouted], SUBJECTS)
    assert await db.get_synthesised_subjects() == {}


async def test_re_deriving_watch_terms_lets_an_old_paper_reach_a_new_subject(temp_db):
    """matched_terms is frozen at fetch time, so a term added to the watchlist later would
    otherwise never appear against papers already stored, and routing reads that field."""
    await db.init_db()
    await db.store_papers([WatchedPaper(
        paper_id="pubmed:1", source="pubmed", title="Poplar rust assembly",
        abstract="Melampsora larici-populina haplotypes.", matched_terms=[])])

    rust = [SubjectGroup(name="rust genomics", terms=["Melampsora"])]
    assert rolling.group_backlog([_paper("pubmed:1", [])], rust) == {"rust genomics": []}

    await db.update_matched_terms([("pubmed:1", ["Melampsora"])])
    refreshed = _paper("pubmed:1", ["Melampsora"])
    assert [p["paper_id"] for p in rolling.group_backlog([refreshed], rust)["rust genomics"]] == ["pubmed:1"]
