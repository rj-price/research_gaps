"""Dataset models for the eval suites.

Every case file lives in `evals/datasets/` and is validated on load, so a malformed
dataset fails immediately rather than halfway through a paid run.
"""
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from modules.models import WatchedPaper

DATASET_DIR = Path(__file__).parent / "datasets"


# --- Relevance screening ---

class RelevanceCase(BaseModel):
    paper_id: str
    title: str
    abstract: str
    relevant: bool = Field(description="Ground truth: should the screener keep this paper?")
    tags: List[str] = Field(default_factory=list)

    def to_paper(self) -> WatchedPaper:
        return WatchedPaper(
            paper_id=self.paper_id,
            source="eval",
            title=self.title,
            abstract=self.abstract,
            published="2026-01-01",
        )


class RelevanceDataset(BaseModel):
    name: str
    note: str = ""
    organisms: List[str]
    interests: str
    cases: List[RelevanceCase]


# --- Gap matching ---

class GapFixture(BaseModel):
    gap_id: str
    subject: str
    category: str
    title: str
    description: str

    def as_row(self) -> dict:
        """The shape `match_paper_to_gaps` expects from the database."""
        return self.model_dump()


class ExpectedMatch(BaseModel):
    gap_id: str
    relationships: List[str] = Field(
        description="Any of these relationship labels counts as correct for this match.",
    )


class GapMatchCase(BaseModel):
    paper_id: str
    title: str
    abstract: str
    published: str = "2026-01-01"
    expected: List[ExpectedMatch] = Field(default_factory=list)
    allowed: List[str] = Field(
        default_factory=list,
        description="Gap IDs that are defensible but not required: neither credited nor penalised.",
    )

    def to_paper(self) -> WatchedPaper:
        return WatchedPaper(
            paper_id=self.paper_id,
            source="eval",
            title=self.title,
            abstract=self.abstract,
            published=self.published,
        )

    @property
    def expected_ids(self) -> set:
        return {match.gap_id for match in self.expected}


class GapMatchDataset(BaseModel):
    name: str
    note: str = ""
    gaps: List[GapFixture]
    cases: List[GapMatchCase]


# --- Gap analysis (LLM-as-judge) ---

class GapAnalysisCase(BaseModel):
    case_id: str
    subject: str
    summaries: List[str]
    expected_themes: List[str] = Field(
        description="Gaps genuinely present in the summaries that a competent critic should surface.",
    )
    distractor_claims: List[str] = Field(
        default_factory=list,
        description="Statements unsupported by the summaries; a grounded critic never asserts these.",
    )


class GapAnalysisDataset(BaseModel):
    name: str
    note: str = ""
    cases: List[GapAnalysisCase]


def _load(model, filename: str):
    path = DATASET_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {path}")
    return model.model_validate_json(path.read_text())


def load_relevance() -> RelevanceDataset:
    return _load(RelevanceDataset, "relevance.json")


def load_gap_matching() -> GapMatchDataset:
    return _load(GapMatchDataset, "gap_matching.json")


def load_gap_analysis() -> GapAnalysisDataset:
    return _load(GapAnalysisDataset, "gap_analysis.json")


def load_gap_analysis_abstracts() -> GapAnalysisDataset:
    """The same fixtures as abstracts: what the rolling watcher actually feeds the Critic."""
    return _load(GapAnalysisDataset, "gap_analysis_abstracts.json")
