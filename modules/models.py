from typing import List, Literal
from pydantic import BaseModel, Field, field_validator

class PaperSummary(BaseModel):
    title: str = Field(description="The title or topic of the paper (infer if not explicit).")
    core_research_question: str = Field(description="What problem does this paper solve?")
    methodology: str = Field(description="A brief overview of the methods used.")
    key_findings: str = Field(description="The main results of the paper.")
    limitations: str = Field(description="What did the authors explicitly state as limitations or future directions?")

    def to_markdown(self) -> str:
        return (
            f"1. **Title/Topic**: {self.title}\n"
            f"2. **Core Research Question**: {self.core_research_question}\n"
            f"3. **Methodology**: {self.methodology}\n"
            f"4. **Key Findings**: {self.key_findings}\n"
            f"5. **Limitations & Future Work**: {self.limitations}\n"
        )

# --- Multi-Agent Architecture Models ---

class SynthesisResult(BaseModel):
    narrative: str = Field(description="A cohesive narrative of what is currently known and established based on the papers.")
    dominant_methodologies: str = Field(description="The dominant methodologies and common themes synthesised.")

# Constrained rather than free text: these values are stored, matched against and
# displayed in the digest, so an enum in the JSON schema is what keeps them usable.
GapCategory = Literal["unexplored_territory", "methodological_limitation", "contradiction"]

_CATEGORY_ALIASES = {
    "unexplored_territories": "unexplored_territory",
    "methodological_limitations": "methodological_limitation",
    "contradictions": "contradiction",
    "contradictions_and_tensions": "contradiction",
}


class IdentifiedGap(BaseModel):
    title: str = Field(description="A short, self-contained title for this single research gap (max 15 words).")
    category: GapCategory = Field(description="One of: unexplored_territory, methodological_limitation, contradiction.")
    description: str = Field(description="A 2-3 sentence statement of the gap, specific enough that a new paper could be judged to fill it or not.")

    @field_validator("category", mode="before")
    @classmethod
    def _accept_the_plural_headings(cls, value):
        """Models echo the plural section headings they were shown; take them as the singular.

        Belt and braces behind the schema enum: a provider that ignores the enum would
        otherwise fail validation and burn the whole Critic step on a retry loop.
        """
        if isinstance(value, str):
            normalised = value.strip().lower().replace(" ", "_").replace("-", "_")
            return _CATEGORY_ALIASES.get(normalised, normalised)
        return value

class CriticResult(BaseModel):
    unexplored_territories: str = Field(description="Specific questions or variables consistently ignored or missing across papers.")
    methodological_limitations: str = Field(description="Widespread flaws, limitations, or technologies that should be applied.")
    contradictions: str = Field(description="Conflicting findings between the papers that need resolution.")
    gaps: List[IdentifiedGap] = Field(default_factory=list, description="The same gaps broken out as discrete, individually trackable items.")

class ResearchProposal(BaseModel):
    title: str = Field(description="A professional, academic title.")
    targeted_gap: str = Field(description="Which specific gap this proposal addresses.")
    methodology: str = Field(description="A brief 2-3 sentence overview of how this study would be conducted.")
    expected_impact: str = Field(description="Why solving this gap is important to the broader field.")

class InnovatorResult(BaseModel):
    proposals: List[ResearchProposal] = Field(description="3 novel, highly specific research studies.")


# --- Ambient Literature Watcher Models ---

class WatchedPaper(BaseModel):
    """A candidate paper retrieved from a literature source."""
    paper_id: str = Field(description="Stable identifier: 'pubmed:<pmid>' or 'biorxiv:<doi>'.")
    source: str
    title: str
    abstract: str = ""
    authors: str = ""
    published: str = ""
    url: str = ""
    matched_terms: List[str] = Field(default_factory=list)


class RelevanceVerdict(BaseModel):
    paper_index: int = Field(description="The index of the paper in the provided list.")
    relevant: bool = Field(description="True if the paper is genuinely about the watched organisms and research interests.")
    score: float = Field(description="Confidence in the relevance judgement, 0.0 to 1.0.")
    reason: str = Field(description="One sentence justifying the verdict.")


class RelevanceBatch(BaseModel):
    verdicts: List[RelevanceVerdict] = Field(description="One verdict per paper, in the order provided.")


class GapMatch(BaseModel):
    gap_id: str = Field(description="The exact gap_id from the supplied list of stored gaps.")
    relationship: str = Field(description="One of: fills, partially_addresses, contradicts, informs.")
    confidence: float = Field(description="Confidence in the link, 0.0 to 1.0.")
    evidence: str = Field(description="The specific finding in the paper that supports this link. Quote or closely paraphrase the abstract.")


class GapMatchResult(BaseModel):
    matches: List[GapMatch] = Field(description="Links from this paper to stored gaps. Empty if the paper addresses none of them.")


class GapMergeDecision(BaseModel):
    """Whether a freshly generated gap is genuinely new or a restatement of a stored one."""
    new_gap_index: int = Field(description="The index of the candidate gap in the provided list.")
    duplicate_of: str = Field(
        default="",
        description="The gap_id of the stored gap this restates, or an empty string if it is genuinely new.",
    )
    merged_description: str = Field(
        default="",
        description="When duplicate_of is set, a single description covering both statements of the gap. Empty otherwise.",
    )
    reason: str = Field(description="One sentence justifying the decision.")


class GapMergeResult(BaseModel):
    decisions: List[GapMergeDecision] = Field(description="One decision per candidate gap, in the order provided.")
