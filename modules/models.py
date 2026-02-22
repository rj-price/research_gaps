from typing import List
from pydantic import BaseModel, Field

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
    dominant_methodologies: str = Field(description="The dominant methodologies and common themes synthesized.")

class CriticResult(BaseModel):
    unexplored_territories: str = Field(description="Specific questions or variables consistently ignored or missing across papers.")
    methodological_limitations: str = Field(description="Widespread flaws, limitations, or technologies that should be applied.")
    contradictions: str = Field(description="Conflicting findings between the papers that need resolution.")

class ResearchProposal(BaseModel):
    title: str = Field(description="A professional, academic title.")
    targeted_gap: str = Field(description="Which specific gap this proposal addresses.")
    methodology: str = Field(description="A brief 2-3 sentence overview of how this study would be conducted.")
    expected_impact: str = Field(description="Why solving this gap is important to the broader field.")

class InnovatorResult(BaseModel):
    proposals: List[ResearchProposal] = Field(description="3 novel, highly specific research studies.")
