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
