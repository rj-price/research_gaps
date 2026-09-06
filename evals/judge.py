"""LLM-as-judge for the gap analysis suite.

The judge is a different, stronger model from the one under test — a model grading its
own output flatters itself. It never sees the expected themes while grading individual
gaps, only the summaries the Critic was given, so "grounded" means grounded in the
input rather than agreeing with our answer key.

On the default model choice: plant pathology prose about virulence, effectors and
deletion mutants trips provider-side content filters on some routes. Measured over
repeated calls on the same prompt, `anthropic/claude-sonnet-5` via the AWS route was
refused roughly 40 percent of the time, while `openai/gpt-5.6-terra` and
`anthropic/claude-opus-5` were not refused at all. Any override should be sanity-checked
the same way before being trusted with a run.
"""
import os
from typing import List

from pydantic import BaseModel, Field

from modules.llm import OpenRouterClient, generate_structured

DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-terra"

JUDGE_SYSTEM = (
    "You are a rigorous evaluator of research analysis. You are sceptical, you check "
    "claims against the source text, and you award high scores only when they are earned."
)


def judge_model() -> str:
    """The judge model, as an OpenRouter slug. Override with EVAL_JUDGE_MODEL."""
    return os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


class GapGrade(BaseModel):
    gap_index: int = Field(description="Index of the gap in the numbered list provided.")
    grounded: int = Field(
        description="1 to 5. 5 = every claim traces to the summaries. 1 = the gap rests on "
                    "assertions the summaries do not support."
    )
    specific: int = Field(
        description="1 to 5. 5 = names the organism, variable or method and could be judged "
                    "filled or not by a future paper. 1 = generic filler such as 'more work is needed'."
    )
    comment: str = Field(description="One sentence justifying both scores.")


class GapGradeBatch(BaseModel):
    grades: List[GapGrade] = Field(description="One grade per gap, in the order provided.")


class ThemeVerdict(BaseModel):
    theme_index: int = Field(description="Index of the theme in the numbered list provided.")
    covered: bool = Field(description="True if at least one identified gap substantively covers this theme.")
    covering_gap: str = Field(description="Title of the gap that covers it, or an empty string.")


class DistractorVerdict(BaseModel):
    claim_index: int = Field(description="Index of the claim in the numbered list provided.")
    asserted: bool = Field(
        description="True if the analysis states this claim, or something materially equivalent, as fact."
    )
    quote: str = Field(description="The offending phrase from the analysis, or an empty string.")


class CoverageReport(BaseModel):
    themes: List[ThemeVerdict]
    distractors: List[DistractorVerdict]


def _clamp(score: int) -> int:
    return max(1, min(5, int(score)))


async def grade_gaps(
    client: OpenRouterClient, summaries: List[str], gaps: List, limiter, model: str | None = None
) -> List[GapGrade]:
    """Scores each identified gap for groundedness and specificity against the source summaries."""
    if not gaps:
        return []

    listing = "\n\n".join(
        f"[{i}] title: {gap.title}\n    category: {gap.category}\n    description: {gap.description}"
        for i, gap in enumerate(gaps)
    )
    prompt = f"""
    An analysis agent read the paper summaries below and produced a list of research gaps.
    Grade each gap on two axes.

    Grounded: does the gap follow from what these summaries actually say? A gap that
    invents findings, cites work not present, or misrepresents a summary scores low.
    A gap that correctly identifies something absent from the summaries scores high —
    absence is legitimate evidence, invention is not.

    Specific: could a future paper be judged to fill this gap or not? Named organisms,
    variables, methods and conditions score high. Vague calls for further research score low.

    --- PAPER SUMMARIES GIVEN TO THE AGENT ---
    {chr(10).join(f"### Paper {i + 1}{chr(10)}{s}" for i, s in enumerate(summaries))}
    --- END SUMMARIES ---

    Gaps to grade:
    {listing}
    """

    async with limiter:
        result = await generate_structured(
            client, model or judge_model(), prompt,
            response_model=GapGradeBatch,
            system_instruction=JUDGE_SYSTEM,
        )

    grades = [g for g in result.grades if 0 <= g.gap_index < len(gaps)]
    for grade in grades:
        grade.grounded = _clamp(grade.grounded)
        grade.specific = _clamp(grade.specific)
    return grades


async def check_coverage(
    client: OpenRouterClient, report: str, themes: List[str], distractors: List[str],
    limiter, model: str | None = None,
) -> CoverageReport:
    """Checks the analysis for the gaps it should have found and the claims it must not make."""
    theme_listing = "\n".join(f"[{i}] {theme}" for i, theme in enumerate(themes))
    distractor_listing = "\n".join(f"[{i}] {claim}" for i, claim in enumerate(distractors)) or "(none)"

    prompt = f"""
    Below is a research gap analysis produced by an agent, followed by two checklists.

    For each theme, decide whether the analysis substantively covers it. Wording need not
    match; the substance must. A passing mention that does not identify the theme as a
    problem does not count as coverage.

    For each claim, decide whether the analysis asserts it as fact. These claims are false
    for the material the agent was given, so any assertion of one is a fabrication.
    Quote the offending phrase when you find one.

    --- ANALYSIS ---
    {report}
    --- END ANALYSIS ---

    Themes the analysis should have identified:
    {theme_listing}

    Claims the analysis must not assert:
    {distractor_listing}
    """

    async with limiter:
        return await generate_structured(
            client, model or judge_model(), prompt,
            response_model=CoverageReport,
            system_instruction=JUDGE_SYSTEM,
        )
