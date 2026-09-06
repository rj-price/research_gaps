import logging
from typing import List

from modules.db import store_gaps
from modules.models import SynthesisResult, CriticResult, InnovatorResult

logger = logging.getLogger(__name__)

async def run_synthesiser_agent(client: "OpenRouterClient", model_id: str, summaries: List[str], subject: str, generate_func) -> SynthesisResult:
    """Agent 1: Reads all summaries and creates a cohesive state of the field."""
    logger.info("Agent 1 (Synthesiser) is analysing summaries...")
    combined_summaries = "\n\n---\n\n".join(summaries)
    prompt = f"""
    You are the Synthesiser Agent. You have been provided with summaries of recent academic papers on "{subject}".
    
    Your goal is to conduct a meta-analysis and synthesise the current state of the art.
    Read the following summaries and extract a cohesive narrative of what is established, and the dominant methodologies.
    
    Summaries:
    {combined_summaries}
    """
    
    return await generate_func(
        client, model_id, prompt,
        response_model=SynthesisResult,
        system_instruction="You are an expert academic Synthesiser.",
    )

async def run_critic_agent(client: "OpenRouterClient", model_id: str, summaries: List[str], synthesis: SynthesisResult, generate_func) -> CriticResult:
    """Agent 2: Reads the synthesis and raw summaries to find deep research gaps."""
    logger.info("Agent 2 (Critic) is finding research gaps...")
    combined_summaries = "\n\n---\n\n".join(summaries)
    
    prompt = f"""
    You are the Critic Agent. You have been provided with raw paper summaries and a synthesised 'State of the Field'.
    
    Your goal is to strictly identify systemic Research Gaps. Look for missing variables, methodological flaws, and contradictions.
    Do not be polite; be highly critical and analytical.

    As well as the prose fields, break your analysis into the 'gaps' list: one entry per
    distinct gap, each with a short title and a description specific enough that a future
    paper could be judged to fill it or not. These entries are stored and later matched
    against newly published literature, so avoid vague phrasing such as "more work is
    needed" and name the organism, variable or method involved.
    
    State of the Field Synthesis:
    Narrative: {synthesis.narrative}
    Methodologies: {synthesis.dominant_methodologies}
    
    Raw Summaries for reference:
    {combined_summaries}
    """

    return await generate_func(
        client, model_id, prompt,
        response_model=CriticResult,
        system_instruction="You are a ruthless academic Critic analysing research gaps.",
    )

async def run_innovator_agent(client: "OpenRouterClient", model_id: str, critic_result: CriticResult, generate_func) -> InnovatorResult:
    """Agent 3: Takes the gaps from the Critic and generates novel proposals."""
    logger.info("Agent 3 (Innovator) is formulating research proposals...")
    
    prompt = f"""
    You are the Innovator Agent. Your colleague, the Critic Agent, has identified several severe research gaps in the literature.
    
    Your goal is to invent 3 highly novel, specific research proposals that directly address these gaps.
    
    Identified Research Gaps:
    Unexplored Territories: {critic_result.unexplored_territories}
    Methodological Limitations: {critic_result.methodological_limitations}
    Contradictions: {critic_result.contradictions}
    """

    return await generate_func(
        client, model_id, prompt,
        response_model=InnovatorResult,
        system_instruction="You are a brilliant academic Innovator formulating new studies.",
    )

async def run_multi_agent_pipeline(client: "OpenRouterClient", model_id: str, summaries: List[str], subject: str, generate_func) -> str:
    """Orchestrates the 3-step sequential agent pipeline and formats the final Markdown report."""
    
    synthesis = await run_synthesiser_agent(client, model_id, summaries, subject, generate_func)
    critic = await run_critic_agent(client, model_id, summaries, synthesis, generate_func)
    innovator = await run_innovator_agent(client, model_id, critic, generate_func)

    # Persist the discrete gaps so the ambient watcher can match new papers against them.
    # A storage failure must not lose the report the user is waiting for.
    if critic.gaps:
        try:
            await store_gaps(subject, critic.gaps)
        except Exception as e:
            logger.error(f"Failed to store gaps for tracking: {e}")

    logger.info("Multi-Agent pipeline successfully completed.")
    
    # Format into a clean markdown report
    report = f"""
## 1. State of the Field Synthesis

{synthesis.narrative}

**Dominant Methodologies**: 
{synthesis.dominant_methodologies}

## 2. Critical Analysis of Research Gaps

### Unexplored Territories
{critic.unexplored_territories}

### Methodological Limitations
{critic.methodological_limitations}

### Contradictions & Tensions
{critic.contradictions}

## 3. Formulated Research Proposals

"""
    for i, prop in enumerate(innovator.proposals):
        report += f"""
### Proposal Idea {i+1}: {prop.title}
*   **Targeted Gap**: {prop.targeted_gap}
*   **Proposed Approach/Methodology**: {prop.methodology}
*   **Expected Impact**: {prop.expected_impact}
"""
    return report
