import logging
from typing import List

from google import genai
from google.genai import types

from modules.models import SynthesisResult, CriticResult, InnovatorResult

logger = logging.getLogger(__name__)

async def run_synthesizer_agent(client: genai.Client, model_id: str, summaries: List[str], subject: str, generate_func) -> SynthesisResult:
    """Agent 1: Reads all summaries and creates a cohesive state of the field."""
    logger.info("Agent 1 (Synthesizer) is analyzing summaries...")
    combined_summaries = "\n\n---\n\n".join(summaries)
    prompt = f"""
    You are the Synthesizer Agent. You have been provided with summaries of recent academic papers on "{subject}".
    
    Your goal is to conduct a meta-analysis and synthesize the current state of the art.
    Read the following summaries and extract a cohesive narrative of what is established, and the dominant methodologies.
    
    Summaries:
    {combined_summaries}
    """
    
    config = types.GenerateContentConfig(
        system_instruction="You are an expert academic Synthesizer.",
        response_mime_type="application/json",
        response_schema=SynthesisResult,
    )
    
    response = await generate_func(client, model_id, contents=prompt, config=config)
    return SynthesisResult.model_validate_json(response.text)

async def run_critic_agent(client: genai.Client, model_id: str, summaries: List[str], synthesis: SynthesisResult, generate_func) -> CriticResult:
    """Agent 2: Reads the synthesis and raw summaries to find deep research gaps."""
    logger.info("Agent 2 (Critic) is finding research gaps...")
    combined_summaries = "\n\n---\n\n".join(summaries)
    
    prompt = f"""
    You are the Critic Agent. You have been provided with raw paper summaries and a synthesized 'State of the Field'.
    
    Your goal is to strictly identify systemic Research Gaps. Look for missing variables, methodological flaws, and contradictions.
    Do not be polite; be highly critical and analytical.
    
    State of the Field Synthesis:
    Narrative: {synthesis.narrative}
    Methodologies: {synthesis.dominant_methodologies}
    
    Raw Summaries for reference:
    {combined_summaries}
    """

    config = types.GenerateContentConfig(
        system_instruction="You are a ruthless academic Critic analyzing research gaps.",
        response_mime_type="application/json",
        response_schema=CriticResult,
    )
    
    response = await generate_func(client, model_id, contents=prompt, config=config)
    return CriticResult.model_validate_json(response.text)

async def run_innovator_agent(client: genai.Client, model_id: str, critic_result: CriticResult, generate_func) -> InnovatorResult:
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

    config = types.GenerateContentConfig(
        system_instruction="You are a brilliant academic Innovator formulating new studies.",
        response_mime_type="application/json",
        response_schema=InnovatorResult,
    )
    
    response = await generate_func(client, model_id, contents=prompt, config=config)
    return InnovatorResult.model_validate_json(response.text)

async def run_multi_agent_pipeline(client: genai.Client, model_id: str, summaries: List[str], subject: str, generate_func) -> str:
    """Orchestrates the 3-step sequential agent pipeline and formats the final Markdown report."""
    
    synthesis = await run_synthesizer_agent(client, model_id, summaries, subject, generate_func)
    critic = await run_critic_agent(client, model_id, summaries, synthesis, generate_func)
    innovator = await run_innovator_agent(client, model_id, critic, generate_func)
    
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
