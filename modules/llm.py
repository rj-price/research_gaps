import os
import asyncio
import logging
from typing import List

from google import genai
from google.genai import types
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential

from modules.models import PaperSummary
from modules.prompts import get_summary_prompt, get_gaps_prompt

logger = logging.getLogger(__name__)

def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        raise ValueError("GOOGLE_API_KEY missing")
    return genai.Client(api_key=api_key)

# We use tenacity to retry on common transient errors
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def generate_with_retry(client: genai.Client, model_id: str, contents, config: types.GenerateContentConfig):
    return await client.aio.models.generate_content(
        model=model_id,
        contents=contents,
        config=config
    )

async def summarize_paper(
    client: genai.Client, model_id: str, pdf_path: str, limiter: AsyncLimiter
) -> str:
    """Uploads and uses Gemini to summarize a single paper."""
    filename = os.path.basename(pdf_path)
    
    # Wait for rate limit
    async with limiter:
        uploaded_file = None
        try:
            logger.info(f"Uploading {filename}...")
            # Upload via File API
            uploaded_file = client.files.upload(file=pdf_path)
            
            # System Instruction + Structured Output (Pydantic)
            config = types.GenerateContentConfig(
                system_instruction="You are an expert academic researcher.",
                response_mime_type="application/json",
                response_schema=PaperSummary,
            )
            
            prompt = get_summary_prompt(filename)
            logger.info(f"Generating summary for {filename}...")
            
            # Using client.aio for async generation
            response = await generate_with_retry(
                client, 
                model_id, 
                contents=[uploaded_file, prompt], 
                config=config
            )
            
            # The response text is a JSON string matching the PaperSummary schema
            summary_data = PaperSummary.model_validate_json(response.text)
            logger.info(f"Successfully summarized {filename}.")
            return summary_data.to_markdown()

        except Exception as e:
            logger.error(f"Error summarizing {filename}: {e}")
            return f"Error summarizing {filename}: {e}\n"
        finally:
            # Cleanup File strictly in finally block
            if uploaded_file:
                try:
                    logger.debug(f"Cleaning up file {uploaded_file.name}...")
                    client.files.delete(name=uploaded_file.name)
                except Exception as cleanup_e:
                    logger.error(f"Failed to delete file {uploaded_file.name}: {cleanup_e}")

async def identify_gaps(client: genai.Client, model_id: str, summaries: List[str], subject: str, limiter: AsyncLimiter) -> str:
    """Uses Gemini to synthesize summaries and find gaps."""
    combined_summaries = "\n\n---\n\n".join(summaries)
    prompt = get_gaps_prompt(subject, combined_summaries)
    
    config = types.GenerateContentConfig(
        system_instruction="You are a senior principal investigator.",
    )
    
    async with limiter:
        try:
            logger.info("Generating research gap report...")
            response = await generate_with_retry(
                client,
                model_id,
                contents=prompt,
                config=config
            )
            logger.info("Successfully generated research gap report.")
            return response.text
        except Exception as e:
            logger.error(f"Error analyzing gaps: {e}")
            return f"Error analyzing gaps: {e}"
