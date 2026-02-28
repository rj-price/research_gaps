import os
import asyncio
import logging
import json
from typing import List

from google import genai
from google.genai import types
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential

from modules.models import PaperSummary
from modules.prompts import get_summary_prompt
from modules.db import get_cached_summary, cache_summary, get_file_hash
from modules.agents import run_multi_agent_pipeline

logger = logging.getLogger(__name__)

def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        raise ValueError("GOOGLE_API_KEY missing")
    return genai.Client(api_key=api_key)

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

async def summarise_paper(
    client: genai.Client, model_id: str, pdf_path: str, limiter: AsyncLimiter
) -> str:
    """Uploads and uses Gemini to summarise a single paper (with SQLite caching)."""
    filename = os.path.basename(pdf_path)
    
    # Check cache first
    file_hash = await asyncio.to_thread(get_file_hash, pdf_path)
    cached_json = await get_cached_summary(file_hash)
    if cached_json:
        logger.info(f"Loaded {filename} from SQLite cache.")
        summary_data = PaperSummary.model_validate_json(cached_json)
        return summary_data.to_markdown()

    # Wait for rate limit before calling API
    async with limiter:
        uploaded_file = None
        try:
            logger.info(f"Uploading {filename} to Gemini...")
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
            
            response = await generate_with_retry(
                client, 
                model_id, 
                contents=[uploaded_file, prompt], 
                config=config
            )
            
            # Save to Cache 
            await cache_summary(file_hash, filename, response.text)
            
            summary_data = PaperSummary.model_validate_json(response.text)
            logger.info(f"Successfully summarised {filename}.")
            return summary_data.to_markdown()

        except Exception as e:
            logger.error(f"Error summarising {filename}: {e}")
            return f"Error summarising {filename}: {e}\n"
        finally:
            # Cleanup File strictly in finally block
            if uploaded_file:
                try:
                    logger.debug(f"Cleaning up file {uploaded_file.name}...")
                    client.files.delete(name=uploaded_file.name)
                except Exception as cleanup_e:
                    logger.error(f"Failed to delete file {uploaded_file.name}: {cleanup_e}")

async def identify_gaps(client: genai.Client, model_id: str, summaries: List[str], subject: str, limiter: AsyncLimiter) -> str:
    """Orchestrates the multi-agent synthesis."""
    async with limiter:
        try:
            # We pass down generate_with_retry so the agents get the tenacity benefits
            return await run_multi_agent_pipeline(client, model_id, summaries, subject, generate_with_retry)
        except Exception as e:
            logger.error(f"Error in multi-agent pipeline: {e}")
            return f"Error analysing gaps: {e}"
