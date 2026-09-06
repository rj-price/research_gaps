import os
import base64
import asyncio
import logging
import json
from typing import Any, Dict, List, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from modules.models import PaperSummary
from modules.prompts import get_summary_prompt
from modules.db import get_cached_summary, cache_summary, get_file_hash
from modules.agents import run_multi_agent_pipeline
from modules.pdf import extract_pdf_text, pdf_mode

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class TransientLLMError(Exception):
    """A failure worth retrying: rate limiting, a provider blip, or unparseable output."""


class OpenRouterClient:
    """Thin async client for the OpenRouter chat completions API.

    OpenRouter is OpenAI-compatible, so the request shape here works for any model it
    routes to. `provider.require_parameters` keeps it from silently falling back to a
    provider that ignores the JSON schema.
    """

    def __init__(self, api_key: str, timeout: float = 180.0):
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional attribution shown on OpenRouter's activity dashboard
        if site := os.getenv("OPENROUTER_SITE_URL"):
            headers["HTTP-Referer"] = site
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "Research Gap Identifier")
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, headers=self._headers())
        return self._client

    async def complete(self, payload: Dict[str, Any]) -> str:
        """Posts one completion request and returns the message content."""
        client = await self._get_client()
        try:
            response = await client.post(OPENROUTER_URL, json=payload)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            raise TransientLLMError(f"Network error calling OpenRouter: {e}") from e

        if response.status_code in RETRYABLE_STATUS:
            raise TransientLLMError(f"OpenRouter returned {response.status_code}: {response.text[:300]}")
        if response.status_code >= 400:
            # 401, 402 and 400 are configuration problems; retrying will not help
            raise RuntimeError(f"OpenRouter returned {response.status_code}: {response.text[:500]}")

        data = response.json()
        if "error" in data and data["error"]:
            raise TransientLLMError(f"OpenRouter error: {data['error']}")

        choices = data.get("choices") or []
        if not choices:
            raise TransientLLMError(f"OpenRouter returned no choices: {str(data)[:300]}")

        message = choices[0].get("message") or {}
        if message.get("refusal"):
            raise RuntimeError(f"Model refused the request: {message['refusal']}")

        content = message.get("content")
        if not content:
            raise TransientLLMError("OpenRouter returned an empty message.")
        return content

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def get_client() -> OpenRouterClient:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables.")
        raise ValueError("OPENROUTER_API_KEY missing")
    return OpenRouterClient(api_key)


# --- Structured output ---

def _strictify(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrites a Pydantic JSON schema for OpenAI-style strict structured output.

    Strict mode requires every property to be listed in `required` and forbids
    additional properties, so optional fields with defaults must be marked required too.
    """
    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False
        properties = schema.get("properties", {})
        schema["required"] = list(properties.keys())
        for subschema in properties.values():
            _strictify(subschema)
    for key in ("items", "not"):
        if isinstance(schema.get(key), dict):
            _strictify(schema[key])
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for subschema in schema.get(key, []) or []:
            _strictify(subschema)
    for definition in (schema.get("$defs") or {}).values():
        _strictify(definition)
    return schema


def build_response_format(response_model: Type[BaseModel]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "strict": True,
            "schema": _strictify(response_model.model_json_schema()),
        },
    }


def _strip_fences(text: str) -> str:
    """Some models wrap JSON in a Markdown fence even under a schema constraint."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


@retry(
    retry=retry_if_exception_type(TransientLLMError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def generate_structured(
    client: OpenRouterClient,
    model_id: str,
    prompt: str,
    response_model: Type[T],
    system_instruction: str,
    pdf_path: str | None = None,
) -> T:
    """Calls OpenRouter and returns the response parsed into `response_model`."""
    user_content: Any = prompt
    plugins = None

    if pdf_path:
        mode = pdf_mode()
        if mode == "native":
            # Hand the raw PDF to OpenRouter's file parser rather than extracting locally
            with open(pdf_path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode()
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "file", "file": {
                    "filename": os.path.basename(pdf_path),
                    "file_data": f"data:application/pdf;base64,{encoded}",
                }},
            ]
            plugins = [{"id": "file-parser", "pdf": {"engine": os.getenv("PDF_ENGINE", "pdf-text")}}]
        else:
            text = await asyncio.to_thread(extract_pdf_text, pdf_path, mode)
            user_content = f"{prompt}\n\n--- BEGIN DOCUMENT TEXT ---\n{text}\n--- END DOCUMENT TEXT ---"

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ],
        "response_format": build_response_format(response_model),
        "provider": {"require_parameters": True},
    }
    if plugins:
        payload["plugins"] = plugins

    content = await client.complete(payload)
    try:
        return response_model.model_validate_json(_strip_fences(content))
    except (ValidationError, json.JSONDecodeError) as e:
        # Worth one more attempt: schema violations from a model are usually transient
        raise TransientLLMError(f"Could not parse a {response_model.__name__} from the response: {e}") from e


# --- Pipeline steps ---

async def summarise_paper(
    client: OpenRouterClient, model_id: str, pdf_path: str, limiter: AsyncLimiter
) -> str:
    """Summarises a single paper (with SQLite caching)."""
    filename = os.path.basename(pdf_path)

    # Check cache first
    file_hash = await asyncio.to_thread(get_file_hash, pdf_path)
    cached_json = await get_cached_summary(file_hash)
    if cached_json:
        logger.info(f"Loaded {filename} from SQLite cache.")
        summary_data = PaperSummary.model_validate_json(cached_json)
        return summary_data.to_markdown()

    # Wait for rate limit before calling the API
    async with limiter:
        try:
            logger.info(f"Summarising {filename} via OpenRouter ({model_id}, pdf mode: {pdf_mode()})...")
            summary_data = await generate_structured(
                client,
                model_id,
                prompt=get_summary_prompt(filename),
                response_model=PaperSummary,
                system_instruction="You are an expert academic researcher.",
                pdf_path=pdf_path,
            )

            await cache_summary(file_hash, filename, summary_data.model_dump_json())
            logger.info(f"Successfully summarised {filename}.")
            return summary_data.to_markdown()

        except Exception as e:
            logger.error(f"Error summarising {filename}: {e}")
            return f"Error summarising {filename}: {e}\n"


async def identify_gaps(
    client: OpenRouterClient, model_id: str, summaries: List[str], subject: str, limiter: AsyncLimiter
) -> str:
    """Orchestrates the multi-agent synthesis."""
    async with limiter:
        try:
            # Pass generate_structured down so the agents inherit the retry behaviour
            return await run_multi_agent_pipeline(client, model_id, summaries, subject, generate_structured)
        except Exception as e:
            logger.error(f"Error in multi-agent pipeline: {e}")
            return f"Error analysing gaps: {e}"
