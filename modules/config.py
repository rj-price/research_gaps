"""Watchlist configuration for the ambient literature agent."""
import json
import logging
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from modules.llm import DEFAULT_MODEL

logger = logging.getLogger(__name__)

DEFAULT_PATH = "watchlist.json"


class WatchConfig(BaseModel):
    organisms: List[str] = Field(
        default_factory=lambda: ["Fusarium", "Rubus", "Fragaria", "Rhynchosporium"],
        description="Terms searched in title and abstract across all sources.",
    )
    extra_terms: List[str] = Field(
        default_factory=list,
        description="Additional watch terms ORed with the organisms (e.g. 'Fusarium oxysporum f. sp. fragariae').",
    )
    interests: str = Field(
        default="Plant pathology, comparative genomics, effector biology, host resistance and crop breeding.",
        description="Free-text description of the group's interests, used by the relevance screener.",
    )
    pubmed_query: str = Field(
        default="",
        description="Optional extra PubMed qualifier ANDed with the term clause, e.g. 'genomics OR pathogenicity'.",
    )
    preprint_servers: List[str] = Field(default_factory=lambda: ["biorxiv"])
    min_relevance_score: float = Field(default=0.6, description="Relevance verdicts below this score are dropped.")
    min_match_confidence: float = Field(
        default=0.75,
        description="Gap matches below this confidence are discarded rather than stored. Raise it if the digest links papers to gaps on thin evidence.",
    )
    delivery: List[str] = Field(default_factory=list, description="Any of: email, push.")
    model: str = Field(default=DEFAULT_MODEL, description="Any OpenRouter model ID.")
    rate_limit: int = Field(default=5, description="Max LLM requests per minute.")
    max_results: int = Field(default=200, description="Cap on PubMed hits per run.")

    @property
    def terms(self) -> List[str]:
        return self.organisms + self.extra_terms


def load_config(path: str = DEFAULT_PATH) -> WatchConfig:
    """Loads the watchlist, falling back to the built-in defaults if the file is absent."""
    config_path = Path(path)
    if not config_path.exists():
        logger.warning(f"No watchlist at {path}; using built-in defaults.")
        return WatchConfig()
    config = WatchConfig.model_validate_json(config_path.read_text())
    logger.info(f"Loaded watchlist from {path}: {', '.join(config.terms)}")
    return config
