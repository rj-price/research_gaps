"""Watchlist configuration for the ambient literature agent."""
import json
import logging
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from modules.llm import DEFAULT_MODEL
from modules.sources import DEFAULT_PREPRINT_MAX_PAGES

logger = logging.getLogger(__name__)

DEFAULT_PATH = "watchlist.json"


class SubjectGroup(BaseModel):
    """A topic the rolling synthesis reasons over.

    The Critic needs a subject to be critical about, and one bucket holding rust genomics
    and soft fruit breeding together would produce gaps too general to match anything.
    Papers are routed by their matched watch terms rather than clustered by an LLM: it is
    free, deterministic, and mirrors how the work is actually split.
    """
    name: str = Field(description="The subject passed to the Critic, e.g. 'rust fungi comparative genomics'.")
    terms: List[str] = Field(
        description="Watch terms routing a paper into this group. Matched case-insensitively against the paper's matched_terms.",
    )


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
    preprint_max_pages: int = Field(
        default=DEFAULT_PREPRINT_MAX_PAGES,
        description="Pages of the preprint date-interval API to walk per server, 100 records each. The API has no keyword search, so a full window has to be paged and filtered locally.",
    )
    min_relevance_score: float = Field(default=0.6, description="Relevance verdicts below this score are dropped.")
    min_match_confidence: float = Field(
        default=0.75,
        description="Gap matches below this confidence are discarded rather than stored. Raise it if the digest links papers to gaps on thin evidence.",
    )
    subjects: List[SubjectGroup] = Field(
        default_factory=list,
        description="Subject groups for rolling gap synthesis. With none declared, rolling synthesis is skipped.",
    )
    rolling_min_papers: int = Field(
        default=12,
        description="A subject is synthesised once this many unsynthesised relevant papers have accumulated. Below it, they wait for a later run.",
    )
    rolling_max_papers: int = Field(
        default=30,
        description="Cap on papers fed to one synthesis, so a large backlog cannot blow the context window or the budget.",
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


def route_to_subjects(matched_terms: List[str], subjects: List[SubjectGroup]) -> List[str]:
    """Names the subject groups a paper belongs to, by watch term.

    A subject term matches a paper term only when the subject's term is the broader of the
    two: 'Puccinia' claims a paper matched on 'Puccinia striiformis', but a group declaring
    'Fusarium oxysporum f. sp. fragariae' does not claim a paper matched only on the bare
    'Fusarium oxysporum'. Matching the other way round as well was how a first run put
    banana, maize and soybean papers into one bucket and produced gaps general enough to
    fit any of them.

    A paper may land in more than one group. One landing in none is left in the backlog
    rather than forced into a bucket, and `watch.py synthesise --status` counts those, so
    a watch term with no home is visible rather than silently dropped.
    """
    lowered = [term.lower() for term in matched_terms]
    hits = []
    for subject in subjects:
        for term in subject.terms:
            needle = term.lower()
            if any(needle in matched for matched in lowered):
                hits.append(subject.name)
                break
    return hits
