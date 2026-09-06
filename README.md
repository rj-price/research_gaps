# Academic Research Gap Identifier

A powerful, asynchronous Python tool that uses the OpenRouter API to analyse collections of academic papers (PDFs). It extracts structured summaries from each paper, caches them locally, and then runs them through a pure-Python multi-agent synthesis pipeline to identify systemic research gaps and propose novel follow-up studies.

## Features

-   **Automated Summarisation:** Extracts core research questions, methodologies, key findings, and explicit limitations from each PDF into a structured format.
-   **Any Model, One Key:** Routed through OpenRouter, so the model is a config string — swap `google/gemini-2.5-flash` for an Anthropic, OpenAI or open-weights model without touching the code. Strict JSON-schema structured output is enforced on every call, and `require_parameters` stops OpenRouter falling back to a provider that would ignore the schema.
-   **Intelligent SQLite Caching:** Calculates the SHA-256 hash of your PDFs. If a file has been processed before, its summary is instantly loaded from a local `research_cache.db` database, saving significant time and API costs.
-   **Multi-Agent Synthesis Pipeline:** Replaces generic summarisation with a rigorous 3-step analytical workflow:
    1.  **Synthesiser Agent:** Builds a cohesive narrative and identifies dominant methodologies across all papers.
    2.  **Critic Agent:** Deep-dives into the synthesis and raw summaries to rigorously extract unexplored territories, methodological flaws, and contradictions.
    3.  **Innovator Agent:** Uses the Critic's strict gaps to formulate 3 highly specific and novel research proposals.
-   **Ambient Literature Watcher:** A scheduled agent that watches PubMed and bioRxiv for your study organisms, screens new papers for genuine relevance, and flags when one of your previously identified research gaps has been filled by a new publication.
-   **Async & Rate-Limited:** Uses `asyncio` for high-throughput concurrent processing, gated seamlessly by `aiolimiter` to respect API rate limits. Automatically retries transient network or API errors with exponential backoff using `tenacity`.

## Architecture Overview

The project offers both a Command Line Interface (CLI) and a Web App Interface.

The codebase is modularised to separate concerns securely:
-   `api.py`: The FastAPI backend serving the analysis pipeline for the web app.
-   `frontend/`: The React+Vite frontend featuring a professional academic design and PDF drag-and-drop.
-   `main.py`: The CLI entry point that handles argument parsing, database initialisation, and asynchronous orchestration.
-   `modules/llm.py`: The OpenRouter client and the single `generate_structured` entry point every agent calls, including strict-schema construction and retries on rate limits, provider blips and unparseable output.
-   `modules/pdf.py`: Local PDF text extraction with `pypdf`.
-   `modules/db.py`: Wraps `aiosqlite` to handle the local database caching layer and asynchronous sha-256 file hashing.
-   `modules/agents.py`: Contains the logic for the 3-step sequential agent pipeline (Synthesiser -> Critic -> Innovator).
-   `modules/prompts.py`: Organises the instructions fed to the language models.
-   `modules/models.py`: Defines strict Pydantic models for data structuring throughout the application, enforcing predictable API outputs.
-   `watch.py`: The CLI entry point for the ambient literature watcher.
-   `modules/sources.py`: Fetchers for PubMed (E-utilities) and bioRxiv/medRxiv, with a keyword prefilter.
-   `modules/screening.py`: The two LLM screening steps — relevance triage, then matching papers to stored gaps.
-   `modules/watch.py`: Orchestrates one watch cycle and records it in SQLite.
-   `modules/digest.py`: Renders the Markdown digest and delivers it by email or ntfy push.
-   `modules/config.py`: The watchlist schema and loader.

## Installation

1.  **Clone the repository:**
    Ensure you are in the project folder.

2.  **Set up a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key:**
    The script requires an [OpenRouter](https://openrouter.ai/keys) API key.
    Copy `.env.example` to `.env` and fill it in:
    ```env
    OPENROUTER_API_KEY=your_api_key_here
    ```
    Everything else in `.env` is optional — see the comments in `.env.example`.

5.  **Install Frontend Dependencies (For Web App):**
    ```bash
    cd frontend
    npm install
    cd ..
    ```

## Usage

### Web Interface

The easiest way to use the application is through the Web UI. We provide a `start.sh` script to run both the FastAPI backend and the Vite frontend simultaneously:

```bash
bash start.sh
```

Then, open your browser to `http://localhost:5173`. You can drag and drop your PDFs into the academic-styled interface to generate a report.

### Command Line Interface

You can also run the script from the command line, pointing it to a folder containing your academic PDFs.

```bash
python main.py path/to/pdf_folder --subject "Your Review Subject"
```

### CLI Arguments

*   `folder`: **(Required)** Path to the directory containing the `.pdf` files you want to analyse.
*   `--subject`: *(Optional)* The general topic of the papers. Providing this helps the multi-agent pipeline stay focused during synthesis. Defaults to "the provided topics".
*   `--output`: *(Optional)* The filename for the final generated Markdown report. Defaults to `research_gap_report.md`.
*   `--model`: *(Optional)* Any [OpenRouter model ID](https://openrouter.ai/models). Defaults to `google/gemini-2.5-flash`.
*   `--rate-limit`: *(Optional)* Maximum number of requests allowed per minute to comply with your API tier limits. Defaults to `5`.
*   `--concurrent-requests`: *(Optional)* Maximum number of concurrent active requests. Adjust based on your system and network limits. Defaults to `5`.

### Example

```bash
# Process a folder named "sample_pdfs" about "agricultural microbiology"
# Save it to "advanced_report.md", processing 10 per minute
python main.py sample_pdfs --subject "agricultural microbiology" --output advanced_report.md --rate-limit 10
```

## PDF Handling

`PDF_MODE` in `.env` controls how a paper reaches the model.

| Mode | What it does | Cost |
| --- | --- | --- |
| `auto` *(default)* | docling if it is installed, otherwise pypdf. | — |
| `docling` | Layout-aware conversion to Markdown: real tables, heading hierarchy, de-hyphenated text, OCR for scans. | Slower first run while models load; free thereafter. |
| `text` | `pypdf`'s raw text layer. Reading order is reliable on multi-column papers, but tables are flattened and headings are lost. | Free, no extra dependencies. |
| `native` | Sends the PDF itself to OpenRouter's file-parser plugin. Only works with models that accept file inputs. | `PDF_ENGINE` selects `pdf-text` (free), `mistral-ocr` (paid, best for scans) or `native` (the model's own vision). |

`docling` is the better choice for real journal PDFs: it preserves the heading structure
the summariser needs to locate the Discussion and Limitations sections, and it is the only
local option that reads scans. Measured on an 18-page two-column Frontiers paper:

| | docling | pypdf |
| --- | --- | --- |
| Markdown headings recovered | 57 | 0 |
| Table rows recovered | 63 | 0 |
| Words broken across lines by hyphenation | 0 | 120 |
| Conversion time (CPU, warm) | ~79 s | ~1 s |

It is optional because it is a heavy install — docling pulls in torch and the OCR models,
taking the virtual environment to roughly 6 GB, and it downloads model weights on first
use:

```bash
pip install -r requirements-docling.txt
```

Because summaries are cached by file hash, that slow parse is a one-off cost per paper —
every later run reads the summary from SQLite in milliseconds. If docling fails on a given
PDF the extraction falls back to `pypdf` rather than dropping the paper.

## Ambient Literature Watcher

The gap analysis above is a one-shot exercise: it tells you what was missing from the
literature on the day you ran it. The watcher turns that into a standing service. It
polls PubMed and bioRxiv for your study organisms, screens what it finds, and tells you
when someone has published the study you identified as missing.

### How it works

1.  **Gap storage.** When the multi-agent pipeline runs, the Critic agent now also emits
    each gap as a discrete, individually trackable record, stored in `research_cache.db`
    with a stable ID. This happens automatically for both the CLI and the web app.
2.  **Fetch.** Each run pulls a date window from PubMed (E-utilities, searching title and
    abstract) and bioRxiv (date-interval API, filtered locally on whole-word term
    matches). The window defaults to everything since the last successful run.
3.  **Deduplicate.** Papers already in the database are dropped, so nothing is ever
    screened — or paid for — twice.
4.  **Relevance screening.** Batches of new papers go to the LLM with your stated research
    interests. This is the step that keeps `Rubus`-extract nanoparticle chemistry out of
    a plant pathology digest.
5.  **Gap matching.** Each surviving paper is compared against every stored open gap and
    classified as `fills`, `partially_addresses`, `contradicts` or `informs`, with the
    supporting evidence quoted from the abstract. Matches below `min_match_confidence` are
    discarded. A `fills` at 0.8 or above marks the gap as addressed.
6.  **Digest.** The digest leads with papers that act on a gap — `fills`, `contradicts` or
    `partially_addresses`. An `informs` link is background reading rather than progress, so
    those papers sit in the reading list with a note of the gap they inform. Delivery is by
    email, ntfy push, a file, or all three.

### Configuration

Copy the example watchlist and edit it:

```bash
cp watchlist.example.json watchlist.json
```

| Field | Purpose |
| --- | --- |
| `organisms` | The core watch terms, searched in title and abstract. |
| `extra_terms` | Additional terms ORed with the organisms (`formae speciales`, disease names). |
| `interests` | Free text describing the group's focus. Drives the relevance screener. |
| `pubmed_query` | Optional extra PubMed qualifier, ANDed with the term clause. |
| `preprint_servers` | `biorxiv`, `medrxiv`, or both. Set to `[]` to skip preprints. |
| `min_relevance_score` | Relevance verdicts below this score are dropped. Raise it if the digest is noisy. |
| `min_match_confidence` | Gap matches below this confidence are discarded rather than stored. |
| `delivery` | Any of `email`, `push`. Leave empty to only write the digest to a file. |

Credentials for delivery live in `.env` — see `.env.example`.

### Usage

```bash
# Check what the sources return, without spending any API credit
python watch.py run --dry-run --since 7d

# A real run: screen, match, write the digest, and deliver it
python watch.py run --output digest.md

# Backfill a longer window on first use
python watch.py run --since 2026-06-01 --output digest.md --no-deliver

# Inspect the tracked gaps
python watch.py gaps
python watch.py gaps --all       # including those marked addressed
python watch.py reopen <gap_id>  # if you disagree with the agent
```

`--since` accepts an ISO date or a day count (`7d`). Omit it and the watcher resumes
from the last successful run, overlapping by a day because PubMed entry dates settle
late.

### Scheduling

A weekly digest, every Monday at 07:00:

```cron
0 7 * * 1 cd /path/to/research_gaps && .venv/bin/python watch.py run --output digest.md >> watcher.log 2>&1
```

The run is resumable and deduplicating, so a missed week is caught up by the next run
rather than lost, and running it twice by accident costs nothing.

## Output

The script generates a comprehensive Markdown file (`.md`) containing:
1.  **State of the Field Synthesis:** The cohesive narrative built by the Synthesiser agent.
2.  **Critical Analysis of Research Gaps:** Deep flaws and contradictions identified by the Critic agent.
3.  **Formulated Research Proposals:** Actionable research ideas proposed by the Innovator agent.
4.  **Source Paper Summaries:** The individual Pydantic-structured summaries extracted from each PDF.
