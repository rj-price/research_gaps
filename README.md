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
-   **Rolling Gap Identification:** The watcher does not only match against gaps you already have — once enough relevant papers have accumulated for a subject, it runs the Synthesiser and Critic over their abstracts to identify new gaps, deduplicating them against the ones already tracked.
-   **Scored, Not Vibed:** Four eval suites score the LLM-dependent steps against fixed datasets — relevance screening and gap matching against hand-labelled ground truth, gap analysis quality against an LLM judge, from full text and from abstracts — with regression thresholds enforced in CI.
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
-   `evals/`: The eval harness — datasets, scoring, the LLM judge, the runner and the regression thresholds. See [Evals](#evals).

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
| `preprint_max_pages` | Pages of the preprint API to walk per server, 100 records each (default 400). The API has no keyword search, so the whole window is paged and filtered locally. |
| `min_relevance_score` | Relevance verdicts below this score are dropped. Raise it if the digest is noisy. |
| `min_match_confidence` | Gap matches below this confidence are discarded rather than stored. |
| `subjects` | Subject groups for rolling gap synthesis, each with the watch terms that route papers into it. Leave empty to disable. |
| `rolling_min_papers` | Unsynthesised relevant papers a subject must accumulate before it is synthesised. |
| `rolling_max_papers` | Cap on papers fed to one synthesis, so a large backlog cannot blow the context window. |
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

### Rolling gap identification

The PDF pipeline in `main.py` produces gaps from papers you have deliberately read. The
watcher produces them from the abstracts it is already screening, so the gap store keeps
growing between those sessions:

```bash
python watch.py synthesise --status   # what is banked, per subject. Free: no LLM calls
python watch.py synthesise            # synthesise every subject over its trigger
python watch.py synthesise --force    # ignore the trigger and synthesise anything banked
python watch.py run --no-rolling      # screen and match only, as before
```

It also runs at the end of `watch.py run` by default — after matching, never before, so a
gap minted from this run's papers is not immediately offered back to those same papers as
something to fill.

Four things make it work rather than just run:

**Subjects.** The Critic needs a topic, and one bucket holding rust genomics and soft
fruit breeding together produces gaps too general to match anything. Declare subject
groups in `watchlist.json`; papers route into them by the watch terms they matched, so
routing is free and deterministic rather than another model call.

```json
"subjects": [
  { "name": "rust fungi genomics and resistance breaking",
    "terms": ["Puccinia", "Pucciniales", "yellow rust"] }
],
"rolling_min_papers": 12,
"rolling_max_papers": 30
```

**A trigger, not every run.** A subject waits until `rolling_min_papers` unsynthesised
relevant papers have accumulated. Five abstracts do not make a field, and synthesising
them anyway produces exactly the vague gaps that clutter the store — a single-abstract
subject run under `--force` yielded five, none of them worth keeping.

**Deduplication.** `make_gap_id` hashes the exact title, so a reworded restatement of the
same gap would otherwise become a new row every cycle and the matcher's prompt would grow
without bound. Every candidate is checked against the gaps already stored for that subject
before it is written; a duplicate updates the existing gap's description and inherits the
new paper as a source. The title is never rewritten, because it is the ID's input and
changing it would orphan every match already recorded.

**Provenance.** `gap_sources` records which papers a gap came from, and those papers are
excluded when that gap is offered to the matcher. Without it a paper eventually gets
credited with filling the gap it created.

Gaps found this way are marked `origin='abstract'`, shown as `from: abstracts` in
`watch.py gaps` and flagged in the digest. Treat them as leads: no abstract states its own
limitations, which is the Critic's richest input in the full-text path. The
`gap_analysis_abstracts` eval suite exists to measure exactly that weakness.

### Scheduling

A weekly digest, every Monday at 07:00:

```cron
0 7 * * 1 cd /path/to/research_gaps && .venv/bin/python watch.py run --output digest.md >> watcher.log 2>&1
```

The run is resumable and deduplicating, so a missed week is caught up by the next run
rather than lost, and running it twice by accident costs nothing.

## Evals

Four suites score the parts of the pipeline where the model can quietly get worse.
Each drives the real production code path — `screen_relevance`, `match_paper_to_gaps`,
the Synthesiser and Critic agents — so a regression in the app shows up here rather than
in a digest six weeks later.

| Suite | What it scores | Ground truth | Headline metrics |
| --- | --- | --- | --- |
| `relevance` | The watcher's first filter | 14 hand-labelled papers, including keyword traps: *Fusarium* keratitis, a *Rubus* nutraceutical study, strawberry as a pesticide-residue matrix | precision, recall, F1, hard-negative rejection rate |
| `gap_matching` | Linking a new paper to a stored gap | 4 gap fixtures, 6 papers with expected links and relationships, two of which must match nothing | link precision/recall/F1, relationship accuracy, silence on unrelated papers |
| `gap_analysis` | The quality of the gaps the Critic produces | Summary sets with gaps deliberately planted in them, plus claims the summaries do not support | mean groundedness and specificity (judge, 1–5), theme recall, fabrication rate, structural issues |
| `gap_analysis_abstracts` | The same, on the evidence the rolling watcher actually has | The same two subjects as abstracts, with distractors that are all plausible *author-stated limitations* no abstract contains | as above; `fabrication_rate` is the one that matters |

The first two have objective answers, so they are scored arithmetically. Gap quality has
no such answer, so a **stronger judge model** (`openai/gpt-5.6-terra` by default,
override with `EVAL_JUDGE_MODEL`) grades each gap for whether it is grounded in the
summaries the Critic was actually given and specific enough that a future paper could be
judged to fill it. The judge never sees the answer key while grading individual gaps. A
model grading its own output flatters itself, so keep the judge different from the model
under test.

Alongside the judged scores, both gap analysis suites run free deterministic checks:
valid category, title within 15 words, non-empty description, at least one discrete gap.
These matter because the watcher stores and matches against those fields.

### Running them

```bash
pip install -r requirements-dev.txt

# The harness itself: no API key, no network, no cost
pytest evals -q

# The suites, against the default model
python -m evals.runner --suite all

# One suite, a different model, and fail on a regression
python -m evals.runner --suite relevance --model anthropic/claude-sonnet-5 --check

# Cheap smoke test while iterating on a prompt
python -m evals.runner --suite gap_analysis --limit 1
```

Every run writes a JSON record to `evals/results/` — the metrics, plus every case with
the model's own reasoning — so two models, or the same model before and after a prompt
change, can be compared after the fact. The runner prints only the cases it got wrong,
which is the part worth reading.

The same suites run as a pytest regression test, skipped by default so nobody pays for a
run by accident:

```bash
RUN_LLM_EVALS=1 pytest evals/test_suites.py -v
```

### Baseline

Measured on `google/gemini-2.5-flash`, judged by `openai/gpt-5.6-terra`, September 2026:

| Suite | Result |
| --- | --- |
| `relevance` | precision 1.00, recall 1.00, F1 1.00, all five hard negatives rejected |
| `gap_matching` | precision 1.00, recall 1.00, relationship accuracy 1.00, silent on both unrelated papers |
| `gap_analysis` | grounded 4.77/5, specific 4.92/5, theme recall 1.00, no fabrications, no structural issues |
| `gap_analysis_abstracts` | grounded 4.38/5, specific 4.50/5, theme recall 0.50, no fabrications, no structural issues |

The first three sit at the ceiling, which means those datasets currently prove the
pipeline is not broken rather than discriminating between good and better. Feed real
failures back in as they turn up — that is what sharpens them.

`gap_analysis_abstracts` is the one that discriminates, and its lower theme recall is the
honest cost of the feature rather than a bug: abstracts state findings, not limitations,
so some gaps genuinely are not visible from them. Its thresholds are set accordingly, with
one exception — `fabrication_rate` is held at the full-text bound of 0.05, because weaker
evidence is not a licence to invent. Tightening the Critic's instruction to return fewer,
more concrete gaps moved that suite from 16 gaps at specificity 4.0 to 8 gaps at 4.5,
trading theme recall (0.625 to 0.50) for gaps worth storing.

The `gap_analysis` run above also earned its keep on the first attempt: it found that
**every** gap the Critic produced carried an invalid `category`, because the model echoed
the plural section headings (`unexplored_territories`) rather than the singular values the
schema documented. `IdentifiedGap.category` is now a `Literal`, so the value reaches the
model as a schema enum, with a validator that accepts the plural headings behind it.

### Thresholds

`evals/thresholds.json` is the contract, and `--check` exits non-zero when a bound is
breached. The bounds sit just below the measured baseline: on the 14-case relevance set,
one wrong call is tolerated and two are a failure. A breach means behaviour moved — a
model update, a prompt edit, a schema change — not that a test is flaky. Read the
reported metrics and the failing cases before relaxing a bound.

`judge_errors` is the exception, and it means the judge itself never answered. Plant
pathology prose about virulence, effectors and deletion mutants trips provider-side
content filters on some routes: `anthropic/claude-sonnet-5` via the AWS route was refused
on roughly 40 percent of identical calls, while `openai/gpt-5.6-terra` and
`anthropic/claude-opus-5` were never refused. Filtered calls are retried, and a case the
judge never scored is excluded from the means rather than counted as a zero, so a broken
judge shows up as a `judge_errors` breach instead of a quietly terrible score. Check a new
`EVAL_JUDGE_MODEL` on a handful of calls before trusting it with a run.

### CI

`.github/workflows/evals.yml` runs the offline harness tests on every push and pull
request, then scores all three suites against the thresholds and publishes the scores to
the job summary. The billed job needs an `OPENROUTER_API_KEY` repository secret; without
it, it warns and skips rather than failing. It also runs weekly, so drift on a pinned
model ID is caught without waiting for a push, and `workflow_dispatch` takes a model ID
if you want to compare candidates.

### Extending the datasets

The datasets are plain JSON in `evals/datasets/`, validated on load by the Pydantic
models in `evals/cases.py`. The most valuable additions are real failures: when the
watcher keeps a paper it should have rejected, add it to `relevance.json` with the
correct label. The fixtures shipped here are synthetic — written for the suite, not
drawn from real publications — so they can be shared freely and the ground truth is
unambiguous.

## Output

The script generates a comprehensive Markdown file (`.md`) containing:
1.  **State of the Field Synthesis:** The cohesive narrative built by the Synthesiser agent.
2.  **Critical Analysis of Research Gaps:** Deep flaws and contradictions identified by the Critic agent.
3.  **Formulated Research Proposals:** Actionable research ideas proposed by the Innovator agent.
4.  **Source Paper Summaries:** The individual Pydantic-structured summaries extracted from each PDF.
