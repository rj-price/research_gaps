# Academic Research Gap Identifier

A powerful, asynchronous Python tool that leverages the Google Gemini API to analyse collections of academic papers (PDFs). It extracts structured summaries from each paper, caches them locally, and then runs them through a pure-Python multi-agent synthesis pipeline to identify systemic research gaps and propose novel follow-up studies.

## Features

-   **Automated Summarisation:** Uploads PDFs directly to Gemini to extract core research questions, methodologies, key findings, and explicit limitations into a structured format.
-   **Intelligent SQLite Caching:** Calculates the SHA-256 hash of your PDFs. If a file has been processed before, its summary is instantly loaded from a local `research_cache.db` database, saving significant time and API costs.
-   **Multi-Agent Synthesis Pipeline:** Replaces generic summarisation with a rigorous 3-step analytical workflow:
    1.  **Synthesiser Agent:** Builds a cohesive narrative and identifies dominant methodologies across all papers.
    2.  **Critic Agent:** Deep-dives into the synthesis and raw summaries to rigorously extract unexplored territories, methodological flaws, and contradictions.
    3.  **Innovator Agent:** Uses the Critic's strict gaps to formulate 3 highly specific and novel research proposals.
-   **Async & Rate-Limited:** Uses `asyncio` for high-throughput concurrent processing, gated seamlessly by `aiolimiter` to ensure you respect Gemini API rate limits. Automatically retries transient network or API errors with exponential backoff using `tenacity`.

## Architecture Overview

The project offers both a Command Line Interface (CLI) and a Web App Interface.

The codebase is modularised to separate concerns securely:
-   `api.py`: The FastAPI backend serving the analysis pipeline for the web app.
-   `frontend/`: The React+Vite frontend featuring a professional academic design and PDF drag-and-drop.
-   `main.py`: The CLI entry point that handles argument parsing, database initialisation, and asynchronous orchestration.
-   `modules/llm.py`: Handles all direct interactions with the Gemini SDK, including file uploads, content generation, and strict cleanup in `finally` blocks to prevent orphaned files on your Google account.
-   `modules/db.py`: Wraps `aiosqlite` to handle the local database caching layer and asynchronous sha-256 file hashing.
-   `modules/agents.py`: Contains the logic for the 3-step sequential agent pipeline (Synthesiser -> Critic -> Innovator).
-   `modules/prompts.py`: Organises the instructions fed to the language models.
-   `modules/models.py`: Defines strict Pydantic models for data structuring throughout the application, enforcing predictable API outputs.

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
    The script requires a valid Gemini API key.
    Copy `.env.example` (if present) to `.env` or simply create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

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
*   `--model`: *(Optional)* The Gemini model ID to use. Defaults to `gemini-2.5-flash`.
*   `--rate-limit`: *(Optional)* Maximum number of requests allowed per minute to comply with your API tier limits. Defaults to `5`.
*   `--concurrent-requests`: *(Optional)* Maximum number of concurrent active requests. Adjust based on your system and network limits. Defaults to `5`.

### Example

```bash
# Process a folder named "sample_pdfs" about "agricultural microbiology"
# Save it to "advanced_report.md", processing 10 per minute
python main.py sample_pdfs --subject "agricultural microbiology" --output advanced_report.md --rate-limit 10
```

## Output

The script generates a comprehensive Markdown file (`.md`) containing:
1.  **State of the Field Synthesis:** The cohesive narrative built by the Synthesiser agent.
2.  **Critical Analysis of Research Gaps:** Deep flaws and contradictions identified by the Critic agent.
3.  **Formulated Research Proposals:** Actionable research ideas proposed by the Innovator agent.
4.  **Source Paper Summaries:** The individual Pydantic-structured summaries extracted from each PDF.
