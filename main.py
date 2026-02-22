import os
import argparse
import glob
import asyncio
import logging
from typing import List

from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from aiolimiter import AsyncLimiter

from modules.llm import get_client, summarize_paper, identify_gaps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def process_pdfs(args: argparse.Namespace) -> None:
    try:
        client = get_client()
    except ValueError as e:
        logger.error(str(e))
        return

    pdf_files = glob.glob(os.path.join(args.folder, "*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {args.folder}")
        return

    logger.info(f"Found {len(pdf_files)} PDFs. Processing...")

    # Rate Limiter
    # Limiter allows max rates per period. We define rate_limit requests per 60 seconds.
    limiter = AsyncLimiter(args.rate_limit, 60)
    
    # Semaphore to limit max concurrent active requests to not overwhelm local connections
    semaphore = asyncio.Semaphore(args.concurrent_requests)

    async def bounded_summarize(pdf_path: str) -> str:
        async with semaphore:
            return await summarize_paper(client, args.model, pdf_path, limiter)

    # Process all PDFs concurrently but gated by both limiter and semaphore
    tasks = [
        bounded_summarize(pdf)
        for pdf in pdf_files
    ]
    
    paper_summaries = []
    
    # Use tqdm wrapper for asyncio to show progress bar
    for f in tqdm.as_completed(tasks, total=len(tasks), desc="Summarizing Papers"):
        summary = await f
        paper_summaries.append(summary)

    # Filter out exact error messages if any (optional, but good for clean output)
    valid_summaries = [s for s in paper_summaries if not s.startswith("Error summarizing")]

    if not valid_summaries:
        logger.error("No valid summaries generated.")
        return

    logger.info("Analyzing research gaps...")
    report = await identify_gaps(client, args.model, valid_summaries, args.subject, limiter)

    with open(args.output, "w") as f:
        f.write(f"# Research Gap Analysis: {args.subject}\n\n")
        f.write(report)
        f.write("\n\n## Source Paper Summaries\n\n")
        for i, summary in enumerate(valid_summaries):
            f.write(f"### Paper {i + 1}\n{summary}\n\n")

    logger.info(f"Analysis complete! Report saved to {args.output}")

from modules.db import init_db

# ... (keep existing process_pdfs and imports above) ...
def _silence_ssl_errors():
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)

async def _main_async(args):
    await init_db()
    await process_pdfs(args)

def main():
    _silence_ssl_errors()
    parser = argparse.ArgumentParser(description="Academic Research Gap Identifier")
    parser.add_argument("folder", help="Path to the folder containing PDF papers")
    parser.add_argument(
        "--subject",
        help="The general subject matter (optional, inferred if skipped)",
        default="the provided topics",
    )
    parser.add_argument(
        "--output", help="Output file for the report", default="research_gap_report.md"
    )
    parser.add_argument(
        "--model", help="Gemini model ID to use", default="gemini-2.5-flash"
    )
    parser.add_argument(
        "--rate-limit", help="Max requests per minute", type=int, default=5
    )
    parser.add_argument(
        "--concurrent-requests", help="Max concurrent requests", type=int, default=5
    )

    args = parser.parse_args()
    
    # Run the async main loop
    try:
        asyncio.run(_main_async(args))
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")

if __name__ == "__main__":
    main()
