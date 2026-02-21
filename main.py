import os
import argparse
import glob
import asyncio
from typing import List

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Load environment variables
load_dotenv()

class PaperSummary(BaseModel):
    title: str = Field(description="The title or topic of the paper (infer if not explicit).")
    core_research_question: str = Field(description="What problem does this paper solve?")
    methodology: str = Field(description="A brief overview of the methods used.")
    key_findings: str = Field(description="The main results of the paper.")
    limitations: str = Field(description="What did the authors explicitly state as limitations or future directions?")

    def to_markdown(self) -> str:
        return (
            f"1. **Title/Topic**: {self.title}\n"
            f"2. **Core Research Question**: {self.core_research_question}\n"
            f"3. **Methodology**: {self.methodology}\n"
            f"4. **Key Findings**: {self.key_findings}\n"
            f"5. **Limitations & Future Work**: {self.limitations}\n"
        )


def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file with your GOOGLE_API_KEY.")
        exit(1)
    return genai.Client(api_key=api_key)


async def summarize_paper(
    client: genai.Client, model_id: str, pdf_path: str, semaphore: asyncio.Semaphore
) -> str:
    """Uploads and uses Gemini to summarize a single paper."""
    filename = os.path.basename(pdf_path)
    
    async with semaphore:
        try:
            # 1. Upload via File API
            uploaded_file = client.files.upload(file=pdf_path)
            
            # 2. System Instruction + Structured Output (Pydantic)
            config = types.GenerateContentConfig(
                system_instruction="You are an expert academic researcher.",
                response_mime_type="application/json",
                response_schema=PaperSummary,
            )
            
            prompt = f"""
            Summarize the academic paper content in the provided document (extracted from {filename}). 
            Focus specifically on information relevant to identifying research gaps.
            """
            
            # Using client.aio for async generation
            response = await client.aio.models.generate_content(
                model=model_id,
                contents=[uploaded_file, prompt],
                config=config
            )
            
            # Cleanup File
            client.files.delete(name=uploaded_file.name)
            
            # The response text is a JSON string matching the PaperSummary schema
            summary_data = PaperSummary.model_validate_json(response.text)
            return summary_data.to_markdown()

        except Exception as e:
            return f"Error summarizing {filename}: {e}\n"
        finally:
            # Add a small delay between requests to help respect free-tier rate limits
            await asyncio.sleep(20)


async def identify_gaps(client: genai.Client, model_id: str, summaries: List[str], subject: str) -> str:
    """Uses Gemini to synthesize summaries and find gaps."""
    combined_summaries = "\n\n---\n\n".join(summaries)

    prompt = f"""
    You have read the following summaries of papers regarding "{subject}".
    
    Your goal is to identify **Research Gaps** to formulate a new research proposal.
    
    Analyze the collection and provide a report containing:
    1. **State of the Field**: A brief synthesis of what is known based on these papers.
    2. **Identified Gaps**:
       - What questions remain unanswered?
       - Are there contradictions between papers?
       - Is there a methodology that hasn't been applied yet?
    3. **Proposal Ideas**: Suggest 3 specific, novel research questions or titles for a new study that would address these gaps.
    
    Summaries:
    {combined_summaries}
    """
    
    config = types.GenerateContentConfig(
        system_instruction="You are a senior principal investigator.",
    )
    
    try:
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        return f"Error analyzing gaps: {e}"


async def process_pdfs(args) -> None:
    client = get_client()
    model_id = "gemini-2.5-flash"

    pdf_files = glob.glob(os.path.join(args.folder, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {args.folder}")
        return

    print(f"Found {len(pdf_files)} PDFs. Processing...")

    # Semaphore to limit concurrent requests (adjust based on tier limits; free tier is ~5 RPM)
    # Using 1 to be completely safe against the 5 RPM limit.
    semaphore = asyncio.Semaphore(1)

    # Process all PDFs concurrently
    tasks = [
        summarize_paper(client, model_id, pdf, semaphore)
        for pdf in pdf_files
    ]
    
    paper_summaries = []
    
    # Use tqdm wrapper for asyncio to show progress bar
    for f in tqdm.as_completed(tasks, total=len(tasks), desc="Summarizing Papers"):
        summary = await f
        paper_summaries.append(summary)

    if not paper_summaries:
        print("No valid summaries generated.")
        return

    print("\nAnalyzing research gaps...")
    report = await identify_gaps(client, model_id, paper_summaries, args.subject)

    with open(args.output, "w") as f:
        f.write(f"# Research Gap Analysis: {args.subject}\n\n")
        f.write(report)
        f.write("\n\n## Source Paper Summaries\n\n")
        for i, summary in enumerate(paper_summaries):
            f.write(f"### Paper {i + 1}\n{summary}\n\n")

    print(f"Analysis complete! Report saved to {args.output}")

def _silence_ssl_errors():
    import logging
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)

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

    args = parser.parse_args()
    
    # Run the async main loop
    try:
        asyncio.run(process_pdfs(args))
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise

if __name__ == "__main__":
    main()
