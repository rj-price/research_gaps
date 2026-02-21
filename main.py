import os
import argparse
import glob
import time
from typing import List

from google import genai
from pypdf import PdfReader
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

def get_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file with your GOOGLE_API_KEY.")
        exit(1)
    return genai.Client(api_key=api_key)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def summarize_paper(client, model_id: str, text: str, filename: str) -> str:
    """Uses Gemini to summarize a single paper."""
    prompt = f"""
    You are an expert academic researcher. Summarize the following academic paper content (extracted from {filename}). 
    Focus specifically on information relevant to identifying research gaps. 
    
    Structure your summary exactly as follows:
    1. **Title/Topic**: (Infer if not explicit)
    2. **Core Research Question**: What problem does this paper solve?
    3. **Methodology**: Brief overview of methods used.
    4. **Key Findings**: The main results.
    5. **Limitations & Future Work**: What did the authors explicitly state as limitations or future directions?
    
    Paper Content:
    {text[:100000]}  # Context window is much larger in latest models.
    """
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error summarizing {filename}: {e}"


def identify_gaps(client, model_id: str, summaries: List[str], subject: str) -> str:
    """Uses Gemini to synthesize summaries and find gaps."""
    combined_summaries = "\n\n---\n\n".join(summaries)

    prompt = f"""
    You are a senior principal investigator. You have read the following summaries of papers regarding "{subject}".
    
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
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error analyzing gaps: {e}"


def main():
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

    client = get_client()
    model_id = "gemini-2.5-flash"

    pdf_files = glob.glob(os.path.join(args.folder, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {args.folder}")
        return

    print(f"Found {len(pdf_files)} PDFs. Processing...")

    paper_summaries = []

    for pdf_file in tqdm(pdf_files, desc="Summarizing Papers"):
        text = extract_text_from_pdf(pdf_file)
        if text.strip():
            summary = summarize_paper(client, model_id, text, os.path.basename(pdf_file))
            paper_summaries.append(summary)
            # Add a small delay to avoid hitting rate limits on free tier
            time.sleep(20) 
        else:
            print(f"Skipping empty or unreadable file: {pdf_file}")

    if not paper_summaries:
        print("No valid text extracted from PDFs.")
        return

    print("\nAnalyzing research gaps...")
    report = identify_gaps(client, model_id, paper_summaries, args.subject)

    with open(args.output, "w") as f:
        f.write(f"# Research Gap Analysis: {args.subject}\n\n")
        f.write(report)
        f.write("\n\n## Source Paper Summaries\n\n")
        for i, summary in enumerate(paper_summaries):
            f.write(f"### Paper {i + 1}\n{summary}\n\n")

    print(f"Analysis complete! Report saved to {args.output}")


if __name__ == "__main__":
    main()
