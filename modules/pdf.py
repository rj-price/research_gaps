"""Local PDF text extraction.

Three ways a PDF can reach the model, selected by PDF_MODE:

  auto     - docling if it is installed, otherwise pypdf (the default)
  docling  - layout-aware conversion to Markdown: real tables, heading hierarchy,
             de-hyphenated text, and OCR for scans. Optional, heavier install.
  text     - pypdf's text layer. Zero extra dependencies, but tables are flattened
             and headings are lost, so the model has to infer document structure.
  native   - handled in modules.llm: send the PDF itself to OpenRouter's file parser.
"""
import functools
import importlib.util
import logging
import os

from pypdf import PdfReader

logger = logging.getLogger(__name__)

MAX_CHARS = 400_000  # roughly 100k tokens: ample for a paper, a guard against outliers


def docling_available() -> bool:
    return importlib.util.find_spec("docling") is not None


def pdf_mode() -> str:
    """Resolves PDF_MODE, expanding 'auto' to whichever local extractor is installed."""
    mode = os.getenv("PDF_MODE", "auto").lower()
    if mode == "auto":
        return "docling" if docling_available() else "text"
    if mode not in {"text", "docling", "native"}:
        logger.warning(f"Unknown PDF_MODE '{mode}'; falling back to 'text'.")
        return "text"
    if mode == "docling" and not docling_available():
        logger.warning("PDF_MODE=docling but docling is not installed; falling back to pypdf.")
        return "text"
    return mode


@functools.lru_cache(maxsize=1)
def _converter():
    """Docling loads layout models on first use, so the converter is built once and reused."""
    from docling.document_converter import DocumentConverter

    logger.info("Initialising docling converter (first call loads layout models)...")
    return DocumentConverter()


def extract_with_docling(pdf_path: str) -> str:
    """Converts a PDF to Markdown, preserving tables and heading structure."""
    result = _converter().convert(pdf_path)
    markdown = result.document.export_to_markdown()
    if not markdown.strip():
        raise ValueError(f"docling produced no text for {pdf_path}.")
    return markdown


def extract_with_pypdf(pdf_path: str, max_chars: int = MAX_CHARS) -> str:
    """Returns the raw text layer of a PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    total = 0

    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Could not extract page {number} of {pdf_path}: {e}")
            continue
        if not text.strip():
            continue
        pages.append(f"[page {number}]\n{text.strip()}")
        total += len(text)
        if total >= max_chars:
            logger.warning(f"{pdf_path} truncated at {max_chars} characters ({number} pages read).")
            break

    if not pages:
        raise ValueError(
            f"No extractable text in {pdf_path}. It is probably a scan; "
            "install docling (PDF_MODE=docling) or set PDF_MODE=native."
        )

    return "\n\n".join(pages)


def extract_pdf_text(pdf_path: str, mode: str = "text", max_chars: int = MAX_CHARS) -> str:
    """Extracts text using the requested local extractor, falling back to pypdf on failure."""
    if mode == "docling":
        try:
            text = extract_with_docling(pdf_path)
            return text[:max_chars] if len(text) > max_chars else text
        except Exception as e:
            # A docling failure should degrade to a worse summary, not lose the paper
            logger.warning(f"docling failed on {pdf_path} ({e}); falling back to pypdf.")
    return extract_with_pypdf(pdf_path, max_chars)
