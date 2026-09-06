"""Literature sources for the ambient watcher.

Each fetcher returns WatchedPaper records for a date window. Neither source needs
authentication; an NCBI API key is used if NCBI_API_KEY is set, purely to raise
the PubMed rate limit from 3 to 10 requests per second.
"""
import os
import asyncio
import logging
import re
from datetime import date
from typing import List
from xml.etree import ElementTree

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from modules.models import WatchedPaper

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BIORXIV_API = "https://api.biorxiv.org/details"
USER_AGENT = "research-gaps-watcher/1.0 (https://github.com/rj-price/research_gaps)"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15), reraise=True)
async def _get(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    response = await client.get(url, **kwargs)
    response.raise_for_status()
    return response


def _matched_terms(text: str, terms: List[str]) -> List[str]:
    """Whole-word matching, so 'plant' does not match 'implant' or 'Rubus' 'Rubuscus'."""
    lowered = text.lower()
    matched = []
    for term in terms:
        pattern = r"\b" + r"\s+".join(re.escape(word) for word in term.lower().split()) + r"\b"
        if re.search(pattern, lowered):
            matched.append(term)
    return matched


# --- PubMed ---

def build_pubmed_query(terms: List[str], extra: str = "") -> str:
    """ORs the watch terms across title and abstract, ANDed with any extra qualifier."""
    organism_clause = " OR ".join(f'"{term}"[Title/Abstract]' for term in terms)
    query = f"({organism_clause})"
    if extra:
        query = f"{query} AND ({extra})"
    return query


def _text(node, path: str, default: str = "") -> str:
    found = node.find(path)
    if found is None:
        return default
    return "".join(found.itertext()).strip() or default


def parse_pubmed_xml(xml: str, terms: List[str]) -> List[WatchedPaper]:
    papers = []
    root = ElementTree.fromstring(xml)
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article, "MedlineCitation/PMID")
        if not pmid:
            continue
        title = _text(article, "MedlineCitation/Article/ArticleTitle", "Untitled")
        abstract = " ".join(
            "".join(part.itertext()).strip()
            for part in article.findall("MedlineCitation/Article/Abstract/AbstractText")
        ).strip()

        authors = []
        for author in article.findall("MedlineCitation/Article/AuthorList/Author")[:8]:
            surname = _text(author, "LastName")
            initials = _text(author, "Initials")
            if surname:
                authors.append(f"{surname} {initials}".strip())

        pub = article.find("MedlineCitation/Article/Journal/JournalIssue/PubDate")
        published = ""
        if pub is not None:
            year = _text(pub, "Year")
            month = _text(pub, "Month")
            published = "-".join(bit for bit in (year, month) if bit) or _text(pub, "MedlineDate")

        doi = ""
        for ident in article.findall("PubmedData/ArticleIdList/ArticleId"):
            if ident.get("IdType") == "doi":
                doi = (ident.text or "").strip()

        papers.append(WatchedPaper(
            paper_id=f"pubmed:{pmid}",
            source="pubmed",
            title=title,
            abstract=abstract,
            authors=", ".join(authors),
            published=published,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if not doi else f"https://doi.org/{doi}",
            matched_terms=_matched_terms(f"{title} {abstract}", terms),
        ))
    return papers


async def fetch_pubmed(
    client: httpx.AsyncClient, terms: List[str], start: date, end: date,
    extra_query: str = "", retmax: int = 200,
) -> List[WatchedPaper]:
    """Searches PubMed for the watch terms published in [start, end]."""
    api_key = os.getenv("NCBI_API_KEY")
    common = {"db": "pubmed", "retmode": "xml"}
    if api_key:
        common["api_key"] = api_key

    search_params = {
        **common,
        "term": build_pubmed_query(terms, extra_query),
        "datetype": "edat",  # entry date: catches papers indexed since the last run
        "mindate": start.strftime("%Y/%m/%d"),
        "maxdate": end.strftime("%Y/%m/%d"),
        "retmax": str(retmax),
        "sort": "date",
    }
    response = await _get(client, f"{EUTILS}/esearch.fcgi", params=search_params)
    root = ElementTree.fromstring(response.text)
    pmids = [node.text for node in root.findall(".//IdList/Id") if node.text]

    if not pmids:
        logger.info("PubMed returned no hits for this window.")
        return []

    logger.info(f"PubMed returned {len(pmids)} hits; fetching abstracts.")
    papers: List[WatchedPaper] = []
    for chunk_start in range(0, len(pmids), 100):
        chunk = pmids[chunk_start:chunk_start + 100]
        fetch_params = {**common, "id": ",".join(chunk), "rettype": "abstract"}
        response = await _get(client, f"{EUTILS}/efetch.fcgi", params=fetch_params)
        papers.extend(parse_pubmed_xml(response.text, terms))
        await asyncio.sleep(0.15 if api_key else 0.4)  # stay inside NCBI's rate limit
    return papers


# --- bioRxiv / medRxiv ---

# The date-interval API returns 100 records a page and offers no keyword search, so a
# week of bioRxiv has to be walked in full and filtered locally. At 100 records a page
# this covers 40,000 preprints, comfortably more than any weekly window; the old 40-page
# cap truncated a single week silently. The watcher is a scheduled background job, so
# minutes of paging cost nothing that matters.
DEFAULT_PREPRINT_MAX_PAGES = 400


async def fetch_preprints(
    client: httpx.AsyncClient, terms: List[str], start: date, end: date,
    server: str = "biorxiv", max_pages: int = DEFAULT_PREPRINT_MAX_PAGES,
) -> List[WatchedPaper]:
    """Walks the preprint server's date-interval API and keeps records mentioning a watch term.

    The API has no keyword search, so filtering is done here on title and abstract.
    """
    papers: List[WatchedPaper] = []
    cursor = 0
    total = 0
    seen_dois = set()

    for _ in range(max_pages):
        url = f"{BIORXIV_API}/{server}/{start.isoformat()}/{end.isoformat()}/{cursor}/json"
        response = await _get(client, url)
        payload = response.json()
        collection = payload.get("collection") or []
        if not collection:
            break

        for item in collection:
            doi = (item.get("doi") or "").strip()
            if not doi or doi in seen_dois:
                continue
            seen_dois.add(doi)
            title = (item.get("title") or "").strip()
            abstract = (item.get("abstract") or "").strip()
            matched = _matched_terms(f"{title} {abstract}", terms)
            if not matched:
                continue
            papers.append(WatchedPaper(
                paper_id=f"{server}:{doi}",
                source=server,
                title=title,
                abstract=abstract,
                authors=(item.get("authors") or "").strip(),
                published=(item.get("date") or "").strip(),
                url=f"https://doi.org/{doi}",
                matched_terms=matched,
            ))

        messages = payload.get("messages") or [{}]
        total = int(messages[0].get("total", 0) or 0)
        cursor += len(collection)
        if total == 0 or cursor >= total:
            break
        await asyncio.sleep(0.2)
    else:
        logger.warning(
            f"Stopped paging {server} at the {max_pages}-page cap ({cursor} of {total} records); "
            f"window is incomplete. Raise preprint_max_pages in the watchlist or shorten the window."
        )

    logger.info(f"{server}: {len(papers)} preprints matched the watch terms.")
    return papers


async def fetch_all(
    terms: List[str], start: date, end: date,
    preprint_servers: List[str] | None = None, extra_query: str = "", retmax: int = 200,
    preprint_max_pages: int = DEFAULT_PREPRINT_MAX_PAGES,
) -> List[WatchedPaper]:
    """Fetches from every configured source, tolerating a single source being down."""
    preprint_servers = preprint_servers if preprint_servers is not None else ["biorxiv"]
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient(timeout=60.0, headers=headers, follow_redirects=True) as client:
        jobs = [fetch_pubmed(client, terms, start, end, extra_query, retmax)]
        jobs += [
            fetch_preprints(client, terms, start, end, server, preprint_max_pages)
            for server in preprint_servers
        ]
        results = await asyncio.gather(*jobs, return_exceptions=True)

    papers: List[WatchedPaper] = []
    labels = ["pubmed"] + preprint_servers
    for label, result in zip(labels, results):
        if isinstance(result, Exception):
            logger.error(f"Source '{label}' failed: {result}")
            continue
        papers.extend(result)

    # De-duplicate a preprint that has since been indexed in PubMed under the same DOI
    by_url = {}
    for paper in papers:
        by_url.setdefault(paper.url, paper)
    return list(by_url.values())
