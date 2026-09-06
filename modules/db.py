import hashlib
import json
import sqlite3
import aiosqlite
import logging
from datetime import datetime, timezone
from typing import Iterable, List

logger = logging.getLogger(__name__)

DB_PATH = "research_cache.db"

async def init_db():
    """Initializes the SQLite database with the required tables."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                file_hash TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                json_data TEXT NOT NULL
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS gaps (
                gap_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT,
                authors TEXT,
                published TEXT,
                url TEXT,
                matched_terms TEXT,
                screened INTEGER NOT NULL DEFAULT 0,
                relevant INTEGER,
                relevance_score REAL,
                relevance_reason TEXT,
                first_seen TEXT NOT NULL,
                digested INTEGER NOT NULL DEFAULT 0
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS gap_matches (
                paper_id TEXT NOT NULL,
                gap_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (paper_id, gap_id)
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS watch_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                window_start TEXT,
                fetched INTEGER DEFAULT 0,
                new_papers INTEGER DEFAULT 0,
                relevant INTEGER DEFAULT 0,
                matches INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'running'
            )
        ''')
        await db.commit()
    logger.info("Database initialized.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_gap_id(subject: str, title: str) -> str:
    """A stable ID for a gap, so re-running the analysis updates rather than duplicates."""
    key = f"{subject.strip().lower()}|{title.strip().lower()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# --- Gap store ---

async def store_gaps(subject: str, gaps: Iterable) -> List[str]:
    """Persists IdentifiedGap objects. Existing gaps keep their created_at and status."""
    gap_ids = []
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        for gap in gaps:
            gap_id = make_gap_id(subject, gap.title)
            gap_ids.append(gap_id)
            await db.execute('''
                INSERT INTO gaps (gap_id, subject, category, title, description, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'open', ?, ?)
                ON CONFLICT(gap_id) DO UPDATE SET
                    category = excluded.category,
                    description = excluded.description,
                    updated_at = excluded.updated_at
            ''', (gap_id, subject, gap.category, gap.title, gap.description, now, now))
        await db.commit()
    logger.info(f"Stored {len(gap_ids)} research gaps for subject '{subject}'.")
    return gap_ids


async def get_gaps(status: str | None = "open") -> List[dict]:
    query = 'SELECT gap_id, subject, category, title, description, status, created_at FROM gaps'
    params: tuple = ()
    if status:
        query += ' WHERE status = ?'
        params = (status,)
    query += ' ORDER BY created_at DESC'
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cursor:
            return [dict(row) for row in await cursor.fetchall()]


async def set_gap_status(gap_id: str, status: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('UPDATE gaps SET status = ?, updated_at = ? WHERE gap_id = ?', (status, _now(), gap_id))
        await db.commit()


# --- Paper store ---

async def filter_new_papers(paper_ids: Iterable[str]) -> set:
    """Returns the subset of IDs not already seen, so each paper is screened once only."""
    ids = list(paper_ids)
    if not ids:
        return set()
    placeholders = ','.join('?' * len(ids))
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(f'SELECT paper_id FROM papers WHERE paper_id IN ({placeholders})', ids) as cursor:
            known = {row[0] for row in await cursor.fetchall()}
    return set(ids) - known


async def store_papers(papers: Iterable):
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        for paper in papers:
            await db.execute('''
                INSERT OR IGNORE INTO papers
                    (paper_id, source, title, abstract, authors, published, url, matched_terms, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (paper.paper_id, paper.source, paper.title, paper.abstract, paper.authors,
                  paper.published, paper.url, json.dumps(paper.matched_terms), now))
        await db.commit()


async def record_screening(paper_id: str, relevant: bool, score: float, reason: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            UPDATE papers SET screened = 1, relevant = ?, relevance_score = ?, relevance_reason = ?
            WHERE paper_id = ?
        ''', (int(relevant), score, reason, paper_id))
        await db.commit()


async def record_gap_match(paper_id: str, gap_id: str, relationship: str, confidence: float, evidence: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            INSERT OR REPLACE INTO gap_matches (paper_id, gap_id, relationship, confidence, evidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (paper_id, gap_id, relationship, confidence, evidence, _now()))
        await db.commit()


async def get_undigested(limit: int = 200) -> List[dict]:
    """Relevant, screened papers that have not yet appeared in a digest."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute('''
            SELECT * FROM papers
            WHERE screened = 1 AND relevant = 1 AND digested = 0
            ORDER BY published DESC LIMIT ?
        ''', (limit,)) as cursor:
            papers = [dict(row) for row in await cursor.fetchall()]

        for paper in papers:
            async with db.execute('''
                SELECT m.gap_id, m.relationship, m.confidence, m.evidence, g.title, g.subject, g.category
                FROM gap_matches m JOIN gaps g ON g.gap_id = m.gap_id
                WHERE m.paper_id = ? ORDER BY m.confidence DESC
            ''', (paper["paper_id"],)) as cursor:
                paper["matches"] = [dict(row) for row in await cursor.fetchall()]
    return papers


async def mark_digested(paper_ids: Iterable[str]):
    ids = list(paper_ids)
    if not ids:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany('UPDATE papers SET digested = 1 WHERE paper_id = ?', [(i,) for i in ids])
        await db.commit()


# --- Run bookkeeping ---

async def start_run(window_start: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            'INSERT INTO watch_runs (started_at, window_start) VALUES (?, ?)', (_now(), window_start)
        )
        await db.commit()
        return cursor.lastrowid


async def finish_run(run_id: int, fetched: int, new_papers: int, relevant: int, matches: int, status: str = "ok"):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            UPDATE watch_runs SET finished_at = ?, fetched = ?, new_papers = ?, relevant = ?, matches = ?, status = ?
            WHERE run_id = ?
        ''', (_now(), fetched, new_papers, relevant, matches, status, run_id))
        await db.commit()


async def last_successful_run() -> str | None:
    """ISO timestamp of the last completed run, used as the default fetch window start."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT started_at FROM watch_runs WHERE status = 'ok' ORDER BY run_id DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

def get_file_hash(filepath: str) -> str:
    """Calculates the SHA-256 hash of a file synchronously.
       To be run via asyncio.to_thread in the main loop to prevent blocking.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

async def get_cached_summary(file_hash: str) -> str | None:
    """Retrieves the JSON string of a cached summary if it exists."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute('SELECT json_data FROM summaries WHERE file_hash = ?', (file_hash,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0]
    return None

async def cache_summary(file_hash: str, filename: str, json_data: str):
    """Stores the summary JSON string in the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            INSERT OR REPLACE INTO summaries (file_hash, filename, json_data)
            VALUES (?, ?, ?)
        ''', (file_hash, filename, json_data))
        await db.commit()
