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
        await db.execute('''
            CREATE TABLE IF NOT EXISTS gap_sources (
                gap_id TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (gap_id, paper_id)
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS synthesis_log (
                paper_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (paper_id, subject)
            )
        ''')
        await _add_missing_columns(db)
        await db.commit()
    logger.info("Database initialized.")


# Columns added after the first release. SQLite has no "ADD COLUMN IF NOT EXISTS" and an
# existing research_cache.db must survive the upgrade, so they are applied by hand.
_ADDED_COLUMNS = {
    "papers": [("synthesised", "INTEGER NOT NULL DEFAULT 0")],
    "gaps": [("origin", "TEXT NOT NULL DEFAULT 'full_text'")],
}


async def _add_missing_columns(db):
    for table, columns in _ADDED_COLUMNS.items():
        async with db.execute(f"PRAGMA table_info({table})") as cursor:
            existing = {row[1] for row in await cursor.fetchall()}
        for name, definition in columns:
            if name not in existing:
                await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
                logger.info(f"Migrated {table}: added column {name}.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_gap_id(subject: str, title: str) -> str:
    """A stable ID for a gap, so re-running the analysis updates rather than duplicates."""
    key = f"{subject.strip().lower()}|{title.strip().lower()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# --- Gap store ---

async def store_gaps(
    subject: str, gaps: Iterable, origin: str = "full_text", source_papers: Iterable[str] = (),
) -> List[str]:
    """Persists IdentifiedGap objects. Existing gaps keep their created_at and status.

    `origin` records what the gap was derived from: 'full_text' for the PDF pipeline,
    'abstract' for the rolling watcher. Abstracts carry no stated limitations, so gaps
    drawn from them are weaker evidence and the digest says so.

    `source_papers` are the paper_ids the gap was drawn from. They are excluded from
    later gap matching, so a paper can never be reported as filling its own gap.
    """
    gap_ids = []
    sources = list(source_papers)
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        for gap in gaps:
            gap_id = make_gap_id(subject, gap.title)
            gap_ids.append(gap_id)
            await db.execute('''
                INSERT INTO gaps (gap_id, subject, category, title, description, status, origin, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?)
                ON CONFLICT(gap_id) DO UPDATE SET
                    category = excluded.category,
                    description = excluded.description,
                    updated_at = excluded.updated_at
            ''', (gap_id, subject, gap.category, gap.title, gap.description, origin, now, now))
            for paper_id in sources:
                await db.execute(
                    'INSERT OR IGNORE INTO gap_sources (gap_id, paper_id, created_at) VALUES (?, ?, ?)',
                    (gap_id, paper_id, now),
                )
        await db.commit()
    logger.info(f"Stored {len(gap_ids)} research gaps for subject '{subject}'.")
    return gap_ids


async def merge_into_gap(gap_id: str, description: str, source_papers: Iterable[str] = ()) -> None:
    """Folds a rediscovered gap into the one already stored rather than adding a duplicate.

    The title is left alone: it is the gap_id's input, so changing it would orphan every
    match already recorded against the gap.
    """
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            'UPDATE gaps SET description = ?, updated_at = ? WHERE gap_id = ?', (description, now, gap_id)
        )
        for paper_id in source_papers:
            await db.execute(
                'INSERT OR IGNORE INTO gap_sources (gap_id, paper_id, created_at) VALUES (?, ?, ?)',
                (gap_id, paper_id, now),
            )
        await db.commit()


async def get_gap_sources(gap_ids: Iterable[str] | None = None) -> dict:
    """Maps gap_id to the set of paper_ids that gap was derived from."""
    query = 'SELECT gap_id, paper_id FROM gap_sources'
    params: tuple = ()
    ids = list(gap_ids) if gap_ids is not None else None
    if ids is not None:
        if not ids:
            return {}
        query += f" WHERE gap_id IN ({','.join('?' * len(ids))})"
        params = tuple(ids)
    sources: dict = {}
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(query, params) as cursor:
            for gap_id, paper_id in await cursor.fetchall():
                sources.setdefault(gap_id, set()).add(paper_id)
    return sources


async def get_gaps(status: str | None = "open") -> List[dict]:
    query = 'SELECT gap_id, subject, category, title, description, status, origin, created_at FROM gaps'
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


async def update_matched_terms(updates: Iterable) -> int:
    """Re-derives which watch terms a stored paper hits.

    matched_terms is computed once, at fetch time, so a term added to the watchlist later
    never appears against papers already stored — and since routing reads this field, such
    a paper can never reach the subject group the new term was added for.
    """
    rows = [(json.dumps(terms), paper_id) for paper_id, terms in updates]
    if not rows:
        return 0
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany('UPDATE papers SET matched_terms = ? WHERE paper_id = ?', rows)
        await db.commit()
    return len(rows)


async def reset_screening() -> int:
    """Clears every screening verdict so the papers can be judged again.

    Tightening `interests` only changes how future papers are screened; papers already in
    the store keep the verdict they were given under the old wording, and stay in the
    synthesis backlog on that basis. This puts them back in the queue.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            'UPDATE papers SET screened = 0, relevant = NULL, relevance_score = NULL, relevance_reason = NULL'
        )
        await db.commit()
        return cursor.rowcount


async def get_all_papers() -> List[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            'SELECT paper_id, source, title, abstract, authors, published, url, matched_terms '
            'FROM papers ORDER BY published ASC'
        ) as cursor:
            return [dict(row) for row in await cursor.fetchall()]


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
                SELECT m.gap_id, m.relationship, m.confidence, m.evidence, g.title, g.subject, g.category, g.origin
                FROM gap_matches m JOIN gaps g ON g.gap_id = m.gap_id
                WHERE m.paper_id = ? ORDER BY m.confidence DESC
            ''', (paper["paper_id"],)) as cursor:
                paper["matches"] = [dict(row) for row in await cursor.fetchall()]
    return papers


async def get_synthesis_backlog(limit: int = 2000) -> List[dict]:
    """Relevant papers not yet consumed by a rolling synthesis, oldest first.

    Oldest first so a subject that trickles in is synthesised in publication order rather
    than being permanently pushed out of the window by newer arrivals. The limit is
    generous because a paper routing to no subject stays here indefinitely, waiting for a
    subject group that might cover it later.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute('''
            SELECT paper_id, title, abstract, authors, published, url, source, matched_terms,
                   relevance_reason
            FROM papers
            WHERE screened = 1 AND relevant = 1 AND synthesised = 0
            ORDER BY published ASC LIMIT ?
        ''', (limit,)) as cursor:
            return [dict(row) for row in await cursor.fetchall()]


async def mark_synthesised(paper_ids: Iterable[str], subject: str):
    """Records that these papers have been synthesised *for this subject*.

    Per subject, not globally: subject groups overlap by design (a group on the species
    complex covers papers a forma specialis group also claims), and a global flag let
    whichever subject ran first consume the paper out from under the other.
    """
    ids = list(paper_ids)
    if not ids:
        return
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            'INSERT OR IGNORE INTO synthesis_log (paper_id, subject, created_at) VALUES (?, ?, ?)',
            [(paper_id, subject, now) for paper_id in ids],
        )
        await db.commit()


async def get_synthesised_subjects(paper_ids: Iterable[str] | None = None) -> dict:
    """Maps paper_id to the set of subjects it has already been synthesised for."""
    query = 'SELECT paper_id, subject FROM synthesis_log'
    params: tuple = ()
    ids = list(paper_ids) if paper_ids is not None else None
    if ids is not None:
        if not ids:
            return {}
        query += f" WHERE paper_id IN ({','.join('?' * len(ids))})"
        params = tuple(ids)
    consumed: dict = {}
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(query, params) as cursor:
            for paper_id, subject in await cursor.fetchall():
                consumed.setdefault(paper_id, set()).add(subject)
    return consumed


async def mark_fully_synthesised(paper_ids: Iterable[str]):
    """Flags papers consumed by every subject they route to, keeping the backlog query cheap."""
    ids = list(paper_ids)
    if not ids:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany('UPDATE papers SET synthesised = 1 WHERE paper_id = ?', [(i,) for i in ids])
        await db.commit()


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
