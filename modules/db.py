import hashlib
import sqlite3
import aiosqlite
import logging

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
        await db.commit()
    logger.info("Database initialized.")

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
