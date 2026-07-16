import logging
import os
import random
import asyncio
from typing import Dict, Any

try:
    import aiosqlite
except ImportError:
    pass

logger = logging.getLogger(__name__)

SECURE_PRAGMAS = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "foreign_keys": "ON",
    "secure_delete": "ON",
    "temp_store": "MEMORY",
    "cache_size": "-32000",
    "busy_timeout": "5000",
}

async def apply_secure_pragmas(connection) -> None:
    """Applies secure PRAGMAs to the connection."""
    for key, value in SECURE_PRAGMAS.items():
        await connection.execute(f"PRAGMA {key}={value}")
    await connection.commit()

def set_file_permissions(db_path: str) -> None:
    """Sets strict file permissions on the database file."""
    if not os.path.exists(db_path):
        return

    if os.name == "nt":
        logger.warning(f"Windows detected. Ensure strict ACLs manually for {db_path}.")
    else:
        try:
            os.chmod(db_path, 0o600)
        except OSError as e:
            logger.error(f"Failed to set file permissions on {db_path}: {e}")

class RetryingDatabaseConnection:
    """Async context manager that wraps aiosqlite.connect with exponential backoff."""
    def __init__(self, db_path: str, max_retries: int = 5, base_delay: float = 0.1, max_delay: float = 5.0):
        self.db_path = db_path
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.connection = None

    async def __aenter__(self) -> 'aiosqlite.Connection':
        retries = 0
        while retries <= self.max_retries:
            try:
                self.connection = await aiosqlite.connect(self.db_path, timeout=30.0)
                await apply_secure_pragmas(self.connection)
                return self.connection
            except Exception as e: # Catching generic Exception as aiosqlite/sqlite3 errors might vary
                if "database is locked" in str(e).lower():
                    if retries == self.max_retries:
                        raise e
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    # Add jitter
                    delay = delay * (0.5 + random.random())
                    logger.warning(f"Database locked, retrying in {delay:.2f}s... (Attempt {retries+1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    retries += 1
                else:
                    raise e

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.connection:
            await self.connection.close()

async def check_database_integrity(db_path: str) -> Dict[str, Any]:
    """Runs PRAGMA integrity_check and freelist_count."""
    result = {"status": "unknown"}
    try:
        async with RetryingDatabaseConnection(db_path) as conn:
            async with conn.execute("PRAGMA integrity_check") as cursor:
                row = await cursor.fetchone()
                result["integrity_check"] = row[0] if row else "unknown"
            
            async with conn.execute("PRAGMA freelist_count") as cursor:
                row = await cursor.fetchone()
                result["freelist_count"] = row[0] if row else 0
                
            result["status"] = "ok" if result.get("integrity_check") == "ok" else "error"
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "error"
    return result

def prepare_sqlcipher_hooks() -> Dict[str, str]:
    """
    Returns SQLCipher PRAGMAs for transparent database encryption.
    NOTE: Requires pysqlcipher3 or sqlcipher3 to be installed and used instead of sqlite3/aiosqlite.
    """
    return {
        "key": "'your-secure-encryption-key'", 
        "cipher_page_size": "4096",
        "kdf_iter": "256000",
        "cipher_use_hmac": "OFF",
        "cipher_plaintext_header_size": "32"
    }
