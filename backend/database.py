"""
backend/database.py
====================
Async SQLite database layer using aiosqlite.

Schema:
    jobs          — tracks processing job state (submitted → processing → done/failed)
    feedback_log  — stores all generated feedback records for longitudinal analysis
    misconceptions — aggregated misconception counts per student/class

All writes use WAL mode for concurrent read performance on edge deployments.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import aiosqlite

logger = logging.getLogger("Database")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./socratica.db")

# Strip the driver prefix to get the raw file path
def _db_path() -> str:
    url = DATABASE_URL
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return "./socratica.db"

DB_PATH = _db_path()

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT    PRIMARY KEY,
    student_id      TEXT    NOT NULL DEFAULT 'anonymous',
    status          TEXT    NOT NULL DEFAULT 'queued',   -- queued|processing|done|failed
    filename        TEXT,
    domain          TEXT,
    submitted_at    REAL    NOT NULL,
    updated_at      REAL    NOT NULL,
    error_message   TEXT,
    result_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_student ON jobs(student_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);

CREATE TABLE IF NOT EXISTS feedback_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT    NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    student_id      TEXT    NOT NULL DEFAULT 'anonymous',
    domain          TEXT,
    feedback_type   TEXT,   -- 'socratic'|'corrective'|'affirmative'
    feedback_text   TEXT,
    misconception_class TEXT,
    confidence      REAL,
    created_at      REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_student ON feedback_log(student_id);
CREATE INDEX IF NOT EXISTS idx_feedback_misc    ON feedback_log(misconception_class);

CREATE TABLE IF NOT EXISTS misconception_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      TEXT    NOT NULL DEFAULT 'anonymous',
    domain          TEXT    NOT NULL,
    misconception_class TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    first_seen      REAL    NOT NULL,
    last_seen       REAL    NOT NULL,
    UNIQUE(student_id, domain, misconception_class)
);

CREATE TABLE IF NOT EXISTS teacher_heatmap (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    subdomain       TEXT    NOT NULL,
    misconception_class TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE(subdomain, misconception_class)
);
"""

# ---------------------------------------------------------------------------
# Connection pool (simple global singleton for edge deployment)
# ---------------------------------------------------------------------------

_db_conn: Optional[aiosqlite.Connection] = None
_db_lock = asyncio.Lock()


async def get_db() -> aiosqlite.Connection:
    """Return the global database connection, initializing if needed."""
    global _db_conn
    if _db_conn is None:
        async with _db_lock:
            if _db_conn is None:
                _db_conn = await aiosqlite.connect(DB_PATH, timeout=30)
                _db_conn.row_factory = aiosqlite.Row
                await _db_conn.executescript(DDL)
                await _db_conn.commit()
                logger.info("Database initialized at %s", DB_PATH)
    return _db_conn


async def close_db() -> None:
    """Close the database connection (call on application shutdown)."""
    global _db_conn
    if _db_conn:
        await _db_conn.close()
        _db_conn = None
        logger.info("Database connection closed.")


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------

async def create_job(
    job_id: str,
    filename: str,
    student_id: str = "anonymous",
) -> None:
    db = await get_db()
    now = time.time()
    await db.execute(
        """INSERT INTO jobs (job_id, student_id, filename, status, submitted_at, updated_at)
           VALUES (?, ?, ?, 'queued', ?, ?)""",
        (job_id, student_id, filename, now, now),
    )
    await db.commit()


async def update_job_status(
    job_id: str,
    status: str,
    result_json: Optional[str] = None,
    error_message: Optional[str] = None,
    domain: Optional[str] = None,
) -> None:
    db = await get_db()
    await db.execute(
        """UPDATE jobs SET status=?, updated_at=?, result_json=?, error_message=?, domain=?
           WHERE job_id=?""",
        (status, time.time(), result_json, error_message, domain, job_id),
    )
    await db.commit()


async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    db = await get_db()
    cursor = await db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def list_jobs(student_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    db = await get_db()
    if student_id:
        cursor = await db.execute(
            "SELECT * FROM jobs WHERE student_id=? ORDER BY submitted_at DESC LIMIT ?",
            (student_id, limit),
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM jobs ORDER BY submitted_at DESC LIMIT ?", (limit,)
        )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Feedback logging
# ---------------------------------------------------------------------------

async def log_feedback(
    job_id: str,
    student_id: str,
    domain: str,
    feedback_items: List[Dict[str, Any]],
) -> None:
    """Bulk-insert feedback items generated for a single job."""
    db = await get_db()
    now = time.time()
    rows = [
        (
            job_id,
            student_id,
            domain,
            item.get("type", "socratic"),
            item.get("text", ""),
            item.get("misconception_class", ""),
            item.get("confidence", 0.0),
            now,
        )
        for item in feedback_items
    ]
    await db.executemany(
        """INSERT INTO feedback_log
           (job_id, student_id, domain, feedback_type, feedback_text,
            misconception_class, confidence, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    await db.commit()
    # Update misconception stats
    for item in feedback_items:
        misc_class = item.get("misconception_class", "")
        if misc_class:
            await _upsert_misconception_stat(db, student_id, domain, misc_class, now)
    await db.commit()


async def _upsert_misconception_stat(
    db: aiosqlite.Connection,
    student_id: str,
    domain: str,
    misconception_class: str,
    now: float,
) -> None:
    """Increment occurrence count, or insert new row."""
    await db.execute(
        """INSERT INTO misconception_stats
               (student_id, domain, misconception_class, occurrence_count, first_seen, last_seen)
           VALUES (?, ?, ?, 1, ?, ?)
           ON CONFLICT(student_id, domain, misconception_class) DO UPDATE SET
               occurrence_count = occurrence_count + 1,
               last_seen = excluded.last_seen""",
        (student_id, domain, misconception_class, now, now),
    )


async def get_misconception_stats(
    student_id: Optional[str] = None,
    domain: Optional[str] = None,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    db = await get_db()
    conditions, params = [], []
    if student_id:
        conditions.append("student_id = ?")
        params.append(student_id)
    if domain:
        conditions.append("domain = ?")
        params.append(domain)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(top_n)
    cursor = await db.execute(
        f"""SELECT student_id, domain, misconception_class,
                   SUM(occurrence_count) AS total_occurrences,
                   MIN(first_seen) AS first_seen, MAX(last_seen) AS last_seen
            FROM misconception_stats {where}
            GROUP BY student_id, domain, misconception_class
            ORDER BY total_occurrences DESC LIMIT ?""",
        params,
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]

async def update_teacher_heatmap(subdomain: str, misconception_class: str) -> None:
    """Updates the class-level heatmap with a confidently identified misconception."""
    db = await get_db()
    try:
        await db.execute(
            """
            INSERT INTO teacher_heatmap (subdomain, misconception_class)
            VALUES (?, ?)
            ON CONFLICT(subdomain, misconception_class)
            DO UPDATE SET occurrence_count = occurrence_count + 1
            """,
            (subdomain, misconception_class)
        )
        await db.commit()
    except Exception as e:
        logger.error("Failed to update teacher heatmap: %s", e)
