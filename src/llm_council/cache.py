"""Local SQLite cache for Stage 1 responses to avoid redundant API calls."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger("llm-council")

DEFAULT_CACHE_DIR = Path.home() / ".llm-council"
DEFAULT_CACHE_DB = DEFAULT_CACHE_DIR / "cache.db"


def _cache_key(question: str, model_name: str, model_id: str) -> str:
    """Compute a deterministic cache key from question + model identity."""
    raw = f"{question}\x00{model_name}\x00{model_id}"
    return hashlib.sha256(raw.encode()).hexdigest()


class ResponseCache:
    """SQLite-backed cache for model responses."""

    DEFAULT_TTL = 86400  # 24 hours in seconds

    def __init__(self, db_path: str | Path = DEFAULT_CACHE_DB, ttl: int = DEFAULT_TTL):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    response TEXT NOT NULL,
                    token_usage TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.commit()
            # Startup cleanup of expired entries
            cursor = self._conn.execute(
                "DELETE FROM responses WHERE created_at <= datetime('now', ? || ' seconds')",
                (str(-self.ttl),),
            )
            if cursor.rowcount > 0:
                logger.debug(f"Cleaned up {cursor.rowcount} expired cache entries")
            self._conn.commit()
        return self._conn

    def get(self, question: str, model_name: str, model_id: str) -> tuple[str, dict[str, Any] | None] | None:
        """Look up a cached response. Returns (response_text, token_usage) or None if expired/missing."""
        key = _cache_key(question, model_name, model_id)
        conn = self._get_conn()
        sql = (
            "SELECT response, token_usage FROM responses"
            " WHERE cache_key = ? AND created_at > datetime('now', ? || ' seconds')"
        )
        row = conn.execute(sql, (key, str(-self.ttl))).fetchone()
        if row is None:
            conn.execute("DELETE FROM responses WHERE cache_key = ?", (key,))
            conn.commit()
            return None
        response = row[0]
        token_usage = json.loads(row[1]) if row[1] else None
        logger.debug(f"Cache hit for {model_name}")
        return response, token_usage

    def put(
        self,
        question: str,
        model_name: str,
        model_id: str,
        response: str,
        token_usage: dict[str, Any] | None,
    ) -> None:
        """Store a response in the cache."""
        key = _cache_key(question, model_name, model_id)
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO responses (cache_key, model_name, response, token_usage) VALUES (?, ?, ?, ?)",
            (key, model_name, response, json.dumps(token_usage) if token_usage else None),
        )
        conn.commit()
        logger.debug(f"Cached response for {model_name}")

    def clear(self) -> int:
        """Delete all rows from responses table. Returns count of deleted rows."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM responses")
        count = cursor.rowcount
        conn.commit()
        return count

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def stats(self) -> dict[str, int]:
        """Return cache statistics including expired count."""
        conn = self._get_conn()
        total_row = conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        total = total_row[0] if total_row else 0
        expired_row = conn.execute(
            "SELECT COUNT(*) FROM responses WHERE created_at <= datetime('now', ? || ' seconds')",
            (str(-self.ttl),),
        ).fetchone()
        expired = expired_row[0] if expired_row else 0
        return {"total": total, "expired": expired}
