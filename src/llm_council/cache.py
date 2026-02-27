"""SQLite-backed response cache for Stage 1 model responses.

Exports ``ResponseCache`` which stores and retrieves responses keyed by
(question, model_name, model_id) with configurable TTL and automatic
expiry cleanup. Default database location is ``~/.llm-council/cache.db``.
"""

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
    """Manage an SQLite-backed cache for Stage 1 model responses.

    Stores and retrieves LLM responses keyed by a SHA-256 hash of
    (question, model_name, model_id). Expired entries are cleaned up
    automatically on first connection. The default database location
    is ``~/.llm-council/cache.db``.

    Args:
        db_path: Path to the SQLite database file. Parent directories
            are created automatically if they do not exist.
        ttl: Time-to-live in seconds for cached entries. Entries older
            than this are treated as expired. Defaults to 86 400 (24 h).

    Raises:
        sqlite3.OperationalError: If the database file cannot be opened
            or created (e.g., permission denied).
    """

    DEFAULT_TTL = 86400  # 24 hours in seconds

    def __init__(self, db_path: str | Path = DEFAULT_CACHE_DB, ttl: int = DEFAULT_TTL):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        """Return the SQLite connection, creating the table on first use."""
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
        """Look up a cached response for a given question and model.

        Args:
            question: The user question that was sent to the model.
            model_name: Display name of the model (e.g. ``"claude-sonnet"``).
            model_id: Provider-specific model identifier used to
                disambiguate models that share a display name.

        Returns:
            A ``(response_text, token_usage)`` tuple on cache hit, where
            *token_usage* is a dict of token counts or ``None`` if usage
            data was not recorded. Returns ``None`` on cache miss or if
            the entry has expired.
        """
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
        """Store a model response in the cache.

        Overwrites any existing entry for the same cache key (upsert).

        Args:
            question: The user question that was sent to the model.
            model_name: Display name of the model.
            model_id: Provider-specific model identifier.
            response: The full text response from the model.
            token_usage: Optional dict of token-usage metadata
                (e.g. ``{"input_tokens": 120, "output_tokens": 350}``).
        """
        key = _cache_key(question, model_name, model_id)
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO responses (cache_key, model_name, response, token_usage) VALUES (?, ?, ?, ?)",
            (key, model_name, response, json.dumps(token_usage) if token_usage else None),
        )
        conn.commit()
        logger.debug(f"Cached response for {model_name}")

    def clear(self) -> int:
        """Delete all cached responses.

        Returns:
            The number of rows deleted.
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM responses")
        count = cursor.rowcount
        conn.commit()
        return count

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def stats(self) -> dict[str, int]:
        """Return cache statistics.

        Returns:
            A dict with ``"total"`` (all rows) and ``"expired"``
            (rows older than the configured TTL) counts.
        """
        conn = self._get_conn()
        total_row = conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        total = total_row[0] if total_row else 0
        expired_row = conn.execute(
            "SELECT COUNT(*) FROM responses WHERE created_at <= datetime('now', ? || ' seconds')",
            (str(-self.ttl),),
        ).fetchone()
        expired = expired_row[0] if expired_row else 0
        return {"total": total, "expired": expired}
