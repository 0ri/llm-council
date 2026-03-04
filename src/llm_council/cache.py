"""SQLite-backed response cache for Stage 1 model responses.

Exports ``ResponseCache`` which stores and retrieves responses keyed by
(question, model_name, model_id) with configurable TTL and automatic
expiry cleanup. Default database location is ``~/.llm-council/cache.db``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

from .defaults import DEFAULT_CACHE_TTL

logger = logging.getLogger("llm-council")

DEFAULT_CACHE_DIR = Path.home() / ".llm-council"
DEFAULT_CACHE_DB = DEFAULT_CACHE_DIR / "cache.db"


_OUTPUT_AFFECTING_PARAMS = frozenset(
    {
        "temperature",
        "reasoning_effort",
        "reasoning_max_tokens",
        "max_tokens",
        "budget_tokens",
        "web_search",
        "system_message",
    }
)

# Shared schema for the responses table (avoids duplication across connections)
_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS responses (
    cache_key TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    response TEXT NOT NULL,
    token_usage TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"""


def _to_dict(model_config: dict[str, Any] | BaseModel | None) -> dict[str, Any] | None:
    """Convert a Pydantic model to a dict if needed, or return None."""
    if model_config is None:
        return None
    if isinstance(model_config, dict):
        return model_config
    return model_config.model_dump()


def _cache_key(
    question: str,
    model_name: str,
    model_id: str,
    model_config: dict[str, Any] | BaseModel | None = None,
) -> str:
    """Compute a deterministic cache key from question + model identity + params.

    When *model_config* is provided, output-affecting parameters (e.g.
    ``temperature``, ``reasoning_effort``) are included in the hash so that
    config changes invalidate stale cache entries. Accepts both dict and
    Pydantic ModelConfig objects.
    """
    raw = f"{question}\x00{model_name}\x00{model_id}"
    config_dict = _to_dict(model_config)
    if config_dict:
        params = {k: v for k, v in config_dict.items() if k in _OUTPUT_AFFECTING_PARAMS and v is not None}
        if params:
            raw += "\x00" + json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Shared get/put helpers (connection-agnostic)
# ---------------------------------------------------------------------------


def _do_get(
    conn: sqlite3.Connection,
    key: str,
    model_name: str,
    ttl: int,
) -> tuple[str, dict[str, Any] | None] | None:
    """Look up a cached response on *conn*, returning ``None`` on miss."""
    sql = (
        "SELECT response, token_usage FROM responses"
        " WHERE cache_key = ? AND created_at > datetime('now', ? || ' seconds')"
    )
    row = conn.execute(sql, (key, str(-ttl))).fetchone()
    if row is None:
        conn.execute("DELETE FROM responses WHERE cache_key = ?", (key,))
        conn.commit()
        return None
    response = row[0]
    token_usage = json.loads(row[1]) if row[1] else None
    logger.debug(f"Cache hit for {model_name}")
    return response, token_usage


def _do_put(
    conn: sqlite3.Connection,
    key: str,
    model_name: str,
    response: str,
    token_usage: dict[str, Any] | None,
) -> None:
    """Store a response on *conn*."""
    conn.execute(
        "INSERT OR REPLACE INTO responses (cache_key, model_name, response, token_usage) VALUES (?, ?, ?, ?)",
        (key, model_name, response, json.dumps(token_usage) if token_usage else None),
    )
    conn.commit()
    logger.debug(f"Cached response for {model_name}")


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

    DEFAULT_TTL = DEFAULT_CACHE_TTL

    def __init__(self, db_path: str | Path = DEFAULT_CACHE_DB, ttl: int = DEFAULT_TTL):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self._conn: sqlite3.Connection | None = None
        self._local = threading.local()
        self._thread_conns: list[sqlite3.Connection] = []
        self._thread_conns_lock = threading.Lock()

    def _init_conn(self, conn: sqlite3.Connection) -> None:
        """Ensure the schema exists on *conn*."""
        conn.execute(_SCHEMA_SQL)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Return the SQLite connection, creating the table on first use."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._init_conn(self._conn)
            # Startup cleanup of expired entries
            cursor = self._conn.execute(
                "DELETE FROM responses WHERE created_at <= datetime('now', ? || ' seconds')",
                (str(-self.ttl),),
            )
            if cursor.rowcount > 0:
                logger.debug(f"Cleaned up {cursor.rowcount} expired cache entries")
            self._conn.commit()
        return self._conn

    def _get_thread_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection for use in worker threads."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            self._init_conn(conn)
            self._local.conn = conn
            with self._thread_conns_lock:
                self._thread_conns.append(conn)
        return conn

    def _get_sync(
        self,
        question: str,
        model_name: str,
        model_id: str,
        model_config: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None] | None:
        """Synchronous cache lookup. Used by the async ``get`` wrapper.

        Also usable directly for non-async callers (CLI, tests).
        """
        key = _cache_key(question, model_name, model_id, model_config)
        return _do_get(self._get_conn(), key, model_name, self.ttl)

    def _put_sync(
        self,
        question: str,
        model_name: str,
        model_id: str,
        response: str,
        token_usage: dict[str, Any] | None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Synchronous cache store. Used by the async ``put`` wrapper.

        Also usable directly for non-async callers (CLI, tests).
        """
        key = _cache_key(question, model_name, model_id, model_config)
        _do_put(self._get_conn(), key, model_name, response, token_usage)

    def _get_in_thread(
        self,
        question: str,
        model_name: str,
        model_id: str,
        model_config: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None] | None:
        """Thread-safe cache lookup using a thread-local connection."""
        key = _cache_key(question, model_name, model_id, model_config)
        return _do_get(self._get_thread_conn(), key, model_name, self.ttl)

    def _put_in_thread(
        self,
        question: str,
        model_name: str,
        model_id: str,
        response: str,
        token_usage: dict[str, Any] | None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Thread-safe cache store using a thread-local connection."""
        key = _cache_key(question, model_name, model_id, model_config)
        _do_put(self._get_thread_conn(), key, model_name, response, token_usage)

    # Synchronous public API (backwards-compatible for CLI, tests)

    def get(
        self,
        question: str,
        model_name: str,
        model_id: str,
        model_config: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None] | None:
        """Look up a cached response (synchronous).

        For async callers, use ``aget`` instead.
        """
        return self._get_sync(question, model_name, model_id, model_config)

    def put(
        self,
        question: str,
        model_name: str,
        model_id: str,
        response: str,
        token_usage: dict[str, Any] | None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Store a model response in the cache (synchronous).

        For async callers, use ``aput`` instead.
        """
        self._put_sync(question, model_name, model_id, response, token_usage, model_config)

    # Async public API (non-blocking, uses thread-local connections)

    async def aget(
        self,
        question: str,
        model_name: str,
        model_id: str,
        model_config: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None] | None:
        """Look up a cached response without blocking the event loop."""
        return await asyncio.to_thread(self._get_in_thread, question, model_name, model_id, model_config)

    async def aput(
        self,
        question: str,
        model_name: str,
        model_id: str,
        response: str,
        token_usage: dict[str, Any] | None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Store a model response without blocking the event loop."""
        await asyncio.to_thread(
            self._put_in_thread, question, model_name, model_id, response, token_usage, model_config
        )

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
        """Close all SQLite connections (main and thread-local).

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        with self._thread_conns_lock:
            for conn in self._thread_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._thread_conns.clear()

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
