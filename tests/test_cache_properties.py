"""Property-based tests for ResponseCache TTL behavior."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from llm_council.cache import ResponseCache, _cache_key
from llm_council.cli import _format_file_size

# Strategies for generating cache entry data
question_st = st.text(min_size=1, max_size=200)
model_name_st = st.text(min_size=1, max_size=100)
model_id_st = st.text(min_size=1, max_size=100)
response_st = st.text(min_size=1, max_size=500)
token_usage_st = st.fixed_dictionaries(
    {
        "input_tokens": st.integers(min_value=0, max_value=10000),
        "output_tokens": st.integers(min_value=0, max_value=10000),
    }
)
# TTL between 1 and 3600 seconds
ttl_st = st.integers(min_value=1, max_value=3600)
# How many extra seconds past the TTL the entry should be aged
extra_age_st = st.integers(min_value=1, max_value=7200)


# Feature: cache-ttl-invalidation, Property 1: Expired entries are missed and deleted
@settings(max_examples=100)
@given(
    question=question_st,
    model_name=model_name_st,
    model_id=model_id_st,
    response=response_st,
    token_usage=token_usage_st,
    ttl=ttl_st,
    extra_age=extra_age_st,
)
def test_expired_entries_are_missed_and_deleted(
    tmp_path_factory,
    question: str,
    model_name: str,
    model_id: str,
    response: str,
    token_usage: dict,
    ttl: int,
    extra_age: int,
):
    """For any cache entry whose created_at is older than the TTL,
    get() should return None AND the entry should no longer exist in the database.

    **Validates: Requirements 1.2, 1.3**
    """
    db_path = tmp_path_factory.mktemp("cache") / "test.db"
    cache = ResponseCache(db_path, ttl=ttl)

    # Insert the entry normally
    cache.put(question, model_name, model_id, response, token_usage)

    # Manually backdate created_at so the entry is expired
    age_seconds = ttl + extra_age
    conn = cache._get_conn()
    key = _cache_key(question, model_name, model_id)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "UPDATE responses SET created_at = ? WHERE cache_key = ?",
        (old_timestamp, key),
    )
    conn.commit()

    # get() should return None for the expired entry
    result = cache.get(question, model_name, model_id)
    assert result is None, f"Expected None for expired entry, got {result}"

    # The row should have been deleted from the database
    row = conn.execute("SELECT COUNT(*) FROM responses WHERE cache_key = ?", (key,)).fetchone()
    assert row[0] == 0, f"Expected expired row to be deleted, but found {row[0]} rows"

    cache.close()


# Feature: cache-ttl-invalidation, Property 2: Fresh entry round-trip
@settings(max_examples=100)
@given(
    question=question_st,
    model_name=model_name_st,
    model_id=model_id_st,
    response=response_st,
    token_usage=token_usage_st,
)
def test_fresh_entry_round_trip(
    tmp_path_factory,
    question: str,
    model_name: str,
    model_id: str,
    response: str,
    token_usage: dict,
):
    """For any question, model_name, model_id, response text, and token_usage dict,
    if put() is called and then get() is called within the TTL window, the returned
    response and token_usage should be equal to the values that were stored.

    **Validates: Requirements 1.4**
    """
    db_path = tmp_path_factory.mktemp("cache") / "test.db"
    cache = ResponseCache(db_path, ttl=86400)

    # Store the entry
    cache.put(question, model_name, model_id, response, token_usage)

    # Immediately retrieve — well within the 86400s TTL
    result = cache.get(question, model_name, model_id)

    assert result is not None, "Expected cache hit for freshly stored entry"
    returned_response, returned_token_usage = result
    assert returned_response == response, f"Response mismatch: expected {response!r}, got {returned_response!r}"
    assert returned_token_usage == token_usage, (
        f"Token usage mismatch: expected {token_usage!r}, got {returned_token_usage!r}"
    )

    cache.close()


# Strategy for cache entries with an age (in seconds)
# We use two separate strategies: one for clearly expired entries and one for clearly fresh entries
# to avoid boundary timing issues between Python's datetime and SQLite's datetime('now')
fresh_age_st = st.integers(min_value=0, max_value=0)  # age=0 means just created, always fresh
expired_extra_age_st = st.integers(min_value=2, max_value=7200)  # extra seconds beyond TTL

entry_with_freshness_st = st.tuples(
    question_st,  # question
    model_name_st,  # model_name
    model_id_st,  # model_id
    response_st,  # response
    token_usage_st,  # token_usage
    st.booleans(),  # is_expired
)


# Feature: cache-ttl-invalidation, Property 7: Startup cleanup deletes all expired entries
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    entries=st.lists(
        entry_with_freshness_st,
        min_size=1,
        max_size=20,
        unique_by=lambda e: _cache_key(e[0], e[1], e[2]),
    ),
    ttl=ttl_st,
    expired_extra=expired_extra_age_st,
)
def test_startup_cleanup_deletes_expired_entries(
    tmp_path_factory,
    caplog,
    entries: list[tuple],
    ttl: int,
    expired_extra: int,
):
    """For any set of cache entries with various ages and a given TTL, when a new
    ResponseCache connection is established, all entries older than the TTL should be
    deleted and only fresh entries should remain. When at least one entry is deleted,
    a DEBUG log message containing the deletion count should be emitted.

    **Validates: Requirements 6.1, 6.2**
    """
    import json
    import logging

    db_path = tmp_path_factory.mktemp("cache") / "test.db"

    # Pre-populate the database manually (no ResponseCache yet)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            cache_key TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            response TEXT NOT NULL,
            token_usage TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    expired_keys = set()
    fresh_keys = set()

    for question, model_name, model_id, response, token_usage, is_expired in entries:
        key = _cache_key(question, model_name, model_id)
        if is_expired:
            # Clearly expired: age = ttl + expired_extra (at least 2s past TTL)
            age = ttl + expired_extra
            expired_keys.add(key)
        else:
            # Clearly fresh: use negative age (future timestamp) to avoid
            # boundary timing issues between Python and SQLite clocks
            age = -5
            fresh_keys.add(key)

        created_at = (datetime.now(timezone.utc) - timedelta(seconds=age)).strftime("%Y-%m-%d %H:%M:%S")
        sql = (
            "INSERT OR REPLACE INTO responses"
            " (cache_key, model_name, response, token_usage, created_at) VALUES (?, ?, ?, ?, ?)"
        )
        conn.execute(
            sql,
            (key, model_name, response, json.dumps(token_usage), created_at),
        )

    conn.commit()
    conn.close()

    # Now create a new ResponseCache — _get_conn() should clean up expired entries
    caplog.set_level(logging.DEBUG, logger="llm-council")
    caplog.clear()

    cache = ResponseCache(db_path, ttl=ttl)
    # Trigger _get_conn() by accessing stats
    _ = cache.stats

    # Verify only fresh entries remain
    db_conn = sqlite3.connect(str(db_path))
    remaining_keys = {row[0] for row in db_conn.execute("SELECT cache_key FROM responses").fetchall()}
    db_conn.close()

    assert remaining_keys == fresh_keys, (
        f"Expected only fresh keys {fresh_keys}, but found {remaining_keys}. "
        f"Expired keys that should be gone: {expired_keys & remaining_keys}"
    )

    # Verify DEBUG log message when entries were removed
    if expired_keys:
        cleanup_messages = [
            r.message for r in caplog.records if "expired" in r.message.lower() and str(len(expired_keys)) in r.message
        ]
        assert len(cleanup_messages) > 0, (
            f"Expected DEBUG log about cleaning up {len(expired_keys)} expired entries, "
            f"but found no matching log message. Log records: {[r.message for r in caplog.records]}"
        )

    cache.close()


# Strategy for generating a list of unique cache entries (unique by cache key)
cache_entry_st = st.tuples(
    question_st,
    model_name_st,
    model_id_st,
    response_st,
    token_usage_st,
)


# Feature: cache-ttl-invalidation, Property 4: Clear removes all entries and reports correct count
@settings(max_examples=100)
@given(
    entries=st.lists(
        cache_entry_st,
        min_size=0,
        max_size=20,
        unique_by=lambda e: _cache_key(e[0], e[1], e[2]),
    ),
)
def test_clear_removes_all_entries(
    tmp_path_factory,
    entries: list[tuple],
):
    """For any set of N cache entries (N >= 0), calling clear() should delete all
    rows from the table, return N, and leave the table empty.

    **Validates: Requirements 4.2, 4.3**
    """
    db_path = tmp_path_factory.mktemp("cache") / "test.db"
    # Use a large TTL so no entries expire during the test
    cache = ResponseCache(db_path, ttl=86400)

    # Insert all entries
    for question, model_name, model_id, response, token_usage in entries:
        cache.put(question, model_name, model_id, response, token_usage)

    n = len(entries)

    # Clear and verify returned count
    cleared = cache.clear()
    assert cleared == n, f"clear() returned {cleared}, expected {n}"

    # Verify the table is empty
    assert cache.stats["total"] == 0, f"Expected 0 entries after clear(), got {cache.stats['total']}"

    cache.close()


# Feature: cache-ttl-invalidation, Property 5: Stats report accurate counts
# Use a TTL with a safe minimum to avoid timing flakiness between Python and SQLite
stats_ttl_st = st.integers(min_value=60, max_value=3600)


@settings(max_examples=100)
@given(
    entries=st.lists(
        entry_with_freshness_st,
        min_size=0,
        max_size=20,
        unique_by=lambda e: _cache_key(e[0], e[1], e[2]),
    ),
    ttl=stats_ttl_st,
    expired_extra=expired_extra_age_st,
)
def test_stats_report_accurate_counts(
    tmp_path_factory,
    entries: list[tuple],
    ttl: int,
    expired_extra: int,
):
    """For any set of cache entries with various created_at timestamps and a given TTL,
    stats should report a total equal to the number of entries in the table and an
    expired count equal to the number of entries whose created_at is older than the TTL.

    **Validates: Requirements 5.2, 5.4**
    """

    db_path = tmp_path_factory.mktemp("cache") / "test.db"
    # Use a large TTL for initial cache creation so startup cleanup is a no-op
    cache = ResponseCache(db_path, ttl=86400)

    # Insert all entries via put()
    for question, model_name, model_id, response, token_usage, _is_expired in entries:
        cache.put(question, model_name, model_id, response, token_usage)

    # Set timestamps explicitly for ALL entries via raw SQL to avoid timing issues
    conn = cache._get_conn()
    expected_expired = 0
    for question, model_name, model_id, _response, _token_usage, is_expired in entries:
        key = _cache_key(question, model_name, model_id)
        if is_expired:
            # Clearly expired: age well beyond TTL
            age = ttl + expired_extra
            expected_expired += 1
        else:
            # Clearly fresh: set created_at to now via SQL so it's in sync with
            # SQLite's datetime('now') used by the stats query
            age = 0
        ts = (datetime.now(timezone.utc) - timedelta(seconds=age)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "UPDATE responses SET created_at = ? WHERE cache_key = ?",
            (ts, key),
        )
    conn.commit()

    # Switch to the actual TTL for stats (reuse existing connection to skip cleanup)
    cache.ttl = ttl

    result = cache.stats
    expected_total = len(entries)

    assert result["total"] == expected_total, f"Expected total={expected_total}, got {result['total']}"
    assert result["expired"] == expected_expired, f"Expected expired={expected_expired}, got {result['expired']}"

    cache.close()


import argparse  # noqa: E402

from llm_council.cli import _resolve_ttl  # noqa: E402


# Feature: cache-ttl-invalidation, Property 3: TTL resolution precedence
@settings(max_examples=100)
@given(
    cli_ttl=st.one_of(st.none(), st.integers(min_value=0, max_value=100000)),
    config_ttl=st.one_of(st.none(), st.integers(min_value=0, max_value=100000)),
)
def test_ttl_resolution_precedence(cli_ttl, config_ttl):
    """For any pair of integer TTL values (cli_ttl, config_ttl), when both are provided,
    the resolved TTL should equal cli_ttl. When only config_ttl is provided, the resolved
    TTL should equal config_ttl. When neither is provided, the resolved TTL should equal 86400.

    **Validates: Requirements 3.2, 3.3, 1.5**
    """
    args = argparse.Namespace(cache_ttl=cli_ttl)
    config = {}
    if config_ttl is not None:
        config["cache_ttl"] = config_ttl

    result = _resolve_ttl(args, config)

    if cli_ttl is not None:
        assert result == cli_ttl, f"CLI TTL should take precedence: expected {cli_ttl}, got {result}"
    elif config_ttl is not None:
        assert result == config_ttl, f"Config TTL should be used when CLI is None: expected {config_ttl}, got {result}"
    else:
        assert result == ResponseCache.DEFAULT_TTL, (
            f"Default TTL should be used when both are None: expected {ResponseCache.DEFAULT_TTL}, got {result}"
        )


# Feature: cache-ttl-invalidation, Property 6: File size formatting
@settings(max_examples=100)
@given(
    size_bytes=st.integers(min_value=0, max_value=10_000_000_000),
)
def test_file_size_formatting(size_bytes):
    """For any non-negative integer byte count, the human-readable formatting function
    should produce a string containing the correct numeric value and an appropriate unit
    suffix (bytes, KB, or MB), and the numeric value should round-trip back to
    approximately the original byte count.

    **Validates: Requirements 5.3**
    """
    result = _format_file_size(size_bytes)

    if size_bytes < 1024:
        # Should use "bytes" suffix with exact integer value
        assert result.endswith(" bytes"), f"Expected 'bytes' suffix for {size_bytes}, got '{result}'"
        numeric_str = result.replace(" bytes", "")
        numeric_value = int(numeric_str)
        assert numeric_value == size_bytes, f"Byte value should be exact: expected {size_bytes}, got {numeric_value}"
    elif size_bytes < 1024 * 1024:
        # Should use "KB" suffix
        assert result.endswith(" KB"), f"Expected 'KB' suffix for {size_bytes}, got '{result}'"
        numeric_str = result.replace(" KB", "")
        numeric_value = float(numeric_str)
        # Round-trip: numeric_value * 1024 should be within 0.1 KB (≈102 bytes) of original
        round_tripped = numeric_value * 1024
        assert abs(round_tripped - size_bytes) < 1024 * 0.1, (
            f"KB round-trip failed: {size_bytes} -> '{result}' -> {round_tripped}"
        )
    else:
        # Should use "MB" suffix
        assert result.endswith(" MB"), f"Expected 'MB' suffix for {size_bytes}, got '{result}'"
        numeric_str = result.replace(" MB", "")
        numeric_value = float(numeric_str)
        # Round-trip: numeric_value * 1024 * 1024 should be within 0.1 MB (≈104857 bytes) of original
        round_tripped = numeric_value * 1024 * 1024
        assert abs(round_tripped - size_bytes) < 1024 * 1024 * 0.1, (
            f"MB round-trip failed: {size_bytes} -> '{result}' -> {round_tripped}"
        )
