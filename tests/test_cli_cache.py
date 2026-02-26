"""Unit tests for CLI cache-related flags (--cache-ttl validation)."""

import pytest

from llm_council.cli import _build_parser


def test_cache_ttl_negative_value_exits_with_error():
    """Verify --cache-ttl -1 causes argparse error and exit code 2.

    **Validates: Requirements 2.4**
    """
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--cache-ttl", "-1", "question"])
    assert exc_info.value.code == 2


def test_cache_ttl_zero_is_accepted():
    """Verify --cache-ttl 0 is accepted (bypass cache reads).

    **Validates: Requirements 2.3**
    """
    parser = _build_parser()
    args = parser.parse_args(["--cache-ttl", "0", "question"])
    assert args.cache_ttl == 0


def test_cache_ttl_positive_is_accepted():
    """Verify --cache-ttl 3600 is accepted and stored correctly.

    **Validates: Requirements 2.3**
    """
    parser = _build_parser()
    args = parser.parse_args(["--cache-ttl", "3600", "question"])
    assert args.cache_ttl == 3600


import sys
from pathlib import Path

from llm_council.cache import ResponseCache
from llm_council.cli import main


def _patch_cache_db(monkeypatch, db_path):
    """Patch DEFAULT_CACHE_DB so both the exists() check and ResponseCache() use the temp path."""
    monkeypatch.setattr("llm_council.cache.DEFAULT_CACHE_DB", db_path)
    _original_init = ResponseCache.__init__

    def _patched_init(self, db_path_arg=db_path, ttl=ResponseCache.DEFAULT_TTL):
        _original_init(self, db_path=db_path_arg, ttl=ttl)

    monkeypatch.setattr(ResponseCache, "__init__", _patched_init)


class TestClearCachePopulated:
    """Test --clear-cache with a populated cache database.

    **Validates: Requirements 4.2, 4.3, 4.4**
    """

    def test_clear_cache_populated(self, tmp_path, monkeypatch, capsys):
        """Clearing a populated cache prints correct count and exits 0."""
        db_path = tmp_path / "cache.db"
        cache = ResponseCache(db_path)
        cache.put("q1", "m1", "id1", "response1", {"input_tokens": 10})
        cache.put("q2", "m2", "id2", "response2", {"input_tokens": 20})
        cache.close()

        _patch_cache_db(monkeypatch, db_path)
        monkeypatch.setattr(sys, "argv", ["llm-council", "--clear-cache"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Cleared 2" in captured.err


class TestClearCacheNoDB:
    """Test --clear-cache when no database file exists.

    **Validates: Requirements 4.5**
    """

    def test_clear_cache_no_db(self, tmp_path, monkeypatch, capsys):
        """Clearing with no DB file prints 'already empty' and exits 0."""
        nonexistent_db = tmp_path / "nonexistent" / "cache.db"
        _patch_cache_db(monkeypatch, nonexistent_db)
        monkeypatch.setattr(sys, "argv", ["llm-council", "--clear-cache"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "already empty" in captured.err.lower()


class TestClearCacheNoQuestionRequired:
    """Test --clear-cache exits without requiring a question argument.

    **Validates: Requirements 4.4**
    """

    def test_clear_cache_no_question_required(self, tmp_path, monkeypatch, capsys):
        """--clear-cache does not require a question argument."""
        db_path = tmp_path / "cache.db"
        # Create an empty cache so the DB file exists
        cache = ResponseCache(db_path)
        cache.close()

        _patch_cache_db(monkeypatch, db_path)
        monkeypatch.setattr(sys, "argv", ["llm-council", "--clear-cache"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit cleanly (0), not with argparse error (2)
        assert exc_info.value.code == 0


class TestCacheStatsPopulated:
    """Test --cache-stats with a populated cache database.

    **Validates: Requirements 5.2, 5.3, 5.4**
    """

    def test_cache_stats_populated(self, tmp_path, monkeypatch, capsys):
        """Stats output shows correct entry count, expired count, and DB size."""
        db_path = tmp_path / "cache.db"
        cache = ResponseCache(db_path)
        cache.put("q1", "m1", "id1", "response1", {"input_tokens": 10})
        cache.put("q2", "m2", "id2", "response2", {"input_tokens": 20})
        cache.close()

        _patch_cache_db(monkeypatch, db_path)
        monkeypatch.setattr(sys, "argv", ["llm-council", "--cache-stats"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Cache entries: 2" in captured.err
        assert "Expired entries:" in captured.err
        assert "Database size:" in captured.err


class TestCacheStatsNoDB:
    """Test --cache-stats when no database file exists.

    **Validates: Requirements 5.6**
    """

    def test_cache_stats_no_db(self, tmp_path, monkeypatch, capsys):
        """Stats with no DB file prints 'No cache database found' and exits 0."""
        nonexistent_db = tmp_path / "nonexistent" / "cache.db"
        _patch_cache_db(monkeypatch, nonexistent_db)
        monkeypatch.setattr(sys, "argv", ["llm-council", "--cache-stats"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "No cache database found" in captured.err


class TestCacheStatsNoQuestionRequired:
    """Test --cache-stats exits without requiring a question argument.

    **Validates: Requirements 5.5**
    """

    def test_cache_stats_no_question_required(self, tmp_path, monkeypatch, capsys):
        """--cache-stats does not require a question argument."""
        db_path = tmp_path / "cache.db"
        cache = ResponseCache(db_path)
        cache.close()

        _patch_cache_db(monkeypatch, db_path)
        monkeypatch.setattr(sys, "argv", ["llm-council", "--cache-stats"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit cleanly (0), not with argparse error (2)
        assert exc_info.value.code == 0
