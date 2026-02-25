"""Tests for the local response cache."""

from __future__ import annotations

import tempfile
from pathlib import Path

from llm_council.cache import ResponseCache


class TestResponseCache:
    """Test the ResponseCache."""

    def test_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir) / "test.db")
            assert cache.get("question", "model", "model-id") is None
            cache.close()

    def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir) / "test.db")
            cache.put("What is 2+2?", "GPT-5", "openai/gpt-5", "Four", {"input_tokens": 10, "output_tokens": 5})

            result = cache.get("What is 2+2?", "GPT-5", "openai/gpt-5")
            assert result is not None
            text, usage = result
            assert text == "Four"
            assert usage == {"input_tokens": 10, "output_tokens": 5}
            cache.close()

    def test_different_question_misses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir) / "test.db")
            cache.put("What is 2+2?", "GPT-5", "openai/gpt-5", "Four", None)

            assert cache.get("What is 3+3?", "GPT-5", "openai/gpt-5") is None
            cache.close()

    def test_different_model_misses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir) / "test.db")
            cache.put("What is 2+2?", "GPT-5", "openai/gpt-5", "Four", None)

            assert cache.get("What is 2+2?", "Claude", "anthropic/claude") is None
            cache.close()

    def test_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir) / "test.db")
            cache.put("q", "m", "id", "old answer", None)
            cache.put("q", "m", "id", "new answer", {"input_tokens": 1})

            text, usage = cache.get("q", "m", "id")
            assert text == "new answer"
            assert usage == {"input_tokens": 1}
            cache.close()

    def test_none_token_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir) / "test.db")
            cache.put("q", "m", "id", "answer", None)

            text, usage = cache.get("q", "m", "id")
            assert text == "answer"
            assert usage is None
            cache.close()

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir) / "test.db")
            assert cache.stats["entries"] == 0

            cache.put("q1", "m1", "id1", "a1", None)
            cache.put("q2", "m2", "id2", "a2", None)
            assert cache.stats["entries"] == 2
            cache.close()

    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            cache1 = ResponseCache(db_path)
            cache1.put("q", "m", "id", "answer", {"input_tokens": 42})
            cache1.close()

            cache2 = ResponseCache(db_path)
            result = cache2.get("q", "m", "id")
            assert result is not None
            assert result[0] == "answer"
            cache2.close()
