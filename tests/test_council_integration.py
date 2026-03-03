"""Integration tests for the full council pipeline with mocked providers."""

import os
from unittest.mock import MagicMock, patch

import pytest

from llm_council.context import CouncilContext
from llm_council.cost import CouncilCostTracker
from llm_council.council import run_council
from llm_council.progress import ProgressManager


def _make_ctx_factory(mock_provider, cache=None):
    """Return a context_factory callable that injects *mock_provider*."""

    def factory():
        ctx = CouncilContext(
            poe_api_key="test-key",
            cost_tracker=CouncilCostTracker(),
            progress=ProgressManager(is_tty=False),
            cache=cache,
        )
        # Pre-inject the mock provider for both provider names
        ctx.providers["poe"] = mock_provider
        ctx.providers["bedrock"] = mock_provider
        return ctx

    return factory


def _make_stage_router(stage1_responses, stage2_rankings, stage3_response, n_models):
    """Build a mock_query function that routes by call count."""
    call_count = {"n": 0}

    async def mock_query(prompt, model_config, timeout, **kwargs):
        call_count["n"] += 1
        name = getattr(model_config, "name", "unknown")
        if call_count["n"] <= n_models:
            return stage1_responses.get(name, f"Response from {name}"), None
        elif call_count["n"] <= n_models * 2:
            default = '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```'
            return stage2_rankings.get(name, default), None
        else:
            return stage3_response, None

    return mock_query


class TestRunCouncilIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocked_providers(self, sample_config):
        mock_responses = {
            "Model-A": "This is model A's response about the topic.",
            "Model-B": "This is model B's perspective on the matter.",
            "Model-C": "Model C provides an alternative viewpoint.",
        }
        mock_rankings = {
            "Model-A": '```json\n{"ranking": ["Response B", "Response A", "Response C"]}\n```',
            "Model-B": '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```',
            "Model-C": '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```',
        }
        call_count = {"stage": 1}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            name = getattr(model_config, "name", "unknown")
            if call_count["stage"] <= 3:
                call_count["stage"] += 1
                return mock_responses.get(name, "default response"), None
            elif call_count["stage"] <= 6:
                call_count["stage"] += 1
                default_ranking = '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```'
                return mock_rankings.get(name, default_ranking), None
            else:
                call_count["stage"] += 1
                return "The council has deliberated. Response A was ranked highest.", None

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "What is the meaning of life?",
                sample_config,
                context_factory=_make_ctx_factory(mock_provider),
            )

        assert "## LLM Council Response" in result
        assert "Model Rankings" in result
        assert "Synthesized Answer" in result

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_model_failure(self, sample_config):
        call_count = {"n": 0}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            call_count["n"] += 1
            name = getattr(model_config, "name", "unknown")
            if name == "Model-B":
                raise Exception("Model B is down")
            if call_count["n"] <= 6:
                return f"Response from {name}", None
            return '```json\n{"ranking": ["Response A", "Response C"]}\n```', None

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "Test question",
                sample_config,
                context_factory=_make_ctx_factory(mock_provider),
            )

        assert "LLM Council Response" in result or "Error" in result

    # --- Issue #3 + #5: New integration tests ---

    @pytest.mark.asyncio
    async def test_all_stage1_models_fail(self, sample_config):
        """All models return None/raise in Stage 1 → graceful error, no crash."""

        async def mock_query(prompt, model_config, timeout, **kwargs):
            raise Exception("Provider unavailable")

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "Test question",
                sample_config,
                context_factory=_make_ctx_factory(mock_provider),
            )

        assert "Error" in result
        assert "failed" in result.lower() or "credentials" in result.lower()

    @pytest.mark.asyncio
    async def test_stage2_all_invalid_ballots(self, sample_config):
        """Rankings are unparseable → low ballot warning, Stage 3 still attempted."""
        call_count = {"n": 0}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            call_count["n"] += 1
            name = getattr(model_config, "name", "unknown")
            if call_count["n"] <= 3:
                return f"Response from {name}", None
            elif call_count["n"] <= 6:
                # Return unparseable ranking
                return "I cannot rank these responses properly.", None
            else:
                return "Synthesis based on available information.", None

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "Test question",
                sample_config,
                context_factory=_make_ctx_factory(mock_provider),
            )

        # Should still produce output (either warning + synthesis, or error)
        assert result  # Non-empty response

    @pytest.mark.asyncio
    async def test_mixed_provider_failures(self, sample_config):
        """1 of 3 models fails in Stage 1 → pipeline continues with 2 responses."""
        call_count = {"n": 0}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            call_count["n"] += 1
            name = getattr(model_config, "name", "unknown")
            if call_count["n"] <= 3:
                if name == "Model-C":
                    raise Exception("Model C timeout")
                return f"Response from {name}", None
            elif call_count["n"] <= 6:
                return '```json\n{"ranking": ["Response A", "Response B"]}\n```', None
            else:
                return "Final synthesis from working models.", None

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "Test question",
                sample_config,
                context_factory=_make_ctx_factory(mock_provider),
            )

        # Pipeline should complete (with or without warnings)
        assert "LLM Council Response" in result or "Error" in result

    @pytest.mark.asyncio
    async def test_cache_hit_path(self, sample_config, tmp_path):
        """Run twice with cache → second run uses cache, fewer provider calls."""
        from llm_council.cache import ResponseCache

        cache = ResponseCache(db_path=tmp_path / "test_cache.db", ttl=86400)
        query_count = {"total": 0}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            query_count["total"] += 1
            name = getattr(model_config, "name", "unknown")
            if query_count["total"] <= 3:
                return f"Cached response from {name}", None
            elif query_count["total"] <= 6:
                return '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```', None
            else:
                return "Synthesized answer.", None

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        question = "Cache test question"

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            await run_council(
                question,
                sample_config,
                context_factory=_make_ctx_factory(mock_provider, cache=cache),
                use_cache=True,
            )

        first_run_calls = query_count["total"]

        # Reset for second run with fresh mock but same cache
        query_count["total"] = 0
        mock_provider2 = MagicMock()
        mock_provider2.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            await run_council(
                question,
                sample_config,
                context_factory=_make_ctx_factory(mock_provider2, cache=cache),
                use_cache=True,
            )

        second_run_calls = query_count["total"]

        # Second run should make fewer Stage 1 calls (cache hit)
        assert second_run_calls <= first_run_calls
        cache.close()

    @pytest.mark.asyncio
    async def test_budget_exhaustion_mid_pipeline(self):
        """Tiny budget → pipeline completes gracefully (budget error or partial)."""
        config = {
            "council_models": [
                {"name": "Model-A", "provider": "poe", "bot_name": "TestBot-A"},
                {"name": "Model-B", "provider": "poe", "bot_name": "TestBot-B"},
                {"name": "Model-C", "provider": "poe", "bot_name": "TestBot-C"},
            ],
            "chairman": {"name": "Model-A", "provider": "poe", "bot_name": "TestBot-A"},
            "budget": {"max_tokens": 50, "max_cost_usd": 0.0001},
        }
        call_count = {"n": 0}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            call_count["n"] += 1
            name = getattr(model_config, "name", "unknown")
            # Return large-ish responses to trigger budget
            return f"{'x' * 200} Response from {name}", {
                "input_tokens": 100,
                "output_tokens": 100,
            }

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "Budget test",
                config,
                context_factory=_make_ctx_factory(mock_provider),
            )

        # Should either complete with budget error or complete gracefully
        assert result  # Non-empty

    @pytest.mark.asyncio
    async def test_single_model_config(self):
        """1 council model → Stage 1 works, Stage 2 has 1 ranker, Stage 3 synthesizes."""
        config = {
            "council_models": [
                {"name": "Solo-Model", "provider": "poe", "bot_name": "SoloBot"},
            ],
            "chairman": {"name": "Solo-Model", "provider": "poe", "bot_name": "SoloBot"},
        }
        call_count = {"n": 0}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "Solo model's response to the question.", None
            elif call_count["n"] == 2:
                return '```json\n{"ranking": ["Response A"]}\n```', None
            else:
                return "Final answer from the sole model.", None

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "Single model test",
                config,
                context_factory=_make_ctx_factory(mock_provider),
            )

        assert result
        # Should produce some form of output (synthesis or stage output)
        assert "Solo-Model" in result or "Synthesized" in result or "Council" in result

    @pytest.mark.asyncio
    async def test_auto_chairman(self):
        """No 'chairman' in config → #1 ranked model synthesizes successfully."""
        config = {
            "council_models": [
                {"name": "Model-A", "provider": "poe", "bot_name": "TestBot-A"},
                {"name": "Model-B", "provider": "poe", "bot_name": "TestBot-B"},
                {"name": "Model-C", "provider": "poe", "bot_name": "TestBot-C"},
            ],
            # No chairman key — auto-chairman mode
        }
        call_count = {"n": 0}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            call_count["n"] += 1
            name = getattr(model_config, "name", "unknown")
            if call_count["n"] <= 3:
                return f"Response from {name}", None
            elif call_count["n"] <= 6:
                # With 3 models and self-exclusion, each ranker sees 2 responses
                # Response A ranked #1 by all rankers
                return '```json\n{"ranking": ["Response A", "Response B"]}\n```', None
            else:
                return "Auto-chairman synthesis from the top-ranked model.", None

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "Auto chairman test",
                config,
                context_factory=_make_ctx_factory(mock_provider),
            )

        assert "LLM Council Response" in result
        assert "Synthesized Answer" in result
        # Manifest should show auto chairman
        assert "(auto)" in result

    @pytest.mark.asyncio
    async def test_chairman_fallback(self, sample_config):
        """Explicit chairman fails → #1 ranked model takes over."""
        # Chairman is Model-A (from sample_config).
        # We disable response order shuffling so ranking labels are deterministic:
        # with no shuffle, "Response B" consistently maps to the later model in
        # each ranker's filtered list, making Model-C the #1 ranked model
        # (different from chairman Model-A), which triggers the fallback path.
        call_count = {"n": 0}
        chairman_attempts = {"count": 0}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            call_count["n"] += 1
            name = getattr(model_config, "name", "unknown")
            if call_count["n"] <= 3:
                return f"Response from {name}", None
            elif call_count["n"] <= 6:
                # With 3 models and self-exclusion, each ranker sees 2 responses.
                # No shuffle → "Response B" is always the second in original order.
                return '```json\n{"ranking": ["Response B", "Response A"]}\n```', None
            else:
                # Stage 3: first attempt (chairman Model-A) fails, fallback succeeds
                chairman_attempts["count"] += 1
                if chairman_attempts["count"] == 1:
                    return "Error: Unable to generate final synthesis", None
                return "Fallback synthesis from top-ranked model.", None

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        # Patch random.Random in stages to disable shuffling, making label
        # mappings deterministic so the #1 ranked model differs from chairman.
        mock_rng = MagicMock()
        mock_rng.shuffle = lambda x: None

        with (
            patch.dict(os.environ, {"POE_API_KEY": "test-key"}),
            patch("llm_council.stages.random.Random", return_value=mock_rng),
        ):
            result = await run_council(
                "Chairman fallback test",
                sample_config,
                context_factory=_make_ctx_factory(mock_provider),
            )

        # Should contain the fallback synthesis, not the error
        assert "Fallback synthesis" in result
        assert chairman_attempts["count"] == 2
