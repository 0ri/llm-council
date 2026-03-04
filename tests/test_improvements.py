"""Tests for performance and provider improvements."""

import asyncio
from unittest.mock import patch

import pytest

from llm_council.cost import CouncilCostTracker
from llm_council.manifest import RunManifest
from llm_council.providers.bedrock import is_retryable_bedrock_error
from llm_council.providers.poe import is_retryable_poe_error
from llm_council.stages import query_models_parallel


class TestManifest:
    """Tests for RunManifest."""

    def test_create_manifest(self, sample_config):
        """Test creating a manifest with all required fields."""
        manifest = RunManifest.create(
            question="What is the capital of France?",
            config=sample_config,
            stage1_count=3,
            valid_ballots=2,
            total_ballots=3,
            elapsed_seconds=45.2,
            estimated_tokens=1500,
        )

        assert manifest.run_id  # Should have a UUID
        assert manifest.timestamp  # Should have ISO timestamp
        assert manifest.question == "What is the capital of France?"
        assert manifest.models == ["Model-A", "Model-B", "Model-C"]
        assert manifest.chairman == "Model-A"
        assert manifest.stage1_results == 3
        assert manifest.stage2_valid_ballots == 2
        assert manifest.stage2_total_ballots == 3
        assert manifest.total_elapsed_seconds == 45.2
        assert manifest.estimated_tokens == 1500
        assert manifest.config_hash  # Should have a hash

    def test_manifest_truncates_long_question(self, sample_config):
        """Test that long questions are truncated to 200 chars."""
        long_question = "x" * 500
        manifest = RunManifest.create(
            question=long_question,
            config=sample_config,
            stage1_count=1,
            valid_ballots=1,
            total_ballots=1,
            elapsed_seconds=1.0,
            estimated_tokens=100,
        )
        assert len(manifest.question) == 200
        assert manifest.question == "x" * 200

    def test_manifest_to_json(self, sample_config):
        """Test JSON serialization of manifest."""
        manifest = RunManifest.create(
            question="Test?",
            config=sample_config,
            stage1_count=1,
            valid_ballots=1,
            total_ballots=1,
            elapsed_seconds=1.0,
            estimated_tokens=100,
        )
        json_str = manifest.to_json()
        assert '"run_id"' in json_str
        assert '"question": "Test?"' in json_str

    def test_manifest_to_comment_block(self, sample_config):
        """Test comment block formatting."""
        manifest = RunManifest.create(
            question="Test?",
            config=sample_config,
            stage1_count=2,
            valid_ballots=1,
            total_ballots=2,
            elapsed_seconds=10.5,
            estimated_tokens=500,
        )
        comment = manifest.to_comment_block()
        assert "<!-- Run Manifest" in comment
        assert "Run ID:" in comment
        assert "Stage 1 Results: 2/3" in comment
        assert "Stage 2 Ballots: 1/2 valid" in comment
        assert "Total Time: 10.5s" in comment
        assert "Est. Tokens: ~500" in comment
        assert "-->" in comment


class TestPerModelCircuitBreakers:
    """Tests for per-model circuit breaker isolation via CouncilContext."""

    def test_per_model_circuit_breakers(self):
        """Test that circuit breakers are per-model, not per-provider."""
        from llm_council.context import CouncilContext

        ctx = CouncilContext()
        cb_poe_gpt = ctx.get_circuit_breaker("poe:GPT-5.3-Codex")
        cb_poe_gemini = ctx.get_circuit_breaker("poe:Gemini-3.1-Pro")
        cb_bedrock_opus = ctx.get_circuit_breaker("bedrock:claude-opus")

        assert cb_poe_gpt is not cb_poe_gemini
        assert cb_poe_gpt is not cb_bedrock_opus

        # Failing one model shouldn't affect others
        for _ in range(3):
            cb_poe_gpt.record_failure()

        assert cb_poe_gpt.is_open
        assert not cb_poe_gemini.is_open
        assert not cb_bedrock_opus.is_open


class TestRetryDiscrimination:
    """Tests for retry discrimination logic."""

    def test_bedrock_retryable_errors(self):
        """Test Bedrock error classification."""
        from botocore.exceptions import ClientError

        # Retryable errors
        throttle_error = ClientError({"Error": {"Code": "ThrottlingException"}}, "invoke_model")
        assert is_retryable_bedrock_error(throttle_error)

        timeout_error = asyncio.TimeoutError()
        assert is_retryable_bedrock_error(timeout_error)

        # Non-retryable errors
        auth_error = ClientError({"Error": {"Code": "AccessDeniedException"}}, "invoke_model")
        assert not is_retryable_bedrock_error(auth_error)

        validation_error = ClientError({"Error": {"Code": "ValidationException"}}, "invoke_model")
        assert not is_retryable_bedrock_error(validation_error)

    def test_poe_retryable_errors(self):
        """Test Poe error classification."""
        # Retryable errors
        timeout_error = asyncio.TimeoutError()
        assert is_retryable_poe_error(timeout_error)

        rate_limit_error = Exception("HTTP 429 Too Many Requests")
        assert is_retryable_poe_error(rate_limit_error)

        server_error = Exception("HTTP 503 Service Unavailable")
        assert is_retryable_poe_error(server_error)

        # Non-retryable errors
        auth_error = Exception("HTTP 401 Unauthorized")
        assert not is_retryable_poe_error(auth_error)

        not_found_error = Exception("Bot TestBot not found")
        assert not is_retryable_poe_error(not_found_error)

        bot_invalid_error = Exception("Bot name is invalid")
        assert not is_retryable_poe_error(bot_invalid_error)


class TestActualTokenCounting:
    """Tests for actual token counting from API responses."""

    def test_cost_tracker_with_actual_tokens(self):
        """Test that actual tokens are preferred over estimates."""
        with patch("llm_council.cost._ENCODER", None):
            tracker = CouncilCostTracker()

            # Record with actual token counts
            tracker.record("Model-A", 1, "x" * 400, "y" * 800, actual_input_tokens=120, actual_output_tokens=210)

            # Record with only estimates
            tracker.record("Model-B", 1, "x" * 400, "y" * 800)

            # Model-A should use actual counts
            model_a_usage = tracker.usages[0]
            assert model_a_usage.input_tokens == 120
            assert model_a_usage.output_tokens == 210

            # Model-B should use estimates (fallback: chars // 4)
            model_b_usage = tracker.usages[1]
            assert model_b_usage.input_tokens == 100  # 400/4
            assert model_b_usage.output_tokens == 200  # 800/4

    def test_cost_tracker_record_with_usage(self):
        """Test record_with_usage helper method."""
        tracker = CouncilCostTracker()

        # With token usage dict
        token_usage = {"input_tokens": 150, "output_tokens": 250}
        tracker.record_with_usage("Model-A", 1, "input", "output", token_usage)

        # Without token usage
        tracker.record_with_usage("Model-B", 1, "x" * 400, "y" * 800, None)

        assert tracker.usages[0].actual_input_tokens == 150
        assert tracker.usages[0].actual_output_tokens == 250
        assert tracker.usages[1].actual_input_tokens is None
        assert tracker.usages[1].actual_output_tokens is None

    def test_cost_summary_with_mixed_tokens(self):
        """Test summary output with mixed actual/estimated tokens."""
        with patch("llm_council.cost._ENCODER", None):
            tracker = CouncilCostTracker()

            # Stage 1: one with actual, one estimated
            tracker.record("Model-A", 1, "q", "a", actual_input_tokens=10, actual_output_tokens=20)
            tracker.record("Model-B", 1, "q" * 40, "a" * 80)  # ~10 in, ~20 out (fallback)

            summary = tracker.summary()
            assert "Model-A: 10 in, 20 out" in summary  # No ~ for actuals
            assert "Model-B: ~10 in, ~20 out" in summary  # ~ for estimates
            assert "(~ indicates estimated tokens, actual counts used where available)" in summary


class TestAggressiveTimeout:
    """Tests for aggressive timeout with graceful degradation."""

    @pytest.mark.asyncio
    async def test_soft_timeout_with_min_responses(self, make_ctx):
        """Test that soft timeout triggers when min responses are received."""
        # Create mock model configs
        model_configs = [
            {"name": "Fast-Model", "provider": "poe", "bot_name": "fast"},
            {"name": "Slow-Model-1", "provider": "poe", "bot_name": "slow1"},
            {"name": "Slow-Model-2", "provider": "poe", "bot_name": "slow2"},
        ]

        # Mock query_model to return quickly for first model, slowly for others
        async def mock_query_model(config, messages, ctx, system_message):
            name = config.name
            if name == "Fast-Model":
                await asyncio.sleep(0.1)
                return "fast response", None
            else:
                # These will be cancelled by soft timeout
                await asyncio.sleep(10)
                return "slow response", None

        with patch("llm_council.stages.execution.query_model", side_effect=mock_query_model):
            ctx = make_ctx()
            results, usages = await query_models_parallel(
                model_configs,
                [{"role": "user", "content": "test"}],
                ctx,
                min_responses=1,
                soft_timeout=0.5,
            )

            # Fast model should have completed
            assert results["Fast-Model"] is not None
            assert results["Fast-Model"] == "fast response"

            # Slow models should be None (cancelled)
            assert results["Slow-Model-1"] is None
            assert results["Slow-Model-2"] is None

    @pytest.mark.asyncio
    async def test_all_models_complete_before_soft_timeout(self, make_ctx):
        """Test that all models complete if they finish before soft timeout."""
        model_configs = [
            {"name": "Model-A", "provider": "poe", "bot_name": "a"},
            {"name": "Model-B", "provider": "poe", "bot_name": "b"},
        ]

        async def mock_query_model(config, messages, ctx, system_message):
            await asyncio.sleep(0.1)
            return f"response from {config.name}", None

        with patch("llm_council.stages.execution.query_model", side_effect=mock_query_model):
            ctx = make_ctx()
            results, usages = await query_models_parallel(
                model_configs,
                [{"role": "user", "content": "test"}],
                ctx,
                soft_timeout=5.0,  # Long timeout
            )

            # All models should complete
            assert results["Model-A"] is not None
            assert results["Model-B"] is not None

    @pytest.mark.asyncio
    async def test_min_responses_default_calculation(self, make_ctx):
        """Test default min_responses calculation."""
        # With 4 models, default should be 3 (len-1)
        model_configs = [{"name": f"Model-{i}", "provider": "poe", "bot_name": f"bot-{i}"} for i in range(4)]

        # Mock only 3 models to respond quickly
        async def mock_query_model(config, messages, ctx, system_message):
            name = config.name
            if name != "Model-3":
                await asyncio.sleep(0.1)
                return f"response from {name}", None
            else:
                await asyncio.sleep(10)
                return "slow", None

        with patch("llm_council.stages.execution.query_model", side_effect=mock_query_model):
            ctx = make_ctx()
            results, usages = await query_models_parallel(
                model_configs,
                [{"role": "user", "content": "test"}],
                ctx,
                soft_timeout=0.5,
                # min_responses not specified, defaults to len(models) = 4
            )

            # All 4 models should complete (default waits for all)
            completed = sum(1 for r in results.values() if r is not None)
            assert completed == 4
