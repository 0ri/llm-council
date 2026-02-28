"""Integration tests for the full council pipeline with mocked providers."""

import os
from unittest.mock import MagicMock, patch

import pytest

from llm_council.context import CouncilContext
from llm_council.cost import CouncilCostTracker
from llm_council.council import run_council
from llm_council.progress import ProgressManager


def _make_ctx_factory(mock_provider):
    """Return a context_factory callable that injects *mock_provider*."""

    def factory():
        ctx = CouncilContext(
            poe_api_key="test-key",
            cost_tracker=CouncilCostTracker(),
            progress=ProgressManager(is_tty=False),
        )
        # Pre-inject the mock provider for both provider names
        ctx.providers["poe"] = mock_provider
        ctx.providers["bedrock"] = mock_provider
        return ctx

    return factory


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
