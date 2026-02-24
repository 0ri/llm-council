"""Integration tests for the full council pipeline with mocked providers."""

import os
from unittest.mock import MagicMock, patch

import pytest

try:
    from llm_council.council import run_council

    _PATCH_TARGET = "llm_council.stages.get_provider"
except ImportError:
    from council import run_council

    _PATCH_TARGET = "council.get_provider"


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

        async def mock_query(prompt, model_config, timeout):
            name = model_config.get("name", "unknown")
            if call_count["stage"] <= 3:
                call_count["stage"] += 1
                return mock_responses.get(name, "default response")
            elif call_count["stage"] <= 6:
                call_count["stage"] += 1
                return mock_rankings.get(name, '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```')
            else:
                call_count["stage"] += 1
                return "The council has deliberated. Response A was ranked highest."

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            with patch(_PATCH_TARGET) as mock_get_provider:
                mock_provider = MagicMock()
                mock_provider.query = mock_query
                mock_get_provider.return_value = mock_provider
                result = await run_council("What is the meaning of life?", sample_config)

        assert "## LLM Council Response" in result
        assert "Model Rankings" in result
        assert "Synthesized Answer" in result

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_model_failure(self, sample_config):
        call_count = {"n": 0}

        async def mock_query(prompt, model_config, timeout):
            call_count["n"] += 1
            name = model_config.get("name", "unknown")
            if name == "Model-B":
                raise Exception("Model B is down")
            if call_count["n"] <= 6:
                return f"Response from {name}"
            return '```json\n{"ranking": ["Response A", "Response C"]}\n```'

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            with patch(_PATCH_TARGET) as mock_get_provider:
                mock_provider = MagicMock()
                mock_provider.query = mock_query
                mock_get_provider.return_value = mock_provider
                result = await run_council("Test question", sample_config)

        assert "LLM Council Response" in result or "Error" in result
