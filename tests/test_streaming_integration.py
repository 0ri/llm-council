"""Integration tests for end-to-end streaming in the council pipeline.

Tests cover:
- Full council run with stream=True using mock providers
- Full council run with stream=False produces identical output
- Stage 3 fallback from streaming to query on mid-stream error

Requirements: 9.5, 10.1, 10.2
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from llm_council.council import run_council
from llm_council.providers import StreamResult
from llm_council.run_options import RunOptions

# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


class _MockStreamingProvider:
    """A mock provider that supports both query() and astream().

    Uses a call counter to return stage-appropriate responses:
    - Calls 1-3: Stage 1 model responses
    - Calls 4-6: Stage 2 ranking JSON
    - Call 7+:   Stage 3 chairman synthesis
    """

    def __init__(self):
        self._call_count = 0
        self._model_responses = {
            "Model-A": "Model A thinks the answer involves philosophy.",
            "Model-B": "Model B believes it relates to purpose.",
            "Model-C": "Model C suggests it is about happiness.",
        }
        self._rankings = {
            "Model-A": '```json\n{"ranking": ["Response B", "Response A", "Response C"]}\n```',
            "Model-B": '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```',
            "Model-C": '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```',
        }
        self._synthesis = (
            "The council has deliberated. The meaning of life involves philosophy, purpose, and happiness."
        )

    async def query(self, model_config: dict, timeout: int, **kwargs) -> tuple[str, dict | None]:
        self._call_count += 1
        name = getattr(model_config, "name", "unknown")
        if self._call_count <= 3:
            return self._model_responses.get(name, "default response"), None
        elif self._call_count <= 6:
            default_ranking = '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```'
            return self._rankings.get(name, default_ranking), None
        else:
            return self._synthesis, None

    def astream(self, model_config: dict, timeout: int, **kwargs) -> StreamResult:
        """Stream the chairman synthesis as multiple chunks."""
        synthesis = self._synthesis
        self._call_count += 1

        async def _generate():
            # Split synthesis into word-level chunks
            words = synthesis.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == len(words) - 1 else word + " "
                yield chunk

        return StreamResult(_generate())


class _MockFailingStreamProvider:
    """A mock provider whose astream fails mid-stream but query works.

    Used to test Stage 3 fallback behavior.
    """

    def __init__(self):
        self._call_count = 0
        self._model_responses = {
            "Model-A": "Model A response for testing.",
            "Model-B": "Model B response for testing.",
            "Model-C": "Model C response for testing.",
        }
        self._rankings = {
            "Model-A": '```json\n{"ranking": ["Response B", "Response A", "Response C"]}\n```',
            "Model-B": '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```',
            "Model-C": '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```',
        }
        self._synthesis = "Fallback synthesis after streaming error."

    async def query(self, model_config: dict, timeout: int, **kwargs) -> tuple[str, dict | None]:
        self._call_count += 1
        name = getattr(model_config, "name", "unknown")
        if self._call_count <= 3:
            return self._model_responses.get(name, "default response"), None
        elif self._call_count <= 6:
            default_ranking = '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```'
            return self._rankings.get(name, default_ranking), None
        else:
            return self._synthesis, None

    def astream(self, model_config: dict, timeout: int, **kwargs) -> StreamResult:
        """Stream that fails mid-way through."""
        self._call_count += 1

        async def _generate():
            yield "Partial "
            yield "output "
            raise RuntimeError("connection lost mid-stream")

        return StreamResult(_generate())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamingIntegration:
    """End-to-end integration tests for streaming council runs."""

    @pytest.mark.asyncio
    async def test_full_council_run_with_stream_true(self, sample_config, make_ctx_factory):
        """A full council run with stream=True completes and produces valid output."""
        provider = _MockStreamingProvider()
        chunks_received: list[str] = []

        async def on_chunk(chunk: str) -> None:
            chunks_received.append(chunk)

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "What is the meaning of life?",
                sample_config,
                options=RunOptions(
                    context_factory=make_ctx_factory(provider),
                    stream=True,
                    on_chunk=on_chunk,
                ),
            )

        # The result should contain standard council output sections
        assert "## LLM Council Response" in result
        assert "Synthesized Answer" in result

        # The on_chunk callback should have received chunks
        assert len(chunks_received) > 0
        # Concatenated chunks should form the synthesis text
        concatenated = "".join(chunks_received)
        assert concatenated == provider._synthesis

    @pytest.mark.asyncio
    async def test_stream_false_produces_identical_output(self, sample_config, make_ctx_factory):
        """stream=False and stream=True produce identical final output.

        The run manifest contains a unique Run ID and Timestamp per run,
        so we strip the manifest comment block before comparing.
        """
        # Run with stream=False
        provider_no_stream = _MockStreamingProvider()
        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result_no_stream = await run_council(
                "What is the meaning of life?",
                sample_config,
                options=RunOptions(
                    context_factory=make_ctx_factory(provider_no_stream),
                    stream=False,
                ),
            )

        # Run with stream=True
        provider_stream = _MockStreamingProvider()
        chunks: list[str] = []

        async def on_chunk(chunk: str) -> None:
            chunks.append(chunk)

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result_stream = await run_council(
                "What is the meaning of life?",
                sample_config,
                options=RunOptions(
                    context_factory=make_ctx_factory(provider_stream),
                    stream=True,
                    on_chunk=on_chunk,
                ),
            )

        # Strip the run manifest block (contains unique Run ID and Timestamp)
        def strip_manifest(text: str) -> str:
            idx = text.find("<!-- Run Manifest")
            return text[:idx] if idx != -1 else text

        # Both runs should produce the same council content
        assert strip_manifest(result_no_stream) == strip_manifest(result_stream)

    @pytest.mark.asyncio
    async def test_stage3_fallback_on_mid_stream_error(self, sample_config, make_ctx_factory):
        """When astream fails mid-stream in Stage 3, the council falls back
        to query and still produces complete output."""
        provider = _MockFailingStreamProvider()
        chunks_received: list[str] = []

        async def on_chunk(chunk: str) -> None:
            chunks_received.append(chunk)

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "Test question for fallback",
                sample_config,
                options=RunOptions(
                    context_factory=make_ctx_factory(provider),
                    stream=True,
                    on_chunk=on_chunk,
                ),
            )

        # The council should still produce valid output via fallback
        assert "## LLM Council Response" in result
        assert "Synthesized Answer" in result
        # The fallback synthesis text should appear in the output
        assert provider._synthesis in result
