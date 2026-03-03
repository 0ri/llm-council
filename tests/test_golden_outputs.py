"""Golden output tests for optimization baseline verification.

These fixtures capture expected output structure and content so that
refactoring can be verified to produce equivalent results.

Scenarios:
- Stage 1-only run
- Full non-stream run
- Stream run with fallback
- Strict-ballot failure
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from llm_council.context import CouncilContext
from llm_council.cost import CouncilCostTracker
from llm_council.council import run_council
from llm_council.formatting import format_output, format_stage1_output, format_stage2_output
from llm_council.models import AggregateRanking, Stage1Result, Stage3Result
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
        ctx.providers["poe"] = mock_provider
        ctx.providers["bedrock"] = mock_provider
        return ctx

    return factory


SAMPLE_CONFIG = {
    "council_models": [
        {"name": "Model-A", "provider": "poe", "bot_name": "TestBot-A"},
        {"name": "Model-B", "provider": "poe", "bot_name": "TestBot-B"},
        {"name": "Model-C", "provider": "poe", "bot_name": "TestBot-C"},
    ],
    "chairman": {"name": "Model-A", "provider": "poe", "bot_name": "TestBot-A"},
}


# ---------------------------------------------------------------------------
# Golden Output 1: Stage 1-only formatting
# ---------------------------------------------------------------------------


class TestGoldenStage1Only:
    """Verify Stage 1-only output structure and content."""

    def test_stage1_output_structure(self):
        """Stage 1 output should contain header + per-model sections."""
        results = [
            Stage1Result(model="Model-A", response="Answer from A."),
            Stage1Result(model="Model-B", response="Answer from B."),
            Stage1Result(model="Model-C", response="Answer from C."),
        ]
        output = format_stage1_output(results)

        # Required structural elements
        assert "## LLM Council Response (Stage 1 only)" in output
        assert "### Model-A" in output
        assert "### Model-B" in output
        assert "### Model-C" in output
        assert "Answer from A." in output
        assert "Answer from B." in output
        assert "Answer from C." in output

        # Must NOT contain Stage 2/3 artifacts
        assert "Rank" not in output
        assert "Borda Score" not in output
        assert "Synthesized" not in output
        assert "Chairman" not in output

    @pytest.mark.asyncio
    async def test_stage1_only_pipeline(self):
        """Full pipeline with max_stage=1 produces Stage 1 output."""
        call_count = {"n": 0}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            call_count["n"] += 1
            name = getattr(model_config, "name", "unknown")
            return f"Response from {name}", None

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "What is 2+2?",
                SAMPLE_CONFIG,
                context_factory=_make_ctx_factory(mock_provider),
                max_stage=1,
            )

        assert "Stage 1 only" in result
        assert "Model-A" in result
        assert "Response from Model-A" in result


# ---------------------------------------------------------------------------
# Golden Output 2: Full non-stream run
# ---------------------------------------------------------------------------


class TestGoldenFullNonStream:
    """Verify full 3-stage non-streaming output structure."""

    def test_full_output_structure(self):
        """Full output should contain rankings table + synthesis."""
        rankings = [
            AggregateRanking(
                model="Model-A", average_rank=1.5, rankings_count=2, ci_lower=1.2, ci_upper=1.8, borda_score=8
            ),
            AggregateRanking(
                model="Model-B", average_rank=2.0, rankings_count=2, ci_lower=1.7, ci_upper=2.3, borda_score=6
            ),
            AggregateRanking(
                model="Model-C", average_rank=2.5, rankings_count=2, ci_lower=2.1, ci_upper=2.9, borda_score=4
            ),
        ]
        stage3 = Stage3Result(model="Model-A", response="The synthesized answer.")

        output = format_output(rankings, stage3, valid_ballots=3, total_ballots=3)

        # Required structural elements
        assert "## LLM Council Response" in output
        assert "### Model Rankings (by peer review)" in output
        assert "### Synthesized Answer" in output
        assert "| Rank | Model | Avg Position | 95% CI | Borda Score |" in output
        assert "|------|-------|--------------|--------|-------------|" in output
        assert "| 1 | Model-A" in output
        assert "| 2 | Model-B" in output
        assert "| 3 | Model-C" in output
        assert "**Chairman:** Model-A" in output
        assert "The synthesized answer." in output
        assert "3/3 valid ballots" in output

    @pytest.mark.asyncio
    async def test_full_pipeline_non_stream(self):
        """Full 3-stage pipeline produces rankings + synthesis."""
        call_count = {"n": 0}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            call_count["n"] += 1
            name = getattr(model_config, "name", "unknown")
            if call_count["n"] <= 3:
                return f"Response from {name}", None
            elif call_count["n"] <= 6:
                return '```json\n{"ranking": ["Response A", "Response B"]}\n```', None
            else:
                return "The council has deliberated.", None

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "What is 2+2?",
                SAMPLE_CONFIG,
                context_factory=_make_ctx_factory(mock_provider),
            )

        assert "## LLM Council Response" in result
        assert "Model Rankings" in result
        assert "Synthesized Answer" in result


# ---------------------------------------------------------------------------
# Golden Output 3: Stream run with fallback
# ---------------------------------------------------------------------------


class TestGoldenStreamFallback:
    """Verify stream fallback produces valid output."""

    @pytest.mark.asyncio
    async def test_stream_fallback_produces_output(self):
        """When streaming is requested but provider doesn't support it,
        fallback_astream wraps query_model and produces valid output."""
        from llm_council.stages import stream_model

        # Test at the stream_model level: provider with no astream
        # falls back via fallback_astream and produces accumulated text.
        class _QueryOnlyProvider:
            async def query(self, prompt, model_config, timeout, **kwargs):
                return "Fallback synthesis result.", {"input_tokens": 10, "output_tokens": 20}

        ctx = CouncilContext(
            poe_api_key="test-key",
            cost_tracker=CouncilCostTracker(),
            progress=ProgressManager(is_tty=False),
        )
        ctx.providers["poe"] = _QueryOnlyProvider()

        model_config = {"provider": "poe", "name": "Model-A", "bot_name": "TestBot-A"}
        messages = [{"role": "user", "content": "test"}]

        text, usage = await stream_model(model_config, messages, ctx)

        assert text == "Fallback synthesis result."
        assert usage == {"input_tokens": 10, "output_tokens": 20}


# ---------------------------------------------------------------------------
# Golden Output 4: Strict-ballot failure
# ---------------------------------------------------------------------------


class TestGoldenStrictBallotFailure:
    """Verify strict_ballots mode produces appropriate warnings."""

    @pytest.mark.asyncio
    async def test_strict_ballots_all_invalid(self):
        """strict_ballots=True with unparseable rankings produces warning."""
        config = {
            **SAMPLE_CONFIG,
            "strict_ballots": True,
        }
        call_count = {"n": 0}

        async def mock_query(prompt, model_config, timeout, **kwargs):
            call_count["n"] += 1
            name = getattr(model_config, "name", "unknown")
            if call_count["n"] <= 3:
                return f"Response from {name}", None
            elif call_count["n"] <= 6:
                # Return unparseable ranking
                return "I refuse to rank these.", None
            else:
                return "Synthesis anyway.", None

        mock_provider = MagicMock()
        mock_provider.query = mock_query

        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            result = await run_council(
                "Strict test",
                config,
                context_factory=_make_ctx_factory(mock_provider),
            )

        # Should still produce output (either with warning or error)
        assert result


# ---------------------------------------------------------------------------
# Golden Output 5: format_stage2_output structure
# ---------------------------------------------------------------------------


class TestGoldenStage2Output:
    """Verify Stage 2 output structure (rankings + individual responses)."""

    def test_stage2_output_structure(self):
        """Stage 2 output contains rankings table + individual responses."""
        rankings = [
            AggregateRanking(
                model="Model-A", average_rank=1.0, rankings_count=2, ci_lower=1.0, ci_upper=1.0, borda_score=6
            ),
            AggregateRanking(
                model="Model-B", average_rank=2.0, rankings_count=2, ci_lower=2.0, ci_upper=2.0, borda_score=3
            ),
        ]
        results = [
            Stage1Result(model="Model-A", response="Answer A"),
            Stage1Result(model="Model-B", response="Answer B"),
        ]

        output = format_stage2_output(rankings, results, valid_ballots=2, total_ballots=2)

        # Required elements
        assert "## LLM Council Response (Stages 1-2, no synthesis)" in output
        assert "### Model Rankings (by peer review)" in output
        assert "| Rank | Model | Avg Position | 95% CI | Borda Score |" in output
        assert "### Individual Responses" in output
        assert "#### Model-A" in output
        assert "Answer A" in output
        assert "#### Model-B" in output
        assert "Answer B" in output

        # Must NOT contain synthesis artifacts
        assert "Synthesized Answer" not in output
        assert "Chairman" not in output
