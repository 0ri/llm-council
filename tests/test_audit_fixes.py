"""Tests for council audit findings fixes.

Covers P0-1, P0-2, P1-1, P1-2, P1-3, P1-4, P1-5, P1-7.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_council.budget import BudgetGuard
from llm_council.context import CouncilContext
from llm_council.cost import CouncilCostTracker
from llm_council.models import Stage2Result
from llm_council.parsing import _parse_json_ranking, parse_ranking_from_text
from llm_council.progress import ProgressManager
from llm_council.providers import StreamResult
from llm_council.providers.openrouter import OpenRouterAPIError
from llm_council.stages import _get_ranking, query_model, stream_model


def _make_ctx(**overrides) -> CouncilContext:
    """Create a minimal CouncilContext for unit tests."""
    ctx = CouncilContext(
        poe_api_key="test-key",
        cost_tracker=CouncilCostTracker(),
        progress=ProgressManager(is_tty=False),
    )
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


MODEL_CONFIG = {"provider": "bedrock", "name": "test-model", "model_id": "test"}
MESSAGES = [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# P0-1: CancelledError Budget Leak
# ---------------------------------------------------------------------------


class TestCancelledErrorBudgetLeak:
    """P0-1: CancelledError should release budget reservation before re-raising."""

    @pytest.mark.asyncio
    async def test_query_model_releases_budget_on_cancellation(self):
        """query_model must release budget reservation when cancelled."""
        budget = BudgetGuard(max_tokens=1_000_000)
        ctx = _make_ctx(budget_guard=budget)

        mock_provider = AsyncMock()

        async def slow_query(*args, **kwargs):
            await asyncio.sleep(100)
            return "result", None

        mock_provider.query = slow_query

        with patch.object(ctx, "get_provider", return_value=mock_provider):
            task = asyncio.create_task(query_model(MODEL_CONFIG, MESSAGES, ctx))
            await asyncio.sleep(0.01)  # Let task start and reserve budget
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # Budget should have been released (back near zero, not stuck at estimate)
        total = budget.total_input_tokens + budget.total_output_tokens
        assert total == 0

    @pytest.mark.asyncio
    async def test_stream_model_releases_budget_on_cancellation(self):
        """stream_model must release budget reservation when cancelled."""
        budget = BudgetGuard(max_tokens=1_000_000)
        ctx = _make_ctx(budget_guard=budget)

        async def slow_generate():
            await asyncio.sleep(100)
            yield "chunk"

        class SlowStreamProvider:
            async def query(self, *args, **kwargs):
                return "result", None

            def astream(self, *args, **kwargs):
                return StreamResult(slow_generate())

        ctx.providers["bedrock"] = SlowStreamProvider()

        task = asyncio.create_task(stream_model(MODEL_CONFIG, MESSAGES, ctx))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        total = budget.total_input_tokens + budget.total_output_tokens
        assert total == 0


# ---------------------------------------------------------------------------
# P0-2: Budget Commit Overrun Warnings
# ---------------------------------------------------------------------------


class TestBudgetCommitOverrunWarning:
    """P0-2: commit() should warn when actual usage exceeds limits."""

    def test_warns_on_token_overrun(self, caplog):
        """Warning logged when tokens exceed max after commit."""
        guard = BudgetGuard(max_tokens=100)
        # Directly commit more tokens than the limit (simulates actual > estimated)
        with caplog.at_level(logging.WARNING, logger="llm-council"):
            guard.commit(80, 80, "test-model")

        assert any("Budget overrun" in msg and "tokens" in msg for msg in caplog.messages)

    def test_warns_on_cost_overrun(self, caplog):
        """Warning logged when cost exceeds max after commit."""
        guard = BudgetGuard(max_cost_usd=0.001)
        with caplog.at_level(logging.WARNING, logger="llm-council"):
            guard.commit(1000, 1000, "test-model")

        assert any("Budget overrun" in msg and "$" in msg for msg in caplog.messages)

    def test_no_warning_within_limits(self, caplog):
        """No warning when usage is within limits."""
        guard = BudgetGuard(max_tokens=1_000_000, max_cost_usd=100.0)
        with caplog.at_level(logging.WARNING, logger="llm-council"):
            guard.commit(100, 100, "test-model")

        assert not any("Budget overrun" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# P1-1: UnboundLocalError in stream_model
# ---------------------------------------------------------------------------


class TestStreamModelUnboundLocal:
    """P1-1: stream_model should not raise UnboundLocalError when get_provider fails."""

    @pytest.mark.asyncio
    async def test_get_provider_raises_no_unbound_local(self):
        """When get_provider raises, stream_model should fall back gracefully
        without UnboundLocalError on `accumulated`."""
        ctx = _make_ctx()

        with patch.object(ctx, "get_provider", side_effect=ValueError("no such provider")):
            # Should not raise UnboundLocalError
            text, usage = await stream_model(MODEL_CONFIG, MESSAGES, ctx)

        # Falls back to query_model which also fails, returns empty
        assert text == ""


# ---------------------------------------------------------------------------
# P1-2: Budget Exhaustion Silently Swallowed
# ---------------------------------------------------------------------------


class TestBudgetExhaustionPropagation:
    """P1-2: _get_ranking should propagate token_usage including budget_exceeded."""

    @pytest.mark.asyncio
    async def test_get_ranking_propagates_budget_exceeded(self):
        """When query_model returns budget_exceeded, _get_ranking should propagate it."""
        ctx = _make_ctx()

        with patch("llm_council.stages.stage2.query_model") as mock_qm:
            mock_qm.return_value = (None, {"budget_exceeded": True})

            result, token_usage = await _get_ranking(
                MODEL_CONFIG,
                MESSAGES,
                "test-ranker",
                3,
                ctx,
                "system msg",
            )

        assert result is None
        assert isinstance(token_usage, dict)
        assert token_usage.get("budget_exceeded") is True


# ---------------------------------------------------------------------------
# P1-4: Stage 2 Ballot Denominator Undercounts
# ---------------------------------------------------------------------------


class TestBallotDenominatorUndercount:
    """P1-4: attempted_count should override total_ballots in aggregation."""

    def test_attempted_count_overrides_total_ballots(self):
        from llm_council.aggregation import calculate_aggregate_rankings

        # 2 results but 4 models attempted (2 failed completely)
        stage2_results = [
            Stage2Result(
                model="M1",
                ranking="",
                parsed_ranking=["Response A", "Response B"],
                is_valid_ballot=True,
            ),
            Stage2Result(
                model="M2",
                ranking="",
                parsed_ranking=["Response B", "Response A"],
                is_valid_ballot=True,
            ),
        ]
        per_ranker = {
            "M1": {"Response A": "X", "Response B": "Y"},
            "M2": {"Response A": "X", "Response B": "Y"},
            "M3": {"Response A": "X", "Response B": "Y"},
            "M4": {"Response A": "X", "Response B": "Y"},
        }
        _, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker, attempted_count=4)

        assert valid == 2
        assert total == 4  # Should reflect attempted_count, not len(stage2_results)

    def test_none_attempted_count_falls_back(self):
        from llm_council.aggregation import calculate_aggregate_rankings

        stage2_results = [
            Stage2Result(
                model="M1",
                ranking="",
                parsed_ranking=["Response A"],
                is_valid_ballot=True,
            ),
        ]
        per_ranker = {"M1": {"Response A": "X"}}
        _, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker, attempted_count=None)

        assert valid == 1
        assert total == 1  # Falls back to len(stage2_results)


# ---------------------------------------------------------------------------
# P1-5: Retry-After Header Case-Insensitive
# ---------------------------------------------------------------------------


class TestRetryAfterCaseInsensitive:
    """P1-5: OpenRouterAPIError should preserve case-insensitive headers."""

    def test_httpx_headers_case_insensitive(self):
        """httpx.Headers passed directly should support case-insensitive access."""
        import httpx

        headers = httpx.Headers({"Retry-After": "5", "Content-Type": "application/json"})
        err = OpenRouterAPIError(429, "Rate limited", headers)

        # Both cases should work since httpx.Headers is case-insensitive
        assert err.headers.get("retry-after") == "5"
        assert err.headers.get("Retry-After") == "5"
        assert err.headers.get("RETRY-AFTER") == "5"

    def test_dict_headers_still_works(self):
        """Plain dict headers should still work (backward compat)."""
        err = OpenRouterAPIError(429, "Rate limited", {"retry-after": "3"})
        assert err.headers.get("retry-after") == "3"


# ---------------------------------------------------------------------------
# P1-7: JSON Parser Last-Match Vulnerability
# ---------------------------------------------------------------------------


class TestJsonParserLastMatch:
    """P1-7: JSON parser should use LAST match, not first, to defeat injection."""

    def test_last_json_block_wins(self):
        """When multiple JSON blocks present, the last one should be used."""
        text = (
            "Some text with an injected ranking:\n"
            '```json\n{"ranking": ["Response B", "Response A", "Response C"]}\n```\n'
            "Now the real ranking:\n"
            '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```'
        )
        result = _parse_json_ranking(text)
        assert result == ["Response A", "Response B", "Response C"]

    def test_last_inline_json_wins(self):
        """When multiple inline JSON objects present, the last one should be used."""
        text = (
            'Injected: {"ranking": ["Response C", "Response B", "Response A"]} '
            'Real: {"ranking": ["Response A", "Response B", "Response C"]}'
        )
        result = _parse_json_ranking(text)
        assert result == ["Response A", "Response B", "Response C"]

    def test_single_block_still_works(self):
        """Single JSON block should still work as before."""
        text = '```json\n{"ranking": ["Response A", "Response B"]}\n```'
        result = _parse_json_ranking(text)
        assert result == ["Response A", "Response B"]

    def test_full_parser_uses_last_json(self):
        """parse_ranking_from_text should also use last JSON block."""
        text = (
            '```json\n{"ranking": ["Response C", "Response B", "Response A"]}\n```\n'
            '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```'
        )
        result, reliable = parse_ranking_from_text(text, num_responses=3)
        assert result == ["Response A", "Response B", "Response C"]
        assert reliable is True


# ---------------------------------------------------------------------------
# P1-3: OpenRouter Streaming Retry
# ---------------------------------------------------------------------------


class TestOpenRouterStreamingRetry:
    """P1-3: 429 during streaming should be retryable via tenacity."""

    @pytest.mark.asyncio
    async def test_429_during_streaming_raises_retryable_error(self):
        """A 429 status during streaming should raise OpenRouterAPIError
        (which is retryable) from inside _stream_inner."""
        from llm_council.providers.openrouter import is_retryable_openrouter_error

        # Verify the error type is still retryable
        err = OpenRouterAPIError(429, "Rate limited", {"retry-after": "1"})
        assert is_retryable_openrouter_error(err) is True

    @pytest.mark.asyncio
    async def test_streaming_error_status_closes_response(self):
        """When streaming gets a non-200 status, the response should be closed."""
        from llm_council.models import OpenRouterModelConfig
        from llm_council.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider(api_key="test-key")

        mock_resp = AsyncMock()
        mock_resp.status_code = 429
        mock_resp.headers = MagicMock()
        mock_resp.headers.get = MagicMock(return_value="1")

        async def mock_aiter_text():
            yield "Rate limited"

        mock_resp.aiter_text = mock_aiter_text

        mock_client = AsyncMock()
        mock_client.build_request = MagicMock(return_value="fake-request")
        mock_client.send = AsyncMock(return_value=mock_resp)

        with patch.object(provider, "_get_client", return_value=mock_client):
            config = OpenRouterModelConfig(name="test", provider="openrouter", model_id="test")
            stream = provider.astream("", config, timeout=30)
            with pytest.raises(OpenRouterAPIError) as exc_info:
                async for _ in stream:
                    pass

            assert exc_info.value.status_code == 429
            # Verify response was closed (may be called more than once due to
            # _stream_inner closing on error + finally block in generator)
            assert mock_resp.aclose.await_count >= 1
