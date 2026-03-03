"""Invariant tests for the execution guard lifecycle.

Verifies that budget reservation, commit, and release happen correctly
across success, failure, cancellation, and streaming fallback paths.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_council.context import CouncilContext
from llm_council.stages.execution import _managed_execution, query_model, stream_model


def _make_ctx(*, budget: bool = True) -> CouncilContext:
    """Build a minimal CouncilContext with mock budget guard and providers."""
    ctx = MagicMock(spec=CouncilContext)
    ctx.progress = None
    ctx.stage2_max_retries = 1

    # Circuit breaker: default to closed
    cb = MagicMock()
    cb.is_open = False
    ctx.get_circuit_breaker.return_value = cb

    # Semaphore: no-op
    sem = asyncio.Semaphore(10)
    ctx.get_semaphore.return_value = sem

    # Budget guard
    if budget:
        bg = AsyncMock()
        bg.areserve = AsyncMock()
        bg.acommit = AsyncMock()
        bg.arelease = AsyncMock()
        ctx.budget_guard = bg
    else:
        ctx.budget_guard = None

    return ctx


def _make_config() -> dict:
    """Return a minimal model config dict for testing."""
    return {"provider": "openrouter", "model_id": "test/model", "name": "test-model"}


class TestManagedExecutionGuard:
    """Tests for the _managed_execution context manager."""

    @pytest.mark.asyncio
    async def test_budget_reserved_before_yield(self):
        """Budget is reserved before the caller body executes."""
        ctx = _make_ctx()
        reserve_called = False

        original_reserve = ctx.budget_guard.areserve

        async def tracking_reserve(*args, **kwargs):
            nonlocal reserve_called
            reserve_called = True
            return await original_reserve(*args, **kwargs)

        ctx.budget_guard.areserve = tracking_reserve

        async with _managed_execution(_make_config(), [{"role": "user", "content": "hi"}], ctx) as guard:
            assert reserve_called, "Budget should be reserved before yield"
            guard.token_usage = {"input_tokens": 10, "output_tokens": 5}

    @pytest.mark.asyncio
    async def test_budget_committed_on_success(self):
        """Budget is committed (not released) when the context exits normally."""
        ctx = _make_ctx()

        async with _managed_execution(_make_config(), [{"role": "user", "content": "hi"}], ctx) as guard:
            guard.token_usage = {"input_tokens": 10, "output_tokens": 5}

        ctx.budget_guard.acommit.assert_awaited_once()
        ctx.budget_guard.arelease.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_budget_released_on_exception(self):
        """Budget is released (not committed) when an exception occurs."""
        ctx = _make_ctx()

        with pytest.raises(RuntimeError):
            async with _managed_execution(_make_config(), [{"role": "user", "content": "hi"}], ctx):
                raise RuntimeError("provider error")

        ctx.budget_guard.arelease.assert_awaited_once()
        ctx.budget_guard.acommit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_budget_released_on_cancellation(self):
        """Budget is released when the task is cancelled."""
        ctx = _make_ctx()

        with pytest.raises(asyncio.CancelledError):
            async with _managed_execution(_make_config(), [{"role": "user", "content": "hi"}], ctx):
                raise asyncio.CancelledError()

        ctx.budget_guard.arelease.assert_awaited_once()
        ctx.budget_guard.acommit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_circuit_breaker_success_on_normal_exit(self):
        """Circuit breaker records success when context exits normally."""
        ctx = _make_ctx(budget=False)

        async with _managed_execution(_make_config(), [{"role": "user", "content": "hi"}], ctx):
            pass

        cb = ctx.get_circuit_breaker.return_value
        cb.record_success.assert_called_once()
        cb.record_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_on_exception(self):
        """Circuit breaker records failure when an exception occurs."""
        ctx = _make_ctx(budget=False)

        with pytest.raises(ValueError):
            async with _managed_execution(_make_config(), [{"role": "user", "content": "hi"}], ctx):
                raise ValueError("boom")

        cb = ctx.get_circuit_breaker.return_value
        cb.record_failure.assert_called_once()
        cb.record_success.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_budget_ops_without_guard(self):
        """No budget operations when budget_guard is None."""
        ctx = _make_ctx(budget=False)

        async with _managed_execution(_make_config(), [{"role": "user", "content": "hi"}], ctx):
            pass

        # No budget guard — nothing to assert on, just verify no exception


class TestQueryModelInvariants:
    """Invariant tests for query_model budget lifecycle."""

    @pytest.mark.asyncio
    async def test_budget_committed_on_success(self):
        """query_model commits budget on successful provider response."""
        ctx = _make_ctx()
        mock_provider = AsyncMock()
        mock_provider.query = AsyncMock(return_value=("response text", {"input_tokens": 10, "output_tokens": 5}))
        ctx.get_provider.return_value = mock_provider

        result, usage = await query_model(_make_config(), [{"role": "user", "content": "hi"}], ctx)

        assert result is not None
        ctx.budget_guard.acommit.assert_awaited_once()
        ctx.budget_guard.arelease.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_budget_released_on_timeout(self):
        """query_model releases budget on timeout."""
        ctx = _make_ctx()
        mock_provider = AsyncMock()
        mock_provider.query = AsyncMock(side_effect=asyncio.TimeoutError())
        ctx.get_provider.return_value = mock_provider

        result, usage = await query_model(_make_config(), [{"role": "user", "content": "hi"}], ctx)

        assert result is None
        ctx.budget_guard.arelease.assert_awaited_once()
        ctx.budget_guard.acommit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_budget_released_on_provider_error(self):
        """query_model releases budget on provider exception."""
        ctx = _make_ctx()
        mock_provider = AsyncMock()
        mock_provider.query = AsyncMock(side_effect=RuntimeError("provider down"))
        ctx.get_provider.return_value = mock_provider

        result, usage = await query_model(_make_config(), [{"role": "user", "content": "hi"}], ctx)

        assert result is None
        ctx.budget_guard.arelease.assert_awaited_once()
        ctx.budget_guard.acommit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_budget_exceeded_returns_sentinel(self):
        """query_model returns budget_exceeded sentinel without commit or release."""
        from llm_council.budget import BudgetExceededError

        ctx = _make_ctx()
        ctx.budget_guard.areserve = AsyncMock(side_effect=BudgetExceededError("over budget"))

        result, usage = await query_model(_make_config(), [{"role": "user", "content": "hi"}], ctx)

        assert result is None
        assert usage == {"budget_exceeded": True}
        ctx.budget_guard.acommit.assert_not_awaited()
        ctx.budget_guard.arelease.assert_not_awaited()


class TestStreamModelInvariants:
    """Invariant tests for stream_model budget lifecycle."""

    @pytest.mark.asyncio
    async def test_stream_fallback_no_double_charge(self):
        """When streaming fails and falls back to query_model, budget is not double-charged.

        The stream's budget is released on error, then query_model manages its own.
        """
        ctx = _make_ctx()

        # Streaming provider that fails
        mock_provider = MagicMock()
        mock_provider.astream = MagicMock(side_effect=RuntimeError("stream broke"))

        # Make provider non-StreamingProvider so fallback_astream is used
        ctx.get_provider.return_value = mock_provider

        # query_model fallback needs a working provider
        with patch(
            "llm_council.stages.execution.query_model",
            new_callable=AsyncMock,
            return_value=({"content": "fallback response"}, {"input_tokens": 10, "output_tokens": 5}),
        ):
            with patch(
                "llm_council.stages.execution.fallback_astream",
                side_effect=RuntimeError("stream broke"),
            ):
                result, usage = await stream_model(_make_config(), [{"role": "user", "content": "hi"}], ctx)

        assert result == "fallback response"
        # Budget was reserved once for stream, released on failure
        ctx.budget_guard.areserve.assert_awaited_once()
        ctx.budget_guard.arelease.assert_awaited_once()
        # Commit NOT called by stream_model (query_model fallback manages its own)
        ctx.budget_guard.acommit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_budget_released_on_cancellation(self):
        """stream_model releases budget when cancelled."""
        ctx = _make_ctx()

        async def slow_stream(*args, **kwargs):
            yield "chunk"
            await asyncio.sleep(100)  # Will be cancelled

        mock_provider = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__aiter__ = slow_stream
        mock_provider.astream.return_value = mock_stream

        # Use isinstance check bypass
        with patch("llm_council.stages.execution.isinstance", return_value=False):
            with patch(
                "llm_council.stages.execution.fallback_astream",
            ) as mock_fallback:
                # Make fallback raise CancelledError
                async def cancelled_stream(*args, **kwargs):
                    raise asyncio.CancelledError()
                    yield  # noqa: F811

                mock_fallback.side_effect = asyncio.CancelledError()

                with pytest.raises(asyncio.CancelledError):
                    await stream_model(_make_config(), [{"role": "user", "content": "hi"}], ctx)

        ctx.budget_guard.arelease.assert_awaited_once()
        ctx.budget_guard.acommit.assert_not_awaited()
