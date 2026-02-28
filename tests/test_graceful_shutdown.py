"""Property-based tests for graceful shutdown.

Feature: graceful-shutdown
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from llm_council.context import CouncilContext
from llm_council.progress import ProgressManager

# --- Strategies ---


@st.composite
def progress_manager_states(draw):
    """Generate a ProgressManager in a random state.

    Produces managers with/without an active render task and with/without
    a live display, covering the four combinations that shutdown() must handle.
    """
    has_render_task = draw(st.booleans())
    has_live = draw(st.booleans())

    with patch("llm_council.progress.Console"), patch("llm_council.progress.Live") as mock_live_cls:
        mock_live_instance = Mock()
        mock_live_cls.return_value = mock_live_instance

        # Create in non-TTY mode so __init__ doesn't auto-create Live
        manager = ProgressManager(is_tty=False)

        if has_render_task:
            # Create a real async task that runs a short coroutine
            async def _dummy_loop():
                try:
                    while True:
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    pass

            manager._render_task = asyncio.create_task(_dummy_loop())

        if has_live:
            manager._live = mock_live_instance

    return manager


# --- Property 1: Shutdown stops render resources ---


@pytest.mark.asyncio
@given(data=st.data())
@settings(max_examples=100)
async def test_shutdown_stops_render_resources(data):
    """Property 1: Shutdown stops render resources

    For any ProgressManager that has an active render loop task and/or live
    display, calling shutdown() should result in the render task being done
    (cancelled) and the live display being None.

    **Validates: Requirements 1.1, 1.2, 1.3**

    Tag: Feature: graceful-shutdown, Property 1: Shutdown stops render resources
    """
    manager = data.draw(progress_manager_states())

    await manager.shutdown()

    # _render_task must be None or done (cancelled)
    assert manager._render_task is None or manager._render_task.done(), (
        f"_render_task should be None or done after shutdown(), got {manager._render_task}"
    )

    # _live must be None
    assert manager._live is None, f"_live should be None after shutdown(), got {manager._live}"


# --- Unit Tests for ProgressManager shutdown (Task 1.3) ---


@pytest.mark.asyncio
async def test_shutdown_on_fresh_progress_manager():
    """Call shutdown() on a ProgressManager that was never started.

    A fresh manager has no render task and no live display.
    shutdown() should complete without raising.

    Validates: Requirements 1.3
    """
    manager = ProgressManager(is_tty=False)
    # Should not raise
    await manager.shutdown()

    assert manager._shutdown_called is True
    assert manager._render_task is None
    assert manager._live is None


@pytest.mark.asyncio
async def test_shutdown_idempotent():
    """Call shutdown() twice — _cleanup logic should only run once.

    Validates: Requirements 1.3, 1.4
    """
    with patch("llm_council.progress.Console"), patch("llm_council.progress.Live") as mock_live_cls:
        mock_live_instance = Mock()
        mock_live_cls.return_value = mock_live_instance

        manager = ProgressManager(is_tty=False)
        mock_live_instance.reset_mock()

        # Give it a live display so _cleanup has something to do
        manager._live = mock_live_instance

        # First shutdown — should run _cleanup
        await manager.shutdown()
        assert manager._shutdown_called is True
        assert manager._live is None
        # Live.stop() should have been called once
        mock_live_instance.stop.assert_called_once()

        # Reset mock to track second call
        mock_live_instance.reset_mock()

        # Second shutdown — should be a no-op
        await manager.shutdown()
        # stop() should NOT have been called again
        mock_live_instance.stop.assert_not_called()


# --- Strategies for Property 2 ---


@st.composite
def mock_provider_list(draw):
    """Generate 0–5 mock providers, each randomly configured to raise or succeed on close()."""
    count = draw(st.integers(min_value=0, max_value=5))
    providers = {}
    for i in range(count):
        should_raise = draw(st.booleans())
        provider = Mock()
        if should_raise:
            provider.close = AsyncMock(side_effect=RuntimeError(f"provider-{i} close failed"))
        else:
            provider.close = AsyncMock()
        providers[f"provider-{i}"] = provider
    return providers


@st.composite
def mock_cache_strategy(draw):
    """Generate a mock cache that randomly raises or succeeds on close()."""
    has_cache = draw(st.booleans())
    if not has_cache:
        return None
    should_raise = draw(st.booleans())
    cache = Mock()
    if should_raise:
        cache.close = Mock(side_effect=RuntimeError("cache close failed"))
    else:
        cache.close = Mock()
    return cache


@st.composite
def mock_progress_strategy(draw):
    """Generate a mock ProgressManager that randomly raises or succeeds on shutdown()."""
    should_raise = draw(st.booleans())
    progress = Mock(spec=ProgressManager)
    if should_raise:
        progress.shutdown = AsyncMock(side_effect=RuntimeError("progress shutdown failed"))
    else:
        progress.shutdown = AsyncMock()
    return progress


# --- Property 2: Resilient resource cleanup ---


@pytest.mark.asyncio
@given(data=st.data())
@settings(max_examples=100)
async def test_resilient_resource_cleanup(data):
    """Property 2: Resilient resource cleanup

    For any CouncilContext containing an arbitrary set of providers (some of
    whose close() methods raise exceptions), a ResponseCache (whose close()
    may raise), and a ProgressManager (whose shutdown() may raise), calling
    CouncilContext.shutdown() should: (a) attempt close() on every provider
    (up to the first failure, since close() doesn't catch per-provider errors),
    (b) attempt close() on the cache (if no provider raised), (c) attempt
    shutdown() on the progress manager, and (d) never propagate an exception
    to the caller.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

    Tag: Feature: graceful-shutdown, Property 2: Resilient resource cleanup
    """
    providers = data.draw(mock_provider_list())
    cache = data.draw(mock_cache_strategy())
    progress = data.draw(mock_progress_strategy())

    ctx = CouncilContext()
    ctx.providers = providers
    ctx.cache = cache
    ctx.progress = progress

    # shutdown() must never propagate an exception
    await ctx.shutdown()

    # Determine which providers should have had close() called.
    # The existing close() iterates providers in order and does NOT catch
    # per-provider exceptions. So close() is called on each provider until
    # one raises, at which point the remaining providers are skipped.
    provider_list = list(providers.values())
    first_failure_idx = None
    for idx, p in enumerate(provider_list):
        if p.close.side_effect is not None:
            first_failure_idx = idx
            break

    # Providers up to and including the first failure should have close() called
    if first_failure_idx is not None:
        for idx in range(first_failure_idx + 1):
            provider_list[idx].close.assert_awaited_once()
        # Providers after the first failure are NOT called (close() doesn't catch)
        for idx in range(first_failure_idx + 1, len(provider_list)):
            provider_list[idx].close.assert_not_awaited()
    else:
        # No failures — all providers should have close() called
        for p in provider_list:
            p.close.assert_awaited_once()

    # Cache close() should be called if no provider raised (because cache close
    # happens after provider iteration in close()). If a provider raised,
    # close() threw before reaching cache.
    if cache is not None:
        if first_failure_idx is None:
            cache.close.assert_called_once()
        # If a provider raised, cache.close may or may not have been called
        # (it won't be, since close() exits on the exception)

    # Progress shutdown() must ALWAYS be called — shutdown() wraps close()
    # in try/except, so progress.shutdown() runs regardless
    progress.shutdown.assert_awaited_once()


# --- Property 3: Shutdown idempotency ---


@pytest.mark.asyncio
@given(data=st.data())
@settings(max_examples=100)
async def test_shutdown_idempotency(data):
    """Property 3: Shutdown idempotency

    For any CouncilContext, calling shutdown() N times (N >= 1) should produce
    the same observable side effects as calling it exactly once. Specifically,
    each provider's close(), cache close(), and progress shutdown() should each
    be invoked exactly once regardless of how many times shutdown() is called.

    **Validates: Requirements 1.4, 5.1, 5.2**

    Tag: Feature: graceful-shutdown, Property 3: Shutdown idempotency
    """
    call_count = data.draw(st.integers(min_value=1, max_value=10))
    provider_count = data.draw(st.integers(min_value=0, max_value=5))

    # Build providers that all succeed on close()
    providers = {}
    for i in range(provider_count):
        provider = Mock()
        provider.close = AsyncMock()
        providers[f"provider-{i}"] = provider

    # Build a cache that succeeds on close()
    cache = Mock()
    cache.close = Mock()

    # Build a progress manager that succeeds on shutdown()
    progress = Mock(spec=ProgressManager)
    progress.shutdown = AsyncMock()

    ctx = CouncilContext()
    ctx.providers = providers
    ctx.cache = cache
    ctx.progress = progress

    # Call shutdown() N times
    for _ in range(call_count):
        await ctx.shutdown()

    # Each provider's close() must have been called exactly once
    for name, provider in providers.items():
        (
            provider.close.assert_awaited_once(),
            (
                f"{name}.close() should be called exactly once, "
                f"but was called {provider.close.await_count} times after {call_count} shutdown() calls"
            ),
        )

    # Cache close() must have been called exactly once
    cache.close.assert_called_once()

    # Progress shutdown() must have been called exactly once
    progress.shutdown.assert_awaited_once()


# --- Unit Tests for CouncilContext shutdown (Task 2.4) ---


@pytest.mark.asyncio
async def test_shutdown_after_close():
    """Call close() then shutdown() — progress shutdown() should still be called, no exception.

    When close() is called first, providers and cache are already cleaned up.
    A subsequent shutdown() should still call progress.shutdown() and not raise,
    even though _shutdown_called was False before shutdown() was invoked.

    Validates: Requirements 4.1, 4.2, 5.2
    """
    # Build a provider that succeeds on close()
    provider = Mock()
    provider.close = AsyncMock()

    cache = Mock()
    cache.close = Mock()

    progress = Mock(spec=ProgressManager)
    progress.shutdown = AsyncMock()

    ctx = CouncilContext()
    ctx.providers = {"p0": provider}
    ctx.cache = cache
    ctx.progress = progress

    # First: call close() directly — closes providers and cache
    await ctx.close()
    provider.close.assert_awaited_once()
    cache.close.assert_called_once()

    # Then: call shutdown() — should still run (flag was not set by close())
    await ctx.shutdown()

    # progress.shutdown() must have been called
    progress.shutdown.assert_awaited_once()

    # provider.close() will have been called twice total (once by close(), once
    # by shutdown() which internally calls close() again), but no exception
    assert provider.close.await_count == 2
    assert cache.close.call_count == 2


@pytest.mark.asyncio
async def test_shutdown_delegates_to_close():
    """Mock _close_resources(), call shutdown(), verify it was called.

    Validates: Requirements 4.2
    """
    progress = Mock(spec=ProgressManager)
    progress.shutdown = AsyncMock()

    ctx = CouncilContext()
    ctx.progress = progress

    with patch.object(ctx, "_close_resources", new_callable=AsyncMock) as mock_close:
        await ctx.shutdown()
        mock_close.assert_awaited_once()

    # progress.shutdown() should also have been called
    progress.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_preserved():
    """Call close() directly with providers — verify providers are closed and cache is closed.

    This ensures the existing close() behavior is unchanged (Requirement 4.1).

    Validates: Requirements 4.1
    """
    provider_a = Mock()
    provider_a.close = AsyncMock()
    provider_b = Mock()
    provider_b.close = AsyncMock()

    cache = Mock()
    cache.close = Mock()

    ctx = CouncilContext()
    ctx.providers = {"a": provider_a, "b": provider_b}
    ctx.cache = cache

    await ctx.close()

    provider_a.close.assert_awaited_once()
    provider_b.close.assert_awaited_once()
    cache.close.assert_called_once()


# --- Unit Tests for run_council finally block (Task 4.2) ---


@pytest.mark.asyncio
async def test_finally_calls_shutdown_on_success():
    """Run council to success and verify shutdown() was called.

    We use context_factory to inject a mock CouncilContext, then patch the
    stage functions to return canned results so run_council completes normally.

    Validates: Requirements 3.2
    """
    from llm_council.council import run_council

    # Build a mock context
    progress = Mock(spec=ProgressManager)
    progress.shutdown = AsyncMock()
    progress.complete_council = AsyncMock()
    progress.update_stage1 = Mock()
    progress.update_stage2 = Mock()
    progress.update_stage3 = Mock()
    progress.start_stage = Mock()

    ctx = Mock(spec=CouncilContext)
    ctx.shutdown = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=ctx)

    async def _mock_aexit(*args):
        await ctx.shutdown()

    ctx.__aexit__ = _mock_aexit
    ctx.progress = progress
    ctx.budget_guard = None
    ctx.cache = None
    ctx.cost_tracker = Mock()
    ctx.cost_tracker.record_with_usage = Mock()
    ctx.cost_tracker.summary = Mock(return_value="summary")
    ctx.cost_tracker.total_tokens = 100

    # Stage 1 results
    stage1_result = Mock()
    stage1_result.model = "test-model"
    stage1_result.response = "test response"

    # Stage 2 results
    stage2_result = Mock()
    stage2_result.model = "test-model"
    stage2_result.ranking = "A > B"

    # Stage 3 result
    stage3_result = Mock()
    stage3_result.model = "test-model"
    stage3_result.response = "final answer"

    config = {"council_models": [{"name": "test-model", "provider": "poe", "bot_name": "test"}], "chairman": {"name": "test-model", "provider": "poe", "bot_name": "test"}}

    with (
        patch("llm_council.council.validate_config", return_value=[]),
        patch("llm_council.council.stage1_collect_responses", new_callable=AsyncMock) as mock_s1,
        patch("llm_council.council.stage2_collect_rankings", new_callable=AsyncMock) as mock_s2,
        patch("llm_council.council.stage3_synthesize_final", new_callable=AsyncMock) as mock_s3,
        patch("llm_council.council.calculate_aggregate_rankings") as mock_agg,
        patch("llm_council.council.format_output", return_value="output"),
    ):
        mock_s1.return_value = ([stage1_result], {"test-model": None})
        mock_s2.return_value = ([stage2_result], {"ranker": {"A": "m1"}}, {"test-model": None})
        mock_agg.return_value = ({"test-model": 1}, 1, 1)
        mock_s3.return_value = (stage3_result, None)

        await run_council("test question", config, context_factory=lambda: ctx)

    ctx.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_finally_calls_shutdown_on_budget_error():
    """Raise BudgetExceededError during stage 1 and verify shutdown() was called.

    Validates: Requirements 3.3
    """
    from llm_council.budget import BudgetExceededError
    from llm_council.council import run_council

    progress = Mock(spec=ProgressManager)
    progress.shutdown = AsyncMock()
    progress.start_stage = Mock()

    ctx = Mock(spec=CouncilContext)
    ctx.shutdown = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=ctx)

    async def _mock_aexit(*args):
        await ctx.shutdown()

    ctx.__aexit__ = _mock_aexit
    ctx.progress = progress
    ctx.budget_guard = None
    ctx.cache = None
    ctx.cost_tracker = Mock()

    config = {"council_models": [{"name": "test-model", "provider": "poe", "bot_name": "test"}], "chairman": {"name": "test-model", "provider": "poe", "bot_name": "test"}}

    with (
        patch("llm_council.council.validate_config", return_value=[]),
        patch(
            "llm_council.council.stage1_collect_responses",
            new_callable=AsyncMock,
            side_effect=BudgetExceededError("token limit reached"),
        ),
    ):
        result = await run_council("test question", config, context_factory=lambda: ctx)

    assert "Budget limit exceeded" in result
    ctx.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_finally_calls_shutdown_on_unexpected_error():
    """Raise a generic exception during stage 1 and verify shutdown() was called.

    Validates: Requirements 3.4
    """
    from llm_council.council import run_council

    progress = Mock(spec=ProgressManager)
    progress.shutdown = AsyncMock()
    progress.start_stage = Mock()

    ctx = Mock(spec=CouncilContext)
    ctx.shutdown = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=ctx)

    async def _mock_aexit(*args):
        await ctx.shutdown()

    ctx.__aexit__ = _mock_aexit
    ctx.progress = progress
    ctx.budget_guard = None
    ctx.cache = None
    ctx.cost_tracker = Mock()

    config = {"council_models": [{"name": "test-model", "provider": "poe", "bot_name": "test"}], "chairman": {"name": "test-model", "provider": "poe", "bot_name": "test"}}

    with (
        patch("llm_council.council.validate_config", return_value=[]),
        patch(
            "llm_council.council.stage1_collect_responses",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected failure"),
        ),
    ):
        with pytest.raises(RuntimeError, match="unexpected failure"):
            await run_council("test question", config, context_factory=lambda: ctx)

    ctx.shutdown.assert_awaited_once()
