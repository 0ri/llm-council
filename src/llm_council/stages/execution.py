"""Low-level model execution: query, stream, and parallel dispatch.

Provides ``query_model``, ``stream_model``, and ``query_models_parallel``
with budget tracking, circuit-breaker integration, and soft timeouts.

Budget and circuit-breaker lifecycle is centralized in ``_managed_execution``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from ..budget import BudgetExceededError
from ..context import CouncilContext
from ..cost import estimate_tokens
from ..models import ModelConfig, coerce_model_config
from ..progress import ModelStatus
from ..providers import MODEL_TIMEOUT, SOFT_TIMEOUT, ProviderRequest, StreamingProvider, fallback_astream

logger = logging.getLogger("llm-council")

_DEFAULT_OUTPUT_ESTIMATE = 2000  # Conservative output token estimate


def _estimate_request_tokens(
    messages: list[dict[str, str]], system_message: str | None = None
) -> tuple[int, int]:
    """Return (estimated_input_tokens, estimated_output_tokens) for a request."""
    input_text = " ".join(m.get("content", "") for m in messages)
    if system_message:
        input_text += " " + system_message
    return estimate_tokens(input_text), _DEFAULT_OUTPUT_ESTIMATE


def _actual_or_estimated(
    token_usage: dict[str, Any] | None,
    estimated_input: int,
    estimated_output: int,
) -> tuple[int, int]:
    """Prefer actual token counts from provider, fall back to estimates."""
    if token_usage:
        return (
            token_usage.get("input_tokens", estimated_input),
            token_usage.get("output_tokens", estimated_output),
        )
    return estimated_input, estimated_output


def _circuit_breaker_key(model_config: ModelConfig) -> str:
    """Build a model-specific circuit breaker key from *model_config*."""
    provider_name = model_config.provider
    model_name = model_config.name
    bot_name = getattr(model_config, "bot_name", "")
    if provider_name == "poe" and bot_name:
        return f"{provider_name}:{bot_name}"
    if provider_name == "openrouter":
        return f"openrouter:{getattr(model_config, 'model_id', model_name)}"
    return f"{provider_name}:{model_name}"


# ---------------------------------------------------------------------------
# Unified execution guard
# ---------------------------------------------------------------------------


@dataclass
class _GuardContext:
    """State yielded by ``_managed_execution`` for callers to use."""

    model_config: ModelConfig
    model_name: str
    provider_name: str
    request: ProviderRequest
    token_usage: dict[str, Any] | None = field(default=None, repr=False)


@asynccontextmanager
async def _managed_execution(
    model_config: ModelConfig,
    messages: list[dict[str, str]],
    ctx: CouncilContext,
    system_message: str | None = None,
    *,
    suppress_provider_flags: bool = False,
) -> AsyncGenerator[_GuardContext, None]:
    """Context manager for budget reservation and circuit-breaker recording.

    Handles:
    - Token estimation and budget reservation (before yield)
    - Budget commit on normal exit (caller must set ``guard.token_usage``)
    - Budget release on cancellation or exception
    - Circuit-breaker success/failure recording

    Raises ``BudgetExceededError`` before yielding if budget is exhausted.
    """
    model_config = coerce_model_config(model_config)
    model_name = model_config.name
    provider_name = model_config.provider

    cb = ctx.get_circuit_breaker(_circuit_breaker_key(model_config))

    # Budget reservation
    est_in = 0
    est_out = 0
    reserved = False
    if ctx.budget_guard is not None:
        est_in, est_out = _estimate_request_tokens(messages, system_message)
        await ctx.budget_guard.areserve(est_in, est_out, model_name)
        reserved = True

    request = ProviderRequest(
        messages=messages,
        system_message=system_message,
        suppress_provider_flags=suppress_provider_flags,
    )

    guard = _GuardContext(
        model_config=model_config,
        model_name=model_name,
        provider_name=provider_name,
        request=request,
    )

    try:
        yield guard
        # Normal exit — success
        cb.record_success()
        if reserved:
            actual_in, actual_out = _actual_or_estimated(guard.token_usage, est_in, est_out)
            await ctx.budget_guard.acommit(actual_in, actual_out, model_name, est_in, est_out)
    except asyncio.CancelledError:
        if reserved:
            await ctx.budget_guard.arelease(est_in, est_out, model_name)
        raise
    except Exception:
        cb.record_failure()
        if reserved:
            await ctx.budget_guard.arelease(est_in, est_out, model_name)
        raise


# ---------------------------------------------------------------------------
# Public query / stream functions
# ---------------------------------------------------------------------------


async def query_model(
    model_config: ModelConfig,
    messages: list[dict[str, str]],
    ctx: CouncilContext,
    system_message: str | None = None,
    *,
    suppress_provider_flags: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Query a model through its provider, with timeout.

    Returns:
        Tuple of (response dict with 'content' key or None, token usage dict or None)
    """
    model_config = coerce_model_config(model_config)
    model_name = model_config.name

    cb = ctx.get_circuit_breaker(_circuit_breaker_key(model_config))
    if cb.is_open:
        logger.warning(f"Circuit breaker open for {_circuit_breaker_key(model_config)}, skipping {model_name}")
        return None, None

    try:
        async with _managed_execution(
            model_config, messages, ctx, system_message, suppress_provider_flags=suppress_provider_flags
        ) as guard:
            provider = ctx.get_provider(guard.provider_name)
            sem = ctx.get_semaphore()
            async with sem:
                result = await asyncio.wait_for(
                    provider.query("", guard.model_config, MODEL_TIMEOUT, request=guard.request),
                    timeout=MODEL_TIMEOUT,
                )
            content, token_usage = result
            guard.token_usage = token_usage
            return {"content": content}, token_usage
    except BudgetExceededError:
        logger.warning(f"Budget exceeded, skipping {model_name}")
        return None, {"budget_exceeded": True}
    except asyncio.TimeoutError:
        logger.warning(f"Timeout querying {model_name} after {MODEL_TIMEOUT}s")
        return None, None
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Error querying {model_name}: {e}")
        return None, None


async def stream_model(
    model_config: ModelConfig,
    messages: list[dict[str, str]],
    ctx: CouncilContext,
    on_chunk: Callable[[str], Awaitable[None]] | None = None,
    system_message: str | None = None,
    *,
    suppress_provider_flags: bool = False,
) -> tuple[str, dict | None]:
    """Stream a model response, falling back to query_model on error.

    Returns:
        Tuple of (accumulated text, token usage dict or None).
    """
    model_config = coerce_model_config(model_config)
    model_name = model_config.name

    cb = ctx.get_circuit_breaker(_circuit_breaker_key(model_config))
    if cb.is_open:
        logger.warning(f"Circuit breaker open for {_circuit_breaker_key(model_config)}, skipping {model_name}")
        return "", None

    accumulated = ""
    try:
        async with _managed_execution(
            model_config, messages, ctx, system_message, suppress_provider_flags=suppress_provider_flags
        ) as guard:
            provider = ctx.get_provider(guard.provider_name)

            if isinstance(provider, StreamingProvider):
                stream_result = provider.astream("", guard.model_config, MODEL_TIMEOUT, request=guard.request)
            else:
                stream_result = fallback_astream(
                    provider, "", guard.model_config, MODEL_TIMEOUT, request=guard.request
                )
            async for chunk in stream_result:
                accumulated += chunk
                if on_chunk is not None:
                    await on_chunk(chunk)

            guard.token_usage = stream_result.usage
            return accumulated, stream_result.usage
    except BudgetExceededError:
        logger.warning(f"Budget exceeded, skipping {model_name}")
        return "", None
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"Streaming error for {model_name}: {e}, falling back to query_model")

        # Notify the caller that streaming was interrupted before falling back
        if len(accumulated) > 0 and on_chunk is not None:
            await on_chunk("\n\n[Streaming interrupted, regenerating...]\n\n")

        # Fall back to query_model (manages its own budget lifecycle)
        result, usage = await query_model(model_config, messages, ctx, system_message)
        if result is not None:
            return result.get("content", ""), usage
        return "", None


async def query_models_parallel(
    model_configs: list[ModelConfig],
    messages: list[dict[str, str]],
    ctx: CouncilContext,
    *,
    system_message: str | None = None,
    min_responses: int | None = None,
    soft_timeout: float | None = None,
) -> tuple[dict[str, dict[str, Any] | None], dict[str, dict[str, Any] | None]]:
    """Query multiple models in parallel via hybrid providers.

    Args:
        model_configs: List of typed ModelConfig objects
        messages: Messages to send to each model
        ctx: CouncilContext providing providers, circuit breakers, semaphore, progress
        system_message: Optional system message for injection hardening
        min_responses: Minimum responses needed to proceed (default: all models)
        soft_timeout: Time to wait before proceeding with min_responses (default: SOFT_TIMEOUT=300s)

    Returns:
        Tuple of (response dict, token usage dict) where both map model name to values
    """
    model_configs = [coerce_model_config(c) for c in model_configs]
    progress = ctx.progress

    # Default: wait for all models (was min(2, n-1) which dropped slow models)
    if min_responses is None:
        min_responses = len(model_configs)
    # Default soft_timeout from provider config
    if soft_timeout is None:
        soft_timeout = float(SOFT_TIMEOUT)

    async def safe_query(config: ModelConfig) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
        name = config.name
        start = time.time()
        if progress:
            await progress.update_model(name, ModelStatus.QUERYING)
        try:
            result, token_usage = await query_model(config, messages, ctx, system_message)
            elapsed = time.time() - start
            if progress:
                if result:
                    status = ModelStatus.DONE
                elif isinstance(token_usage, dict) and token_usage.get("budget_exceeded"):
                    status = ModelStatus.BUDGET
                else:
                    status = ModelStatus.FAILED
                await progress.update_model(name, status, elapsed)
            return (name, result, token_usage)
        except Exception as e:
            elapsed = time.time() - start
            if progress:
                await progress.update_model(name, ModelStatus.FAILED, elapsed)
            logger.error(f"Error querying {name}: {e}")
            return (name, None, None)

    # Create tasks for all models
    tasks = [asyncio.create_task(safe_query(config)) for config in model_configs]

    # Track which tasks are done
    completed_results: dict[str, dict[str, Any] | None] = {}
    token_usages: dict[str, dict[str, Any] | None] = {}
    pending_tasks = set(tasks)

    start_time = time.time()

    # Map tasks to model names for safe lookup on cancellation
    task_to_model: dict[asyncio.Task, str] = {
        task: config.name for task, config in zip(tasks, model_configs, strict=True)
    }

    while pending_tasks:
        elapsed = time.time() - start_time
        remaining_time = soft_timeout - elapsed

        if remaining_time <= 0:
            # Soft timeout expired — check if we have enough responses
            successful_responses = sum(1 for r in completed_results.values() if r is not None)
            if successful_responses >= min_responses:
                # Cancel remaining and break
                skipped_models = [task_to_model.get(t, "unknown") for t in pending_tasks]
                for task in pending_tasks:
                    task.cancel()
                if skipped_models:
                    logger.warning(
                        f"Soft timeout ({soft_timeout}s) reached with {successful_responses} responses. "
                        f"Skipping models: {', '.join(skipped_models)}"
                    )
                for model_name in skipped_models:
                    completed_results[model_name] = None
                    token_usages[model_name] = None
                break
            else:
                # Not enough responses yet — wait for next completion without timeout
                remaining_time = None

        done, pending = await asyncio.wait(
            pending_tasks,
            timeout=remaining_time,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Process completed tasks
        for task in done:
            name, result, token_usage = await task
            completed_results[name] = result
            token_usages[name] = token_usage
            pending_tasks.discard(task)

        if not pending_tasks:
            break

    # Wait for any remaining cancelled tasks to settle
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    return completed_results, token_usages
