"""Stage execution logic for the 3-stage council deliberation pipeline.

Implements ``stage1_collect_responses`` (parallel queries with caching),
``stage2_collect_rankings`` (anonymized peer ranking with self-exclusion
and retry), and ``stage3_synthesize_final`` (chairman synthesis with
optional streaming). Also provides ``query_model`` and ``stream_model``.
"""

from __future__ import annotations

import asyncio
import logging
import random
import secrets
import sqlite3
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..budget import BudgetExceededError
from ..context import CouncilContext
from ..cost import estimate_tokens
from ..models import (
    AggregateRanking,
    ModelConfig,
    Stage1Result,
    Stage2Result,
    Stage3Result,
    coerce_model_config,
    generate_letter_labels,
    get_model_identifier,
)
from ..parsing import parse_ranking_from_text
from ..progress import ModelStatus
from ..prompts import (
    RANKING_PROMPT_TEMPLATE,
    RANKING_SYSTEM_MESSAGE_TEMPLATE,
    SYNTHESIS_PROMPT_TEMPLATE,
    SYNTHESIS_SYSTEM_MESSAGE_TEMPLATE,
)
from ..providers import MODEL_TIMEOUT, ProviderRequest
from ..security import build_manipulation_resistance_msg, format_anonymized_responses, sanitize_model_output

# Re-export execution helpers so existing imports continue to work
from .execution import (  # noqa: F401
    _circuit_breaker_key,
    query_model,
    query_models_parallel,
    stream_model,
)

logger = logging.getLogger("llm-council")


async def stage1_collect_responses(
    user_query: str,
    council_models: list[ModelConfig],
    ctx: CouncilContext,
    *,
    soft_timeout: float | None = None,
    min_responses: int | None = None,
) -> tuple[list[Stage1Result], dict[str, dict[str, Any] | None]]:
    """Stage 1: Collect individual responses from all council models.

    Checks the local cache first; only queries models with cache misses.

    Returns:
        Tuple of (stage1_results, token_usages)
    """
    council_models = [coerce_model_config(c) for c in council_models]
    progress = ctx.progress

    if progress:
        model_names = [m.name for m in council_models]
        await progress.start_stage(1, "Collecting responses", model_names)

    messages = [{"role": "user", "content": user_query}]

    # Check cache for each model
    cached_responses: dict[str, dict[str, Any] | None] = {}
    cached_usages: dict[str, dict[str, Any] | None] = {}
    models_to_query: list[ModelConfig] = []

    for model_config in council_models:
        model_name = model_config.name
        model_id = get_model_identifier(model_config)
        if ctx.cache is not None:
            try:
                hit = await ctx.cache.aget(user_query, model_name, model_id, model_config)
            except (sqlite3.OperationalError, Exception):
                logger.warning(f"Cache read failed for {model_name}, querying model instead")
                hit = None
            if hit is not None:
                response_text, token_usage = hit
                cached_responses[model_name] = {"content": response_text}
                cached_usages[model_name] = token_usage
                logger.info(f"Cache hit for {model_name}")
                if progress:
                    await progress.update_model(model_name, ModelStatus.DONE, 0.0)
                continue
        models_to_query.append(model_config)

    # Query uncached models in parallel
    if models_to_query:
        responses, token_usages = await query_models_parallel(
            models_to_query,
            messages,
            ctx,
            soft_timeout=soft_timeout,
            min_responses=min_responses,
        )
    else:
        responses, token_usages = {}, {}

    # Store new responses in cache
    if ctx.cache is not None:
        for model_config in models_to_query:
            model_name = model_config.name
            model_id = get_model_identifier(model_config)
            response = responses.get(model_name)
            if response is not None:
                try:
                    await ctx.cache.aput(
                        user_query,
                        model_name,
                        model_id,
                        response.get("content", ""),
                        token_usages.get(model_name),
                        model_config,
                    )
                except (sqlite3.OperationalError, Exception):
                    logger.warning(f"Cache write failed for {model_name}, continuing without cache")

    # Merge cached + fresh responses
    all_responses = {**cached_responses, **responses}
    all_usages = {**cached_usages, **token_usages}

    # Format results in original config order (not completion order)
    stage1_results: list[Stage1Result] = []
    for model_config in council_models:
        model_name = model_config.name
        response = all_responses.get(model_name)
        if response is not None:
            content = response.get("content", "")
            if not content or not content.strip():
                logger.warning(f"Filtering out empty response from {model_name}")
                continue
            stage1_results.append(Stage1Result(model=model_name, response=content))

    if progress:
        cache_hits = len(cached_responses)
        suffix = f" ({cache_hits} cached)" if cache_hits else ""
        await progress.complete_stage(f"{len(stage1_results)}/{len(council_models)} responses{suffix}")

    return stage1_results, all_usages


def build_ranking_prompt(
    question: str,
    responses: list[tuple[str, str]],
    response_order: list[int] | None = None,
    *,
    custom_template: str | None = None,
) -> str:
    """Construct the prompt for peer ranking with anonymized responses.

    Args:
        question: The original user question
        responses: List of (model_name, response_text) tuples
        response_order: Optional list of indices for response ordering.
                       If None, uses original order.

    Returns:
        The formatted ranking prompt
    """
    # Generate a random nonce for XML delimiters to prevent fence-breaking
    nonce = secrets.token_hex(8)

    # Apply custom ordering if specified
    if response_order is not None:
        ordered_responses = [responses[i] for i in response_order]
    else:
        ordered_responses = responses

    # Build responses with randomized XML delimiters for injection hardening
    responses_text = format_anonymized_responses(ordered_responses, nonce=nonce)

    template = custom_template or RANKING_PROMPT_TEMPLATE
    ranking_prompt = template.format(
        question=question,
        nonce=nonce,
        responses_text=responses_text,
    )

    return ranking_prompt


async def _get_ranking(
    model_config: ModelConfig,
    messages: list[dict[str, str]],
    ranker_name: str,
    num_responses: int,
    ctx: CouncilContext,
    system_message: str,
    *,
    suppress_provider_flags: bool = False,
) -> tuple[Stage2Result | None, dict[str, Any] | None]:
    """Execute a single ranking query and parse the result."""
    response, token_usage = await query_model(
        model_config, messages, ctx, system_message, suppress_provider_flags=suppress_provider_flags
    )
    if response is not None:
        full_text = response.get("content", "")
        parsed_ranking, is_valid = parse_ranking_from_text(full_text, num_responses=num_responses)
        return Stage2Result(
            model=ranker_name,
            ranking=full_text,
            parsed_ranking=parsed_ranking,
            is_valid_ballot=is_valid,
        ), token_usage
    return None, token_usage


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: list[Stage1Result],
    council_models: list[ModelConfig],
    ctx: CouncilContext,
    stage2_max_retries: int | None = None,
) -> tuple[list[Stage2Result], dict[str, dict[str, str]], dict[str, dict[str, Any] | None]]:
    """Stage 2: Each model ranks the anonymized responses.

    Uses injection hardening: fenced blocks + system message + structured JSON output.
    Implements self-exclusion and response order randomization.
    Invalid ballots are retried up to stage2_max_retries times.

    Args:
        user_query: The user's question
        stage1_results: List of Stage1Result objects
        council_models: List of council model configs
        ctx: CouncilContext providing providers, circuit breakers, semaphore, progress
        stage2_max_retries: Maximum retries for invalid ballots (default: ctx.stage2_max_retries)

    Returns:
        Tuple of (stage2_results, per_ranker_label_mappings, token_usages)
        where per_ranker_label_mappings is {ranker_name: {label: model_name}}
    """
    council_models = [coerce_model_config(c) for c in council_models]
    if stage2_max_retries is None:
        stage2_max_retries = ctx.stage2_max_retries

    rng = random.Random()

    progress = ctx.progress

    if progress:
        model_names = [m.name for m in council_models]
        await progress.start_stage(2, f"Peer ranking ({len(stage1_results)} responses)", model_names)

    # System message for injection hardening (supports custom templates via config)
    base_resistance_msg = build_manipulation_resistance_msg()
    prompt_config = getattr(ctx, "prompt_config", None)
    ranking_sys_template = (
        prompt_config.resolve("ranking_system") if prompt_config else None
    ) or RANKING_SYSTEM_MESSAGE_TEMPLATE
    system_message = ranking_sys_template.format(manipulation_resistance_msg=base_resistance_msg)

    # Build responses tuples for prompt construction, sanitizing outputs
    response_tuples = [(result.model, sanitize_model_output(result.response)) for result in stage1_results]

    # Store per-ranker label mappings for proper aggregation
    per_ranker_label_mappings: dict[str, dict[str, str]] = {}

    # Store per-ranker task context for reuse during retries
    ranker_task_context: dict[str, dict[str, Any]] = {}

    # Prepare ranking tasks for each model
    ranking_tasks = []
    ranking_task_names: list[str] = []  # Parallel list tracking ranker name per task

    for ranker_model in council_models:
        ranker_name = ranker_model.name

        # Exclude self from ranking (self-ranking exclusion)
        filtered_indices = []
        filtered_responses = []
        for i, result in enumerate(stage1_results):
            if result.model != ranker_name:
                filtered_indices.append(i)
                filtered_responses.append(response_tuples[i])

        # Skip if model has no other responses to rank
        if not filtered_responses:
            continue

        # Randomize order for this ranker (position bias mitigation)
        response_order = list(range(len(filtered_responses)))
        rng.shuffle(response_order)

        # Create labels for this ranker's view
        labels = generate_letter_labels(len(filtered_responses))

        # Create mapping from label to model name for this ranker
        label_to_model = {}
        for label_idx, order_idx in enumerate(response_order):
            original_idx = filtered_indices[order_idx]
            result = stage1_results[original_idx]
            label_to_model[f"Response {labels[label_idx]}"] = result.model

        per_ranker_label_mappings[ranker_name] = label_to_model

        # Build the ranking prompt with randomized order
        custom_ranking_user = prompt_config.resolve("ranking_user") if prompt_config else None
        ranking_prompt = build_ranking_prompt(
            user_query, filtered_responses, response_order, custom_template=custom_ranking_user
        )

        messages = [{"role": "user", "content": ranking_prompt}]

        # Store context for potential retries
        ranker_task_context[ranker_name] = {
            "model_config": ranker_model,
            "messages": messages,
            "num_responses": len(filtered_responses),
        }

        # Suppress provider flags (web_search, reasoning_effort) for ranking queries
        ranking_tasks.append(
            _get_ranking(
                ranker_model,
                messages,
                ranker_name,
                len(filtered_responses),
                ctx,
                system_message,
                suppress_provider_flags=True,
            )
        )
        ranking_task_names.append(ranker_name)

    # Execute all ranking tasks in parallel
    if progress:
        for model_name in [m.name for m in council_models]:
            await progress.update_model(model_name, ModelStatus.QUERYING)

    start_times = {m.name: time.time() for m in council_models}
    ranking_results = await asyncio.gather(*ranking_tasks)

    # Format results — iterate ALL results to ensure failed models get status updates
    stage2_results: list[Stage2Result] = []
    token_usages: dict[str, dict[str, Any] | None] = {}

    for i, result in enumerate(ranking_results):
        ranker_name = ranking_task_names[i]
        if result is not None:
            ranking_data, token_usage = result
            if ranking_data is not None:
                stage2_results.append(ranking_data)
                token_usages[ranking_data.model] = token_usage
                if progress:
                    elapsed = time.time() - start_times.get(ranking_data.model, time.time())
                    status = ModelStatus.DONE if ranking_data.is_valid_ballot else ModelStatus.FAILED
                    await progress.update_model(ranking_data.model, status, elapsed)
            else:
                # ranking_data is None — model failed to produce a ranking
                if progress:
                    elapsed = time.time() - start_times.get(ranker_name, time.time())
                    if isinstance(token_usage, dict) and token_usage.get("budget_exceeded"):
                        await progress.update_model(ranker_name, ModelStatus.BUDGET, elapsed)
                    else:
                        await progress.update_model(ranker_name, ModelStatus.FAILED, elapsed)
        else:
            # result is None — handle defensively
            if progress:
                elapsed = time.time() - start_times.get(ranker_name, time.time())
                await progress.update_model(ranker_name, ModelStatus.FAILED, elapsed)

    # Retry loop for invalid ballots
    for retry_round in range(stage2_max_retries):
        invalid_indices = [i for i, r in enumerate(stage2_results) if not r.is_valid_ballot]
        if not invalid_indices:
            break

        logger.info(
            f"Stage 2 retry round {retry_round + 1}/{stage2_max_retries}: "
            f"retrying {len(invalid_indices)} invalid ballot(s)"
        )

        retry_tasks = []
        retry_indices: list[int] = []

        for idx in invalid_indices:
            ranker_name = stage2_results[idx].model
            task_ctx = ranker_task_context.get(ranker_name)
            if task_ctx is None:
                continue

            logger.info(f"Retrying ranking for {ranker_name}")
            if progress:
                await progress.update_model(ranker_name, ModelStatus.QUERYING)

            retry_tasks.append(
                _get_ranking(
                    task_ctx["model_config"],
                    task_ctx["messages"],
                    ranker_name,
                    task_ctx["num_responses"],
                    ctx,
                    system_message,
                    suppress_provider_flags=True,
                )
            )
            retry_indices.append(idx)

        if not retry_tasks:
            break

        retry_results = await asyncio.gather(*retry_tasks)

        for retry_idx, retry_result in zip(retry_indices, retry_results, strict=True):
            if retry_result is not None:
                ranking_data, token_usage = retry_result
                if ranking_data is not None:
                    if ranking_data.is_valid_ballot:
                        logger.info(f"Retry succeeded for {ranking_data.model}")
                    else:
                        logger.info(f"Retry still invalid for {ranking_data.model}")
                    stage2_results[retry_idx] = ranking_data
                    token_usages[ranking_data.model] = token_usage
                    if progress:
                        elapsed = time.time() - start_times.get(ranking_data.model, time.time())
                        status = ModelStatus.DONE if ranking_data.is_valid_ballot else ModelStatus.FAILED
                        await progress.update_model(ranking_data.model, status, elapsed)

    if progress:
        valid = sum(1 for r in stage2_results if r.is_valid_ballot)
        await progress.complete_stage(f"{len(stage2_results)} rankings, {valid} valid ballots")

    return stage2_results, per_ranker_label_mappings, token_usages


def build_synthesis_prompt(
    question: str,
    responses: list[tuple[str, str]],
    rankings: dict[str, str],
    aggregate_rankings: list[AggregateRanking],
    *,
    custom_template: str | None = None,
) -> str:
    """Construct the chairman synthesis prompt with anonymized context.

    Args:
        question: The original user question
        responses: List of (model_name, response_text) tuples
        rankings: Mapping of labels to model names
        aggregate_rankings: Sorted AggregateRanking objects

    Returns:
        The formatted synthesis prompt
    """
    # Generate a random nonce for XML delimiters to prevent fence-breaking
    nonce = secrets.token_hex(8)

    # Build anonymized context for chairman with randomized XML delimiters
    stage1_text = format_anonymized_responses(responses, nonce=nonce)

    # Create reverse mapping: model name -> label
    model_to_label = {v: k for k, v in rankings.items()}

    # Summarize rankings anonymously (which responses were ranked highest)
    ranking_summary_lines = []
    for rank_info in aggregate_rankings:
        label = model_to_label.get(rank_info.model, "Unknown")
        ranking_summary_lines.append(f"- {label}: Average position {rank_info.average_rank}")
    ranking_summary = "\n".join(ranking_summary_lines)

    template = custom_template or SYNTHESIS_PROMPT_TEMPLATE
    chairman_prompt = template.format(
        question=question,
        stage1_text=stage1_text,
        ranking_summary=ranking_summary,
    )

    return chairman_prompt


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: list[Stage1Result],
    label_to_model: dict[str, str],
    aggregate_rankings: list[AggregateRanking],
    chairman_config: ModelConfig,
    ctx: CouncilContext,
    stream: bool = False,
    on_chunk: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[Stage3Result, dict[str, Any] | None]:
    """Stage 3: Chairman synthesizes final response.

    Uses anonymized labels to prevent bias toward specific models.

    Args:
        user_query: The user's question
        stage1_results: List of Stage1Result objects
        label_to_model: Label to model mapping
        aggregate_rankings: List of AggregateRanking objects
        chairman_config: Typed ModelConfig for the chairman model
        ctx: CouncilContext providing providers, circuit breakers, semaphore, progress
        stream: If True, use streaming for the chairman query.
        on_chunk: Optional async callback invoked for each streamed text chunk.

    Returns:
        Tuple of (result, token usage dict or None)
    """
    chairman_config = coerce_model_config(chairman_config)
    progress = ctx.progress
    chairman_name = chairman_config.name

    if progress:
        await progress.start_stage(3, f"Chairman ({chairman_name}) synthesizing", [chairman_name])
        await progress.update_model(chairman_name, ModelStatus.QUERYING)

    # Build responses tuples for prompt construction, sanitizing outputs
    response_tuples = [(result.model, sanitize_model_output(result.response)) for result in stage1_results]

    # Build the synthesis prompt (supports custom templates via config)
    prompt_config = getattr(ctx, "prompt_config", None)
    custom_synthesis_user = prompt_config.resolve("synthesis_user") if prompt_config else None
    chairman_prompt = build_synthesis_prompt(
        user_query,
        response_tuples,
        label_to_model,
        aggregate_rankings,
        custom_template=custom_synthesis_user,
    )

    # System message for injection hardening (supports custom templates via config)
    base_resistance_msg = build_manipulation_resistance_msg()
    synthesis_sys_template = (
        prompt_config.resolve("synthesis_system") if prompt_config else None
    ) or SYNTHESIS_SYSTEM_MESSAGE_TEMPLATE
    system_message = synthesis_sys_template.format(manipulation_resistance_msg=base_resistance_msg)

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model with system message (suppress provider flags for synthesis)
    start = time.time()

    if stream:
        if progress:
            await progress.pause_live()
        text, token_usage = await stream_model(
            chairman_config,
            messages,
            ctx,
            on_chunk=on_chunk,
            system_message=system_message,
            suppress_provider_flags=True,
        )
        elapsed = time.time() - start

        if progress:
            status = ModelStatus.DONE if text else ModelStatus.FAILED
            await progress.update_model(chairman_name, status, elapsed)
            await progress.complete_stage()

        return Stage3Result(model=chairman_name, response=text), token_usage
    else:
        response, token_usage = await query_model(
            chairman_config, messages, ctx, system_message, suppress_provider_flags=True
        )
        elapsed = time.time() - start

        if progress:
            status = ModelStatus.DONE if response else ModelStatus.FAILED
            await progress.update_model(chairman_name, status, elapsed)
            await progress.complete_stage()

        if response is None:
            return Stage3Result(model=chairman_name, response="Error: Unable to generate final synthesis."), None

        return Stage3Result(model=chairman_name, response=response.get("content", "")), token_usage
