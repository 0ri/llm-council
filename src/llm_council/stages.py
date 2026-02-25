"""3-stage deliberation logic for LLM Council."""

from __future__ import annotations

import asyncio
import logging
import random
import secrets
import time
from typing import Any

from .budget import BudgetExceededError
from .context import CouncilContext
from .cost import estimate_tokens
from .models import AggregateRanking, Stage1Result, Stage2Result, Stage3Result
from .parsing import parse_ranking_from_text
from .progress import ModelStatus
from .prompts import (
    RANKING_PROMPT_TEMPLATE,
    RANKING_SYSTEM_MESSAGE_TEMPLATE,
    SYNTHESIS_PROMPT_TEMPLATE,
    SYNTHESIS_SYSTEM_MESSAGE_TEMPLATE,
)
from .providers import MODEL_TIMEOUT
from .security import build_manipulation_resistance_msg, format_anonymized_responses, sanitize_model_output

logger = logging.getLogger("llm-council")


async def query_model(
    model_config: dict[str, Any],
    messages: list[dict[str, str]],
    ctx: CouncilContext,
    system_message: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Query a model through its provider, with timeout.

    Args:
        model_config: Dict with 'provider', 'model_id' or 'bot_name', and optional:
            - budget_tokens: int (Bedrock extended thinking)
            - web_search: bool (Poe web search)
            - reasoning_effort: str (Poe reasoning level)
        messages: List of message dicts
        ctx: CouncilContext providing providers, circuit breakers, semaphore
        system_message: Optional system message for injection hardening

    Returns:
        Tuple of (response dict with 'content' key or None, token usage dict or None)
    """
    provider_name = model_config.get("provider", "poe")
    model_name = model_config.get("name", "unknown")
    bot_name = model_config.get("bot_name", "")

    # Build model-specific circuit breaker key
    if provider_name == "poe" and bot_name:
        cb_key = f"{provider_name}:{bot_name}"
    elif provider_name == "openrouter":
        cb_key = f"openrouter:{model_config.get('model_id', model_name)}"
    else:
        cb_key = f"{provider_name}:{model_name}"

    # Check circuit breaker
    cb = ctx.get_circuit_breaker(cb_key)
    if cb.is_open:
        logger.warning(f"Circuit breaker open for {cb_key}, skipping {model_name}")
        return None, None

    # Budget preflight (no mutation — only commit after success)
    if ctx.budget_guard is not None:
        input_text = " ".join(m.get("content", "") for m in messages)
        if system_message:
            input_text += " " + system_message
        estimated_input = estimate_tokens(input_text)
        estimated_output = 2000  # conservative estimate for preflight
        try:
            ctx.budget_guard.can_afford(estimated_input, estimated_output, model_name)
        except BudgetExceededError:
            logger.warning(f"Budget exceeded, skipping {model_name}")
            return None, None

    # Add messages and system_message to config for provider access
    config_with_context = model_config.copy()
    config_with_context["_messages"] = messages
    config_with_context["_system_message"] = system_message

    try:
        provider = ctx.get_provider(provider_name)
        sem = ctx.get_semaphore()
        async with sem:
            result = await asyncio.wait_for(
                provider.query("", config_with_context, MODEL_TIMEOUT),
                timeout=MODEL_TIMEOUT,
            )
        cb.record_success()

        # Extract token usage if available (tuple response from provider)
        content, token_usage = result

        # Commit actual usage to budget (only on success)
        if ctx.budget_guard is not None:
            actual_in = token_usage.get("input_tokens", estimated_input) if token_usage else estimated_input
            actual_out = token_usage.get("output_tokens", estimated_output) if token_usage else estimated_output
            ctx.budget_guard.commit(actual_in, actual_out, model_name)

        return {"content": content}, token_usage
    except asyncio.TimeoutError:
        cb.record_failure()
        logger.warning(f"Timeout querying {model_name} after {MODEL_TIMEOUT}s")
        return None, None
    except Exception as e:
        cb.record_failure()
        logger.error(f"Error querying {model_name}: {e}")
        return None, None


async def query_models_parallel(
    model_configs: list[dict[str, Any]],
    messages: list[dict[str, str]],
    ctx: CouncilContext,
    *,
    system_message: str | None = None,
    min_responses: int | None = None,
    soft_timeout: float = 120.0,
) -> tuple[dict[str, dict[str, Any] | None], dict[str, dict[str, Any] | None]]:
    """Query multiple models in parallel via hybrid providers.

    Args:
        model_configs: List of model config dicts
        messages: Messages to send to each model
        ctx: CouncilContext providing providers, circuit breakers, semaphore, progress
        system_message: Optional system message for injection hardening
        min_responses: Minimum responses needed to proceed (default: len(models)-1 or 2)
        soft_timeout: Time to wait before proceeding with min_responses (default: 120s)

    Returns:
        Tuple of (response dict, token usage dict) where both map model name to values
    """
    progress = ctx.progress

    # Calculate default min_responses if not provided
    if min_responses is None:
        min_responses = min(2, max(1, len(model_configs) - 1))

    async def safe_query(config: dict[str, Any]) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
        name = config.get("name", "unknown")
        start = time.time()
        if progress:
            await progress.update_model(name, ModelStatus.QUERYING)
        try:
            result, token_usage = await query_model(config, messages, ctx, system_message)
            elapsed = time.time() - start
            if progress:
                await progress.update_model(
                    name,
                    ModelStatus.DONE if result else ModelStatus.FAILED,
                    elapsed,
                )
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

    while pending_tasks:
        # Wait for either soft_timeout or some tasks to complete
        remaining_time = max(0, soft_timeout - (time.time() - start_time))

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

        # Check if we've reached soft timeout and have minimum responses
        elapsed = time.time() - start_time
        successful_responses = sum(1 for r in completed_results.values() if r is not None)

        if elapsed >= soft_timeout and successful_responses >= min_responses:
            # Cancel remaining tasks
            skipped_models = []
            for task in pending_tasks:
                task.cancel()
                # Try to get the model name from the task
                try:
                    # Extract model name from the config that was passed to safe_query
                    # This is a bit hacky but works for logging purposes
                    config_idx = tasks.index(task)
                    model_name = model_configs[config_idx].get("name", "unknown")
                    skipped_models.append(model_name)
                except (ValueError, IndexError):
                    skipped_models.append("unknown")

            if skipped_models:
                logger.warning(
                    f"Soft timeout ({soft_timeout}s) reached with {successful_responses} responses. "
                    f"Skipping models: {', '.join(skipped_models)}"
                )

            # Add None entries for cancelled tasks
            for model_name in skipped_models:
                completed_results[model_name] = None
                token_usages[model_name] = None

            break

        # If no more pending tasks, we're done
        if not pending_tasks:
            break

    # Wait for any remaining tasks to complete (shouldn't happen but be safe)
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    return completed_results, token_usages


async def stage1_collect_responses(
    user_query: str,
    council_models: list[dict[str, Any]],
    ctx: CouncilContext,
) -> tuple[list[Stage1Result], dict[str, dict[str, Any] | None]]:
    """Stage 1: Collect individual responses from all council models.

    Returns:
        Tuple of (stage1_results, token_usages)
    """
    progress = ctx.progress

    if progress:
        model_names = [m.get("name", "unknown") for m in council_models]
        await progress.start_stage(1, "Collecting responses", model_names)

    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel
    responses, token_usages = await query_models_parallel(council_models, messages, ctx)

    # Format results
    stage1_results: list[Stage1Result] = []
    for model_name, response in responses.items():
        if response is not None:
            stage1_results.append(Stage1Result(model=model_name, response=response.get("content", "")))

    if progress:
        await progress.complete_stage(f"{len(stage1_results)}/{len(council_models)} responses")

    return stage1_results, token_usages


def build_ranking_prompt(
    question: str,
    responses: list[tuple[str, str]],
    labels: list[str],
    response_order: list[int] | None = None,
) -> str:
    """Construct the prompt for peer ranking with anonymized responses.

    Args:
        question: The original user question
        responses: List of (model_name, response_text) tuples
        labels: List of single-character labels (A, B, C...)
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

    ranking_prompt = RANKING_PROMPT_TEMPLATE.format(
        question=question,
        nonce=nonce,
        responses_text=responses_text,
    )

    return ranking_prompt


async def _get_ranking(
    model_config: dict[str, Any],
    messages: list[dict[str, str]],
    ranker_name: str,
    num_responses: int,
    ctx: CouncilContext,
    system_message: str,
) -> tuple[Stage2Result | None, dict[str, Any] | None]:
    """Execute a single ranking query and parse the result."""
    response, token_usage = await query_model(model_config, messages, ctx, system_message)
    if response is not None:
        full_text = response.get("content", "")
        parsed_ranking, is_valid = parse_ranking_from_text(full_text, num_responses=num_responses)
        return Stage2Result(
            model=ranker_name,
            ranking=full_text,
            parsed_ranking=parsed_ranking,
            is_valid_ballot=is_valid,
        ), token_usage
    return None, None


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: list[Stage1Result],
    council_models: list[dict[str, Any]],
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
    if stage2_max_retries is None:
        stage2_max_retries = ctx.stage2_max_retries

    progress = ctx.progress

    if progress:
        model_names = [m.get("name", "unknown") for m in council_models]
        await progress.start_stage(2, f"Peer ranking ({len(stage1_results)} responses)", model_names)

    # System message for injection hardening
    base_resistance_msg = build_manipulation_resistance_msg()
    system_message = RANKING_SYSTEM_MESSAGE_TEMPLATE.format(manipulation_resistance_msg=base_resistance_msg)

    # Build responses tuples for prompt construction, sanitizing outputs
    response_tuples = [(result.model, sanitize_model_output(result.response)) for result in stage1_results]

    # Store per-ranker label mappings for proper aggregation
    per_ranker_label_mappings: dict[str, dict[str, str]] = {}

    # Store per-ranker task context for reuse during retries
    ranker_task_context: dict[str, dict[str, Any]] = {}

    # Prepare ranking tasks for each model
    ranking_tasks = []

    for ranker_model in council_models:
        ranker_name = ranker_model.get("name", "unknown")

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
        random.shuffle(response_order)

        # Create labels for this ranker's view
        labels = [chr(65 + i) for i in range(len(filtered_responses))]  # A, B, C, ...

        # Create mapping from label to model name for this ranker
        label_to_model = {}
        for label_idx, order_idx in enumerate(response_order):
            original_idx = filtered_indices[order_idx]
            result = stage1_results[original_idx]
            label_to_model[f"Response {labels[label_idx]}"] = result.model

        per_ranker_label_mappings[ranker_name] = label_to_model

        # Build the ranking prompt with randomized order
        ranking_prompt = build_ranking_prompt(user_query, filtered_responses, labels, response_order)

        messages = [{"role": "user", "content": ranking_prompt}]

        # Store context for potential retries
        ranker_task_context[ranker_name] = {
            "model_config": ranker_model,
            "messages": messages,
            "num_responses": len(filtered_responses),
        }

        ranking_tasks.append(
            _get_ranking(ranker_model, messages, ranker_name, len(filtered_responses), ctx, system_message)
        )

    # Execute all ranking tasks in parallel
    if progress:
        for model_name in [m.get("name", "unknown") for m in council_models]:
            await progress.update_model(model_name, ModelStatus.QUERYING)

    start_times = {m.get("name", "unknown"): time.time() for m in council_models}
    ranking_results = await asyncio.gather(*ranking_tasks)

    # Format results
    stage2_results: list[Stage2Result] = []
    token_usages: dict[str, dict[str, Any] | None] = {}

    for result in ranking_results:
        if result is not None:
            ranking_data, token_usage = result
            if ranking_data is not None:
                stage2_results.append(ranking_data)
                token_usages[ranking_data.model] = token_usage
                if progress:
                    elapsed = time.time() - start_times[ranking_data.model]
                    status = ModelStatus.DONE if ranking_data.is_valid_ballot else ModelStatus.FAILED
                    await progress.update_model(ranking_data.model, status, elapsed)

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
    labels: list[str],
    aggregate_rankings: list[AggregateRanking],
) -> str:
    """Construct the chairman synthesis prompt with anonymized context.

    Args:
        question: The original user question
        responses: List of (model_name, response_text) tuples
        rankings: Mapping of labels to model names
        labels: List of response labels
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

    chairman_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        question=question,
        stage1_text=stage1_text,
        ranking_summary=ranking_summary,
    )

    return chairman_prompt


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: list[Stage1Result],
    stage2_results: list[Stage2Result],
    label_to_model: dict[str, str],
    aggregate_rankings: list[AggregateRanking],
    chairman_config: dict[str, Any],
    ctx: CouncilContext,
) -> tuple[Stage3Result, dict[str, Any] | None]:
    """Stage 3: Chairman synthesizes final response.

    Uses anonymized labels to prevent bias toward specific models.

    Args:
        user_query: The user's question
        stage1_results: List of Stage1Result objects
        stage2_results: List of Stage2Result objects
        label_to_model: Label to model mapping
        aggregate_rankings: List of AggregateRanking objects
        chairman_config: Chairman model config
        ctx: CouncilContext providing providers, circuit breakers, semaphore, progress

    Returns:
        Tuple of (result, token usage dict or None)
    """
    progress = ctx.progress
    chairman_name = chairman_config.get("name", "Chairman")

    if progress:
        await progress.start_stage(3, f"Chairman ({chairman_name}) synthesizing", [chairman_name])
        await progress.update_model(chairman_name, ModelStatus.QUERYING)

    # Build responses tuples and labels for prompt construction, sanitizing outputs
    response_tuples = [(result.model, sanitize_model_output(result.response)) for result in stage1_results]
    labels = [f"Response {chr(65 + i)}" for i in range(len(stage1_results))]

    # Build the synthesis prompt
    chairman_prompt = build_synthesis_prompt(
        user_query,
        response_tuples,
        label_to_model,
        labels,
        aggregate_rankings,
    )

    # System message for injection hardening
    base_resistance_msg = build_manipulation_resistance_msg()
    system_message = SYNTHESIS_SYSTEM_MESSAGE_TEMPLATE.format(manipulation_resistance_msg=base_resistance_msg)

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model with system message
    start = time.time()
    response, token_usage = await query_model(chairman_config, messages, ctx, system_message)
    elapsed = time.time() - start

    if progress:
        status = ModelStatus.DONE if response else ModelStatus.FAILED
        await progress.update_model(chairman_name, status, elapsed)
        await progress.complete_stage()

    if response is None:
        return Stage3Result(model=chairman_name, response="Error: Unable to generate final synthesis."), None

    return Stage3Result(model=chairman_name, response=response.get("content", "")), token_usage
