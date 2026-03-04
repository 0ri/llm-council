"""Stage 3: Chairman synthesis with optional streaming."""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..context import CouncilContext
from ..models import AggregateRanking, ModelConfig, Stage1Result, Stage3Result, coerce_model_config
from ..progress import ModelStatus
from ..prompts import SYNTHESIS_PROMPT_TEMPLATE, SYNTHESIS_SYSTEM_MESSAGE_TEMPLATE, resolve_template
from ..security import build_manipulation_resistance_msg, format_anonymized_responses, sanitize_model_output
from .execution import query_model, stream_model

logger = logging.getLogger("llm-council")


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
    synthesis_user_template = resolve_template(prompt_config, "synthesis_user", SYNTHESIS_PROMPT_TEMPLATE)
    chairman_prompt = build_synthesis_prompt(
        user_query,
        response_tuples,
        label_to_model,
        aggregate_rankings,
        custom_template=synthesis_user_template,
    )

    # System message for injection hardening (supports custom templates via config)
    base_resistance_msg = build_manipulation_resistance_msg()
    synthesis_sys_template = resolve_template(prompt_config, "synthesis_system", SYNTHESIS_SYSTEM_MESSAGE_TEMPLATE)
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
