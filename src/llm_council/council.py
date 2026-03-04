"""Main orchestrator for the LLM Council 3-stage deliberation pipeline.

Exports ``run_council`` (drives Stage 1 responses → Stage 2 ranking →
Stage 3 synthesis) and ``validate_config`` for pre-flight checks.
Coordinates caching, budget, cost tracking, persistence, and progress
through a ``CouncilContext``.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .aggregation import calculate_aggregate_rankings
from .budget import BudgetExceededError, create_budget_guard
from .context import CouncilContext
from .cost import CouncilCostTracker
from .formatting import format_output, format_stage1_output, format_stage2_output
from .manifest import RunManifest
from .models import CouncilConfig, ModelConfig, generate_response_labels
from .progress import ProgressManager
from .run_options import RunOptions
from .security import sanitize_model_output, sanitize_user_input
from .stages import stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final

if TYPE_CHECKING:
    from .models import AggregateRanking, Stage1Result, Stage3Result
    from .persistence import RunLogger

logger = logging.getLogger("llm-council")


def validate_config(config: dict) -> list[str]:
    """Validate a council configuration dictionary.

    Thin wrapper around ``CouncilConfig.validate_from_dict()``.

    Returns:
        A list of human-readable validation error strings.  An empty list
        means the configuration is valid.
    """
    return CouncilConfig.validate_from_dict(config)


# ---------------------------------------------------------------------------
# Private run state
# ---------------------------------------------------------------------------


@dataclass
class _RunState:
    """Mutable state accumulated across stages of a council run."""

    validated: CouncilConfig
    options: RunOptions
    run_id: str
    start_time: float
    ctx: CouncilContext
    run_logger: RunLogger | None
    council_models: list[ModelConfig]
    chairman_config: ModelConfig | None
    auto_chairman: bool = field(init=False)

    def __post_init__(self) -> None:
        self.auto_chairman = self.chairman_config is None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _setup_context(
    validated: CouncilConfig,
    options: RunOptions,
    run_id: str,
) -> tuple[CouncilContext, RunLogger | None]:
    """Create the per-run context and optional JSONL logger."""
    run_logger = None
    if options.log_dir:
        from .persistence import RunLogger

        run_logger = RunLogger(options.log_dir, run_id)
        run_logger.log_config("[pending]", validated.model_dump())

    if options.context_factory:
        ctx = options.context_factory()
    else:
        from .cache import ResponseCache

        ctx = CouncilContext(
            poe_api_key=os.environ.get("POE_API_KEY"),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            cost_tracker=CouncilCostTracker(),
            budget_guard=create_budget_guard(validated.budget),
            progress=ProgressManager(),
            stage2_max_retries=validated.stage2_retries,
            cache=ResponseCache(ttl=options.cache_ttl) if options.use_cache else None,
            prompt_config=validated.prompts,
        )

    if ctx.budget_guard:
        max_tokens_str = str(ctx.budget_guard.max_tokens) if ctx.budget_guard.max_tokens else "unlimited"
        max_cost_str = str(ctx.budget_guard.max_cost_usd) if ctx.budget_guard.max_cost_usd else "unlimited"
        logger.info(f"Budget limits: {max_tokens_str} tokens, ${max_cost_str}")

    return ctx, run_logger


async def _execute_stage1(
    state: _RunState,
    question: str,
) -> tuple[list[Stage1Result], list[tuple[str, str]]]:
    """Run Stage 1: collect individual responses and record costs.

    Returns (stage1_results, sanitized_responses).
    """
    logger.info("Stage 1: Collecting responses from council...")
    stage1_results, stage1_token_usages = await stage1_collect_responses(
        question,
        state.council_models,
        state.ctx,
        soft_timeout=state.validated.soft_timeout,
        min_responses=state.validated.min_responses,
    )

    if not stage1_results:
        return [], []

    logger.info(f"Stage 1 complete: {len(stage1_results)} responses collected")

    if state.run_logger:
        state.run_logger.log_stage1(stage1_results, stage1_token_usages)

    for result in stage1_results:
        token_usage = stage1_token_usages.get(result.model)
        state.ctx.cost_tracker.record_with_usage(result.model, 1, question, result.response, token_usage)

    sanitized_responses = [(r.model, sanitize_model_output(r.response)) for r in stage1_results]
    return stage1_results, sanitized_responses


async def _execute_stage2(
    state: _RunState,
    question: str,
    stage1_results: list[Stage1Result],
    sanitized_responses: list[tuple[str, str]],
) -> tuple[list[AggregateRanking], int, int, str]:
    """Run Stage 2: collect peer rankings, aggregate, check ballot validity.

    Returns (aggregate_rankings, valid_ballots, total_ballots, low_ballot_warning).
    """
    logger.info("Stage 2: Collecting peer rankings...")
    stage2_results, per_ranker_label_mappings, stage2_token_usages = await stage2_collect_rankings(
        question, stage1_results, state.council_models, state.ctx, response_tuples=sanitized_responses
    )

    logger.info(f"Stage 2 complete: {len(stage2_results)} rankings collected")

    for result in stage2_results:
        token_usage = stage2_token_usages.get(result.model)
        state.ctx.cost_tracker.record_with_usage(result.model, 2, "[ranking prompt]", result.ranking, token_usage)

    if state.run_logger:
        state.run_logger.log_stage2(stage2_results, per_ranker_label_mappings, stage2_token_usages)

    aggregate_rankings, valid_ballots, total_ballots = calculate_aggregate_rankings(
        stage2_results,
        per_ranker_label_mappings,
        seed=state.options.seed,
        attempted_count=len(per_ranker_label_mappings),
    )

    logger.info(f"Valid ballots: {valid_ballots}/{total_ballots}")

    if state.run_logger:
        state.run_logger.log_aggregation(aggregate_rankings, valid_ballots, total_ballots)

    # Check min_valid_ballots threshold
    min_vb = state.validated.min_valid_ballots
    ballot_threshold = min_vb if min_vb is not None else math.ceil(total_ballots / 2)
    low_ballot_warning = ""
    if valid_ballots < ballot_threshold:
        low_ballot_warning = (
            f"\n> \u26a0 Low ballot confidence: only {valid_ballots}/{total_ballots} "
            f"valid ballots (minimum: {ballot_threshold})\n"
        )
        logger.warning(
            f"Low ballot confidence: {valid_ballots}/{total_ballots} valid ballots (minimum: {ballot_threshold})"
        )

    return aggregate_rankings, valid_ballots, total_ballots, low_ballot_warning


async def _execute_stage3(
    state: _RunState,
    question: str,
    stage1_results: list[Stage1Result],
    sanitized_responses: list[tuple[str, str]],
    aggregate_rankings: list[AggregateRanking],
) -> tuple[Stage3Result, dict[str, Any] | None]:
    """Run Stage 3: chairman synthesis with fallback.

    Updates ``state.chairman_config`` if auto-selection or fallback occurs.
    Returns (stage3_result, stage3_token_usage).
    """
    # Build canonical label mapping for the chairman
    label_to_model: dict[str, str] = {}
    labels = generate_response_labels(len(stage1_results))
    for label, result in zip(labels, stage1_results, strict=True):
        label_to_model[label] = result.model

    # Resolve chairman: auto-select #1 ranked model if not configured
    if state.auto_chairman:
        if not aggregate_rankings:
            from .models import Stage3Result as S3R

            err = "Error: No aggregate rankings available to select auto-chairman."
            return S3R(model="unknown", response=err), None
        top_model_name = aggregate_rankings[0].model
        state.chairman_config = next(m for m in state.council_models if m.name == top_model_name)
        logger.info(f"Auto-chairman: selected {top_model_name} (#1 ranked)")

    logger.info("Stage 3: Chairman synthesizing final answer...")
    stage3_result, stage3_token_usage = await stage3_synthesize_final(
        question,
        stage1_results,
        label_to_model,
        aggregate_rankings,
        state.chairman_config,
        state.ctx,
        stream=state.options.stream,
        on_chunk=state.options.on_chunk,
        response_tuples=sanitized_responses,
    )

    # Fallback: if explicit chairman failed, try #1 ranked model
    if not state.auto_chairman and stage3_result.response.startswith("Error:") and aggregate_rankings:
        fallback_name = aggregate_rankings[0].model
        fallback_config = next((m for m in state.council_models if m.name == fallback_name), None)
        if fallback_config and fallback_name != state.chairman_config.name:
            logger.warning(f"Chairman failed, falling back to #1 ranked: {fallback_name}")
            stage3_result, stage3_token_usage = await stage3_synthesize_final(
                question,
                stage1_results,
                label_to_model,
                aggregate_rankings,
                fallback_config,
                state.ctx,
                stream=state.options.stream,
                on_chunk=state.options.on_chunk,
                response_tuples=sanitized_responses,
            )
            state.chairman_config = fallback_config

    logger.info("Stage 3 complete!")

    if state.run_logger:
        state.run_logger.log_stage3(stage3_result, stage3_token_usage)

    state.ctx.cost_tracker.record_with_usage(
        stage3_result.model, 3, "[synthesis prompt]", stage3_result.response, stage3_token_usage
    )

    return stage3_result, stage3_token_usage


def _build_run_summary(cost_tracker: CouncilCostTracker, budget_guard: Any | None) -> str:
    """Compose a unified token usage + budget summary."""
    summary = cost_tracker.summary()
    if budget_guard:
        summary += "\n" + budget_guard.summary()
    return summary


def _assemble_output(
    state: _RunState,
    question: str,
    stage3_result: Stage3Result,
    aggregate_rankings: list[AggregateRanking],
    valid_ballots: int,
    total_ballots: int,
    low_ballot_warning: str,
    stage1_count: int,
) -> str:
    """Format final output with manifest and ballot warnings."""
    summary = _build_run_summary(state.ctx.cost_tracker, state.ctx.budget_guard)
    logger.info(summary)

    total_elapsed = time.time() - state.start_time

    if state.run_logger:
        state.run_logger.log_summary(state.ctx.cost_tracker.summary(), total_elapsed)

    manifest = RunManifest.create(
        question=question,
        config=state.validated.model_dump(),
        stage1_count=stage1_count,
        valid_ballots=valid_ballots,
        total_ballots=total_ballots,
        elapsed_seconds=total_elapsed,
        estimated_tokens=state.ctx.cost_tracker.total_tokens,
        chairman_auto=state.auto_chairman,
        actual_chairman=state.chairman_config.name if state.chairman_config else "unknown",
        run_id=state.run_id,
    )

    if state.options.print_manifest:
        print(manifest.to_json(), file=sys.stderr)

    output = format_output(aggregate_rankings, stage3_result, valid_ballots, total_ballots)
    if low_ballot_warning:
        output = low_ballot_warning + output

    output += manifest.to_comment_block()
    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_council(
    question: str,
    config: CouncilConfig | Mapping[str, Any],
    options: RunOptions | None = None,
) -> str:
    """Run the LLM Council 3-stage deliberation pipeline.

    Execute up to *max_stage* stages of the council process:

    * **Stage 1** — collect independent responses from every council model.
    * **Stage 2** — each model ranks the anonymised peer responses.
    * **Stage 3** — the chairman synthesises a final answer.

    Args:
        question: The user question to present to the council.
        config: Council configuration as a ``CouncilConfig`` instance or
            a plain dict/Mapping (validated internally).
        options: ``RunOptions`` dataclass consolidating all run parameters.
            When ``None``, defaults are used for all options.

    Returns:
        A formatted string containing the council output.

    Raises:
        BudgetExceededError: If the cumulative token or cost usage
            exceeds the limits defined in the ``budget`` section of
            *config*.
    """
    if options is None:
        options = RunOptions()

    # Accept both CouncilConfig and plain dict — parse once, early
    if isinstance(config, CouncilConfig):
        validated = config
    else:
        errors = validate_config(config)
        if errors:
            error_msg = "Configuration errors:\n"
            for e in errors:
                error_msg += f"  - {e}\n"
            return error_msg
        validated = CouncilConfig(**config)

    question = sanitize_user_input(question)
    run_id = str(uuid.uuid4())

    ctx, run_logger = _setup_context(validated, options, run_id)

    # Re-log config with actual question (after sanitization)
    if run_logger:
        run_logger.log_config(question, validated.model_dump())

    state = _RunState(
        validated=validated,
        options=options,
        run_id=run_id,
        start_time=time.time(),
        ctx=ctx,
        run_logger=run_logger,
        council_models=list(validated.council_models),
        chairman_config=validated.chairman,
    )

    async with state.ctx:
        try:
            # Stage 1
            stage1_results, sanitized_responses = await _execute_stage1(state, question)
            if not stage1_results:
                return "Error: All models failed to respond. Please check your API credentials."

            if options.max_stage == 1:
                total_elapsed = time.time() - state.start_time
                await state.ctx.progress.complete_council(total_elapsed)
                return format_stage1_output(stage1_results)

            # Stage 2
            aggregate_rankings, valid_ballots, total_ballots, low_ballot_warning = await _execute_stage2(
                state, question, stage1_results, sanitized_responses
            )

            if state.validated.strict_ballots and low_ballot_warning:
                min_vb = state.validated.min_valid_ballots
                ballot_threshold = min_vb if min_vb is not None else math.ceil(total_ballots / 2)
                total_elapsed = time.time() - state.start_time
                await state.ctx.progress.complete_council(total_elapsed)
                return (
                    f"Error: Insufficient valid ballots ({valid_ballots}/{total_ballots}) "
                    f"to meet threshold ({ballot_threshold}). "
                    f"Set strict_ballots=false or lower min_valid_ballots to proceed."
                )

            if options.max_stage == 2:
                total_elapsed = time.time() - state.start_time
                await state.ctx.progress.complete_council(total_elapsed)
                output = format_stage2_output(aggregate_rankings, stage1_results, valid_ballots, total_ballots)
                return low_ballot_warning + output if low_ballot_warning else output

            # Stage 3
            stage3_result, _ = await _execute_stage3(
                state, question, stage1_results, sanitized_responses, aggregate_rankings
            )

            total_elapsed = time.time() - state.start_time
            await state.ctx.progress.complete_council(total_elapsed)

            return _assemble_output(
                state,
                question,
                stage3_result,
                aggregate_rankings,
                valid_ballots,
                total_ballots,
                low_ballot_warning,
                stage1_count=len(stage1_results),
            )
        except BudgetExceededError as e:
            logger.error(f"Budget exceeded: {e}")
            return f"Error: Budget limit exceeded - {e}"
