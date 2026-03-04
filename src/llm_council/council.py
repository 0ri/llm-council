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
from typing import Any

from .aggregation import calculate_aggregate_rankings
from .budget import BudgetExceededError, create_budget_guard
from .context import CouncilContext
from .cost import CouncilCostTracker
from .formatting import format_output, format_stage1_output, format_stage2_output
from .manifest import RunManifest
from .models import CouncilConfig, generate_response_labels
from .progress import ProgressManager
from .run_options import RunOptions
from .security import sanitize_model_output, sanitize_user_input
from .stages import stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final

logger = logging.getLogger("llm-council")


def validate_config(config: dict) -> list[str]:
    """Validate a council configuration dictionary.

    Thin wrapper around ``CouncilConfig.validate_from_dict()``.

    Returns:
        A list of human-readable validation error strings.  An empty list
        means the configuration is valid.
    """
    return CouncilConfig.validate_from_dict(config)


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

    The function handles input sanitisation, configuration validation,
    caching, budget enforcement, cost tracking, JSONL persistence, and
    progress reporting through the ``CouncilContext``.

    Args:
        question: The user question to present to the council.
        config: Council configuration as a ``CouncilConfig`` instance or
            a plain dict/Mapping (validated internally).
        options: ``RunOptions`` dataclass consolidating all run parameters.
            When ``None``, defaults are used for all options.

    Returns:
        A formatted string containing the council output.  The exact
        format depends on *max_stage*: Stage-1-only output, Stage-1+2
        output with rankings, or the full 3-stage output including the
        chairman synthesis and an appended run-manifest comment block.

    Raises:
        BudgetExceededError: If the cumulative token or cost usage
            exceeds the limits defined in the ``budget`` section of
            *config*.
    """
    if options is None:
        options = RunOptions()

    print_manifest = options.print_manifest
    log_dir = options.log_dir
    context_factory = options.context_factory
    max_stage = options.max_stage
    seed = options.seed
    use_cache = options.use_cache
    cache_ttl = options.cache_ttl
    stream = options.stream
    on_chunk = options.on_chunk

    # Track start time for manifest
    start_time = time.time()

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

    # Sanitize user input
    question = sanitize_user_input(question)

    # Generate run_id early so RunLogger and RunManifest share the same ID
    run_id = str(uuid.uuid4())

    # Set up optional JSONL persistence
    run_logger = None
    if log_dir:
        from .persistence import RunLogger

        run_logger = RunLogger(log_dir, run_id)
        run_logger.log_config(question, validated.model_dump())

    # Extract typed ModelConfig objects from validated config
    council_models = list(validated.council_models)
    chairman_config_typed = validated.chairman
    auto_chairman = chairman_config_typed is None

    # Create the per-run context
    if context_factory:
        ctx = context_factory()
    else:
        from .cache import ResponseCache

        ctx = CouncilContext(
            poe_api_key=os.environ.get("POE_API_KEY"),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            cost_tracker=CouncilCostTracker(),
            budget_guard=create_budget_guard(validated.budget),
            progress=ProgressManager(),
            stage2_max_retries=validated.stage2_retries,
            cache=ResponseCache(ttl=cache_ttl) if use_cache else None,
            prompt_config=validated.prompts,
        )

    if ctx.budget_guard:
        max_tokens_str = str(ctx.budget_guard.max_tokens) if ctx.budget_guard.max_tokens else "unlimited"
        max_cost_str = str(ctx.budget_guard.max_cost_usd) if ctx.budget_guard.max_cost_usd else "unlimited"
        logger.info(f"Budget limits: {max_tokens_str} tokens, ${max_cost_str}")

    async with ctx:
        try:
            logger.info("Stage 1: Collecting responses from council...")
            stage1_results, stage1_token_usages = await stage1_collect_responses(
                question,
                council_models,
                ctx,
                soft_timeout=validated.soft_timeout,
                min_responses=validated.min_responses,
            )

            if not stage1_results:
                return "Error: All models failed to respond. Please check your API credentials."

            logger.info(f"Stage 1 complete: {len(stage1_results)} responses collected")

            if run_logger:
                run_logger.log_stage1(stage1_results, stage1_token_usages)

            # Record stage 1 token usage with actual counts when available
            for result in stage1_results:
                model_name = result.model
                token_usage = stage1_token_usages.get(model_name)
                ctx.cost_tracker.record_with_usage(model_name, 1, question, result.response, token_usage)

            if max_stage == 1:
                total_elapsed = time.time() - start_time
                await ctx.progress.complete_council(total_elapsed)
                return format_stage1_output(stage1_results)

            # Pre-compute sanitized response tuples for Stage 2 and 3
            sanitized_responses = [
                (r.model, sanitize_model_output(r.response)) for r in stage1_results
            ]

            logger.info("Stage 2: Collecting peer rankings...")
            stage2_results, per_ranker_label_mappings, stage2_token_usages = await stage2_collect_rankings(
                question, stage1_results, council_models, ctx, response_tuples=sanitized_responses
            )

            logger.info(f"Stage 2 complete: {len(stage2_results)} rankings collected")

            # Record stage 2 token usage with actual counts when available
            for result in stage2_results:
                model_name = result.model
                token_usage = stage2_token_usages.get(model_name)
                ctx.cost_tracker.record_with_usage(model_name, 2, "[ranking prompt]", result.ranking, token_usage)

            if run_logger:
                run_logger.log_stage2(stage2_results, per_ranker_label_mappings, stage2_token_usages)

            # Calculate aggregate rankings with ballot validity tracking
            # Use len(per_ranker_label_mappings) as attempted count to avoid undercounting
            # when models fail before producing any Stage2Result
            aggregate_rankings, valid_ballots, total_ballots = calculate_aggregate_rankings(
                stage2_results,
                per_ranker_label_mappings,
                seed=seed,
                attempted_count=len(per_ranker_label_mappings),
            )

            logger.info(f"Valid ballots: {valid_ballots}/{total_ballots}")

            # Item 14: Check min_valid_ballots threshold
            min_vb = validated.min_valid_ballots
            ballot_threshold = min_vb if min_vb is not None else math.ceil(total_ballots / 2)
            low_ballot_warning = ""
            if valid_ballots < ballot_threshold:
                low_ballot_warning = (
                    f"\n> \u26a0 Low ballot confidence: only {valid_ballots}/{total_ballots} "
                    f"valid ballots (minimum: {ballot_threshold})\n"
                )
                logger.warning(
                    f"Low ballot confidence: {valid_ballots}/{total_ballots} valid ballots "
                    f"(minimum: {ballot_threshold})"
                )
                if validated.strict_ballots:
                    total_elapsed = time.time() - start_time
                    await ctx.progress.complete_council(total_elapsed)
                    return (
                        f"Error: Insufficient valid ballots ({valid_ballots}/{total_ballots}) "
                        f"to meet threshold ({ballot_threshold}). "
                        f"Set strict_ballots=false or lower min_valid_ballots to proceed."
                    )

            if run_logger:
                run_logger.log_aggregation(aggregate_rankings, valid_ballots, total_ballots)

            if max_stage == 2:
                total_elapsed = time.time() - start_time
                await ctx.progress.complete_council(total_elapsed)
                output = format_stage2_output(aggregate_rankings, stage1_results, valid_ballots, total_ballots)
                return low_ballot_warning + output if low_ballot_warning else output

            # For stage 3, create a unified label mapping (using the first ranker's mapping for consistency)
            # This is needed because the chairman needs a single view of the anonymized labels
            label_to_model = {}
            if per_ranker_label_mappings:
                # Use the mapping from any ranker (they all map to the same models, just different orders)
                # We need to create a canonical mapping for the chairman
                labels = generate_response_labels(len(stage1_results))
                for label, result in zip(labels, stage1_results, strict=True):
                    label_to_model[label] = result.model

            # Resolve chairman: auto-select #1 ranked model if not configured
            if auto_chairman:
                if not aggregate_rankings:
                    return "Error: No aggregate rankings available to select auto-chairman."
                top_model_name = aggregate_rankings[0].model
                chairman_config_typed = next(m for m in council_models if m.name == top_model_name)
                logger.info(f"Auto-chairman: selected {top_model_name} (#1 ranked)")

            logger.info("Stage 3: Chairman synthesizing final answer...")
            stage3_result, stage3_token_usage = await stage3_synthesize_final(
                question,
                stage1_results,
                label_to_model,
                aggregate_rankings,
                chairman_config_typed,
                ctx,
                stream=stream,
                on_chunk=on_chunk,
                response_tuples=sanitized_responses,
            )

            # Fallback: if explicit chairman failed, try #1 ranked model
            if not auto_chairman and stage3_result.response.startswith("Error:") and aggregate_rankings:
                fallback_name = aggregate_rankings[0].model
                fallback_config = next((m for m in council_models if m.name == fallback_name), None)
                if fallback_config and fallback_name != chairman_config_typed.name:
                    logger.warning(f"Chairman failed, falling back to #1 ranked: {fallback_name}")
                    stage3_result, stage3_token_usage = await stage3_synthesize_final(
                        question,
                        stage1_results,
                        label_to_model,
                        aggregate_rankings,
                        fallback_config,
                        ctx,
                        stream=stream,
                        on_chunk=on_chunk,
                        response_tuples=sanitized_responses,
                    )
                    chairman_config_typed = fallback_config

            logger.info("Stage 3 complete!")

            if run_logger:
                run_logger.log_stage3(stage3_result, stage3_token_usage)

            # Record stage 3 token usage with actual counts when available
            ctx.cost_tracker.record_with_usage(
                stage3_result.model, 3, "[synthesis prompt]", stage3_result.response, stage3_token_usage
            )

            logger.info(ctx.cost_tracker.summary())

            # Log budget usage if configured
            if ctx.budget_guard:
                logger.info(ctx.budget_guard.summary())

            total_elapsed = time.time() - start_time
            await ctx.progress.complete_council(total_elapsed)

            if run_logger:
                run_logger.log_summary(ctx.cost_tracker.summary(), total_elapsed)

            # Create run manifest (use the same run_id generated earlier)
            manifest = RunManifest.create(
                question=question,
                config=validated.model_dump(),
                stage1_count=len(stage1_results),
                valid_ballots=valid_ballots,
                total_ballots=total_ballots,
                elapsed_seconds=total_elapsed,
                estimated_tokens=ctx.cost_tracker.total_tokens,
                chairman_auto=auto_chairman,
                actual_chairman=chairman_config_typed.name if chairman_config_typed else "unknown",
                run_id=run_id,
            )

            # Print manifest to stderr if requested
            if print_manifest:
                print(manifest.to_json(), file=sys.stderr)

            # Format output with ballot validity and manifest
            output = format_output(aggregate_rankings, stage3_result, valid_ballots, total_ballots)
            if low_ballot_warning:
                output = low_ballot_warning + output

            # Append manifest as comment block
            output += manifest.to_comment_block()

            return output
        except BudgetExceededError as e:
            logger.error(f"Budget exceeded: {e}")
            return f"Error: Budget limit exceeded - {e}"
