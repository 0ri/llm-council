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
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from .aggregation import calculate_aggregate_rankings
from .budget import BudgetExceededError, create_budget_guard
from .context import CouncilContext
from .cost import CouncilCostTracker
from .formatting import format_output, format_stage1_output, format_stage2_output
from .manifest import RunManifest
from .models import CouncilConfig
from .progress import ProgressManager
from .security import sanitize_user_input
from .stages import stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final

logger = logging.getLogger("llm-council")


def validate_config(config: dict) -> list[str]:
    """Validate a council configuration dictionary.

    Delegates structural validation (model fields, budget constraints,
    provider-specific requirements) to ``CouncilConfig`` via Pydantic,
    then performs environment-variable checks that Pydantic cannot handle.

    Args:
        config: Raw configuration dictionary, typically loaded from a
            ``council-config.json`` file.

    Returns:
        A list of human-readable validation error strings.  An empty list
        means the configuration is valid.
    """
    errors: list[str] = []

    # --- Pydantic structural validation ---
    try:
        CouncilConfig(**config)
    except ValidationError as e:
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]

            if error["type"] == "missing":
                if loc == "council_models":
                    errors.append("Config missing 'council_models' list")
                elif loc == "chairman":
                    errors.append("Config missing 'chairman' entry")
                else:
                    errors.append(f"Missing required field: {loc}")
            elif error["type"] in ["list_min_length", "too_short"] and "council_models" in loc:
                errors.append("'council_models' must be a non-empty list")
            elif error["type"] == "literal_error" and "provider" in loc:
                if "council_models" in loc:
                    idx = loc.split(".")[1] if "." in loc else "0"
                    idx_int = int(idx) if idx.isdigit() else 0
                    model_name = config.get("council_models", [{}])[idx_int].get("name", f"model[{idx}]")
                else:
                    model_name = config.get("chairman", {}).get("name", "chairman")
                errors.append(f"{model_name}: unknown provider (must be one of: bedrock, poe, openrouter)")
            elif "budget_tokens" in loc and error["type"] in ("greater_than_equal", "less_than_equal"):
                if "council_models" in loc:
                    idx = int(loc.split(".")[1]) if "." in loc and loc.split(".")[1].isdigit() else 0
                    model_name = config.get("council_models", [{}])[idx].get("name", f"model[{idx}]")
                else:
                    model_name = config.get("chairman", {}).get("name", "chairman")
                errors.append(f"{model_name}: 'budget_tokens' must be integer between 1024 and 128000")
            elif error["type"] == "union_tag_invalid":
                if "council_models" in loc:
                    idx = int(loc.split(".")[1]) if "." in loc and loc.split(".")[1].isdigit() else 0
                    model_name = config.get("council_models", [{}])[idx].get("name", f"model[{idx}]")
                else:
                    model_name = config.get("chairman", {}).get("name", "chairman")
                input_str = str(error.get("input", {}))
                if "model_id" in input_str:
                    errors.append(f"{model_name}: Bedrock provider requires 'model_id'")
                elif "bot_name" in input_str:
                    errors.append(f"{model_name}: Poe provider requires 'bot_name'")
                else:
                    errors.append(f"{loc}: {msg}")
            else:
                errors.append(f"{loc}: {msg}")

    # --- Check for missing name fields (union dispatch can mask this) ---
    all_models = list(config.get("council_models", []))
    if "chairman" in config:
        all_models.append(config["chairman"])
    for i, model in enumerate(all_models):
        if "name" not in model:
            errors.append(f"Model at index {i} missing 'name'")

    # --- Environment variable checks (Pydantic can't do these) ---
    poe_models = [m for m in config.get("council_models", []) if m.get("provider") == "poe"]
    if config.get("chairman", {}).get("provider") == "poe":
        poe_models.append(config["chairman"])
    if poe_models and not os.environ.get("POE_API_KEY"):
        errors.append("POE_API_KEY environment variable required for Poe provider models")

    or_models = [m for m in config.get("council_models", []) if m.get("provider") == "openrouter"]
    if config.get("chairman", {}).get("provider") == "openrouter":
        or_models.append(config["chairman"])
    if or_models and not os.environ.get("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY environment variable required for OpenRouter provider models")

    # Deduplicate preserving order
    seen: set[str] = set()
    return [e for e in errors if not (e in seen or seen.add(e))]


async def run_council(
    question: str,
    config: dict[str, Any],
    print_manifest: bool = False,
    log_dir: str | None = None,
    context_factory: Any = None,
    max_stage: int = 3,
    seed: int | None = None,
    use_cache: bool = True,
    cache_ttl: int = 86400,
    stream: bool = False,
    on_chunk: Callable[[str], Awaitable[None]] | None = None,
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
        config: Council configuration dictionary (see ``validate_config``
            for the expected schema).
        print_manifest: If ``True``, write the run manifest as JSON to
            *stderr* after the run completes.
        log_dir: Directory path for JSONL run logs.  When set, a
            ``RunLogger`` writes per-stage data to this directory.
        context_factory: Optional zero-argument callable that returns a
            ``CouncilContext``.  Primarily used in tests to inject a
            pre-configured context.
        max_stage: Highest stage to execute (1, 2, or 3).  Stages beyond
            this value are skipped and partial output is returned.
        seed: Random seed passed to the bootstrap confidence-interval
            calculation in Stage 2 aggregation for reproducibility.
        use_cache: If ``True``, Stage 1 responses are cached in a local
            SQLite database so repeated identical queries are served from
            cache.
        cache_ttl: Time-to-live in seconds for cached Stage 1 responses.
            Defaults to 86 400 (24 hours).
        stream: If ``True``, Stage 3 synthesis uses the streaming
            provider interface and delivers chunks via *on_chunk*.
        on_chunk: Async callback invoked with each text chunk during
            streaming Stage 3 synthesis.  Ignored when *stream* is
            ``False``.

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
    # Track start time for manifest
    start_time = time.time()

    # Validate configuration first
    errors = validate_config(config)
    if errors:
        error_msg = "Configuration errors:\n"
        for e in errors:
            error_msg += f"  - {e}\n"
        return error_msg

    # Sanitize user input
    question = sanitize_user_input(question)

    # Generate run_id early so RunLogger and RunManifest share the same ID
    run_id = str(uuid.uuid4())

    # Set up optional JSONL persistence
    run_logger = None
    if log_dir:
        from .persistence import RunLogger

        run_logger = RunLogger(log_dir, run_id)
        run_logger.log_config(question, config)

    council_models = config.get("council_models", [])
    chairman_config = config.get("chairman", {})

    # Create the per-run context
    if context_factory:
        ctx = context_factory()
    else:
        from .cache import ResponseCache

        ctx = CouncilContext(
            poe_api_key=os.environ.get("POE_API_KEY"),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            cost_tracker=CouncilCostTracker(),
            budget_guard=create_budget_guard(config),
            progress=ProgressManager(),
            stage2_max_retries=config.get("stage2_retries", 1),
            cache=ResponseCache(ttl=cache_ttl) if use_cache else None,
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
                soft_timeout=config.get("soft_timeout"),
                min_responses=config.get("min_responses"),
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

            logger.info("Stage 2: Collecting peer rankings...")
            stage2_results, per_ranker_label_mappings, stage2_token_usages = await stage2_collect_rankings(
                question, stage1_results, council_models, ctx
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
            aggregate_rankings, valid_ballots, total_ballots = calculate_aggregate_rankings(
                stage2_results, per_ranker_label_mappings, seed=seed
            )

            logger.info(f"Valid ballots: {valid_ballots}/{total_ballots}")

            # Item 14: Check min_valid_ballots threshold
            min_vb = config.get("min_valid_ballots")
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
                for i, result in enumerate(stage1_results):
                    label_to_model[f"Response {chr(65 + i)}"] = result.model

            logger.info("Stage 3: Chairman synthesizing final answer...")
            stage3_result, stage3_token_usage = await stage3_synthesize_final(
                question,
                stage1_results,
                stage2_results,
                label_to_model,
                aggregate_rankings,
                chairman_config,
                ctx,
                stream=stream,
                on_chunk=on_chunk,
            )

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
                config=config,
                stage1_count=len(stage1_results),
                valid_ballots=valid_ballots,
                total_ballots=total_ballots,
                elapsed_seconds=total_elapsed,
                estimated_tokens=ctx.cost_tracker.total_tokens,
            )
            manifest.run_id = run_id

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
