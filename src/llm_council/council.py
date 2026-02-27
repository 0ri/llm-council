"""Main orchestrator for the LLM Council 3-stage deliberation pipeline.

Exports ``run_council`` (drives Stage 1 responses → Stage 2 ranking →
Stage 3 synthesis) and ``validate_config`` for pre-flight checks.
Coordinates caching, budget, cost tracking, persistence, and progress
through a ``CouncilContext``.
"""

from __future__ import annotations

import logging
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

    Check that *config* contains valid ``council_models`` and ``chairman``
    entries, that provider names are recognised, that required
    provider-specific fields are present, and that budget values (when
    supplied) are within acceptable ranges.  Environment variables
    (``POE_API_KEY``, ``OPENROUTER_API_KEY``) are also checked when the
    corresponding providers appear in the config.

    Args:
        config: Raw configuration dictionary, typically loaded from a
            ``council-config.json`` file.  Expected top-level keys include
            ``council_models`` (list of model dicts) and ``chairman``
            (a single model dict).  An optional ``budget`` dict may
            contain ``max_tokens``, ``max_cost_usd``,
            ``input_cost_per_1k``, and ``output_cost_per_1k``.

    Returns:
        A list of human-readable validation error strings.  An empty list
        means the configuration is valid.

    Raises:
        TypeError: If *config* is not a dictionary (raised by Pydantic
            when unpacking fails).
    """
    errors: list[str] = []

    # Try to validate basic structure with Pydantic
    try:
        # This validates the council structure using Pydantic models
        CouncilConfig(**config)
    except ValidationError as e:
        # Convert Pydantic validation errors to our format
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]

            # Custom formatting for common errors
            if error["type"] == "missing":
                if loc == "council_models":
                    errors.append("Config missing 'council_models' list")
                elif loc == "chairman":
                    errors.append("Config missing 'chairman' entry")
                else:
                    errors.append(f"Missing required field: {loc}")
            elif error["type"] in ["list_min_length", "too_short"] and "council_models" in loc:
                errors.append("'council_models' must be a non-empty list")
            elif error["type"] == "literal_error":
                if "provider" in loc:
                    # Extract model name from location
                    if "council_models" in loc:
                        idx = loc.split(".")[1] if "." in loc else "0"
                        idx_int = int(idx) if idx.isdigit() else 0
                        model_name = config.get("council_models", [{}])[idx_int].get("name", f"model[{idx}]")
                    else:
                        model_name = config.get("chairman", {}).get("name", "chairman")
                    errors.append(f"{model_name}: unknown provider (must be one of: bedrock, poe, openrouter)")
                else:
                    errors.append(f"{loc}: {msg}")
            elif "model_id" in str(error.get("input", {})) and error["type"] == "union_tag_invalid":
                # Handle missing model_id for bedrock
                if "council_models" in loc:
                    idx = int(loc.split(".")[1]) if "." in loc and loc.split(".")[1].isdigit() else 0
                    model_name = config.get("council_models", [{}])[idx].get("name", f"model[{idx}]")
                else:
                    model_name = config.get("chairman", {}).get("name", "chairman")
                errors.append(f"{model_name}: Bedrock provider requires 'model_id'")
            elif "bot_name" in str(error.get("input", {})) and error["type"] == "union_tag_invalid":
                # Handle missing bot_name for poe
                if "council_models" in loc:
                    idx = int(loc.split(".")[1]) if "." in loc and loc.split(".")[1].isdigit() else 0
                    model_name = config.get("council_models", [{}])[idx].get("name", f"model[{idx}]")
                else:
                    model_name = config.get("chairman", {}).get("name", "chairman")
                errors.append(f"{model_name}: Poe provider requires 'bot_name'")
            elif error["type"] == "greater_than_equal" and "budget_tokens" in loc:
                if "council_models" in loc:
                    idx = int(loc.split(".")[1]) if "." in loc and loc.split(".")[1].isdigit() else 0
                    model_name = config.get("council_models", [{}])[idx].get("name", f"model[{idx}]")
                else:
                    model_name = config.get("chairman", {}).get("name", "chairman")
                errors.append(f"{model_name}: 'budget_tokens' must be integer between 1024 and 128000")
            elif error["type"] == "less_than_equal" and "budget_tokens" in loc:
                if "council_models" in loc:
                    idx = int(loc.split(".")[1]) if "." in loc and loc.split(".")[1].isdigit() else 0
                    model_name = config.get("council_models", [{}])[idx].get("name", f"model[{idx}]")
                else:
                    model_name = config.get("chairman", {}).get("name", "chairman")
                errors.append(f"{model_name}: 'budget_tokens' must be integer between 1024 and 128000")
            else:
                # Generic error formatting
                errors.append(f"{loc}: {msg}")

    # Check for missing name fields (Pydantic doesn't validate this well for union types)
    all_models = list(config.get("council_models", []))
    if "chairman" in config:
        all_models.append(config["chairman"])

    for i, model in enumerate(all_models):
        if "name" not in model:
            errors.append(f"Model at index {i} missing 'name'")

    # Check POE_API_KEY if any model uses poe (env check that Pydantic can't do)
    poe_models = [m for m in config.get("council_models", []) if m.get("provider") == "poe"]
    if config.get("chairman", {}).get("provider") == "poe":
        poe_models.append(config["chairman"])
    if poe_models and not os.environ.get("POE_API_KEY"):
        errors.append("POE_API_KEY environment variable required for Poe provider models")

    # Check OPENROUTER_API_KEY if any model uses openrouter
    or_models = [m for m in config.get("council_models", []) if m.get("provider") == "openrouter"]
    if config.get("chairman", {}).get("provider") == "openrouter":
        or_models.append(config["chairman"])
    if or_models and not os.environ.get("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY environment variable required for OpenRouter provider models")

    # Validate budget config if present (not in Pydantic model)
    if "budget" in config:
        budget = config["budget"]
        if "max_tokens" in budget:
            if not isinstance(budget["max_tokens"], int) or budget["max_tokens"] <= 0:
                errors.append("budget.max_tokens must be a positive integer")
        if "max_cost_usd" in budget:
            if not isinstance(budget["max_cost_usd"], (int, float)) or budget["max_cost_usd"] <= 0:
                errors.append("budget.max_cost_usd must be a positive number")
        if "input_cost_per_1k" in budget:
            if not isinstance(budget["input_cost_per_1k"], (int, float)) or budget["input_cost_per_1k"] < 0:
                errors.append("budget.input_cost_per_1k must be a non-negative number")
        if "output_cost_per_1k" in budget:
            if not isinstance(budget["output_cost_per_1k"], (int, float)) or budget["output_cost_per_1k"] < 0:
                errors.append("budget.output_cost_per_1k must be a non-negative number")

    # Remove duplicates while preserving order
    seen = set()
    unique_errors = []
    for error in errors:
        if error not in seen:
            seen.add(error)
            unique_errors.append(error)

    return unique_errors


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

        if run_logger:
            run_logger.log_aggregation(aggregate_rankings, valid_ballots, total_ballots)

        if max_stage == 2:
            total_elapsed = time.time() - start_time
            await ctx.progress.complete_council(total_elapsed)
            return format_stage2_output(aggregate_rankings, stage1_results, valid_ballots, total_ballots)

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

        # Append manifest as comment block
        output += manifest.to_comment_block()

        return output
    except BudgetExceededError as e:
        logger.error(f"Budget exceeded: {e}")
        return f"Error: Budget limit exceeded - {e}"
    finally:
        await ctx.shutdown()
