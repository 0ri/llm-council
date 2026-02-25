"""Main orchestrator for the LLM Council deliberation process."""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from pydantic import ValidationError

from .aggregation import calculate_aggregate_rankings
from .budget import BudgetExceededError, create_budget_guard
from .cost import CouncilCostTracker
from .formatting import format_output
from .manifest import RunManifest
from .models import CouncilConfig
from .progress import ProgressManager
from .security import sanitize_user_input
from .stages import stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final

logger = logging.getLogger("llm-council")


def validate_config(config: dict) -> list[str]:
    """Validate council configuration. Returns list of error messages (empty = valid)."""
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
                    errors.append(f"{model_name}: unknown provider (must be one of: bedrock, poe)")
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


async def run_council(question: str, config: dict[str, Any], print_manifest: bool = False) -> str:
    """Run the full 3-stage council process.

    Args:
        question: The question to ask the council
        config: Council configuration dict
        print_manifest: If True, print manifest JSON to stderr

    Returns:
        Formatted council output string
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

    council_models = config.get("council_models", [])
    chairman_config = config.get("chairman", {})
    poe_api_key = os.environ.get("POE_API_KEY")

    progress = ProgressManager()

    cost_tracker = CouncilCostTracker()

    # Create budget guard if configured
    budget_guard = create_budget_guard(config)
    if budget_guard:
        max_tokens_str = str(budget_guard.max_tokens) if budget_guard.max_tokens else "unlimited"
        max_cost_str = str(budget_guard.max_cost_usd) if budget_guard.max_cost_usd else "unlimited"
        logger.info(f"Budget limits: {max_tokens_str} tokens, ${max_cost_str}")

    try:
        logger.info("Stage 1: Collecting responses from council...")
        stage1_results, stage1_token_usages = await stage1_collect_responses(
            question, council_models, poe_api_key, progress=progress
        )

        if not stage1_results:
            return "Error: All models failed to respond. Please check your API credentials."

        logger.info(f"Stage 1 complete: {len(stage1_results)} responses collected")

        # Record stage 1 token usage with actual counts when available
        for result in stage1_results:
            model_name = result.model
            token_usage = stage1_token_usages.get(model_name)
            cost_tracker.record_with_usage(model_name, 1, question, result.response, token_usage)

        logger.info("Stage 2: Collecting peer rankings...")
        stage2_results, per_ranker_label_mappings, stage2_token_usages = await stage2_collect_rankings(
            question, stage1_results, council_models, poe_api_key, progress=progress
        )

        logger.info(f"Stage 2 complete: {len(stage2_results)} rankings collected")

        # Record stage 2 token usage with actual counts when available
        for result in stage2_results:
            model_name = result.model
            token_usage = stage2_token_usages.get(model_name)
            cost_tracker.record_with_usage(model_name, 2, "[ranking prompt]", result.ranking, token_usage)

        # Calculate aggregate rankings with ballot validity tracking
        aggregate_rankings, valid_ballots, total_ballots = calculate_aggregate_rankings(
            stage2_results, per_ranker_label_mappings
        )

        logger.info(f"Valid ballots: {valid_ballots}/{total_ballots}")

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
            poe_api_key,
            progress=progress,
        )

        logger.info("Stage 3 complete!")

        # Record stage 3 token usage with actual counts when available
        cost_tracker.record_with_usage(
            stage3_result.model, 3, "[synthesis prompt]", stage3_result.response, stage3_token_usage
        )

        logger.info(cost_tracker.summary())

        # Log budget usage if configured
        if budget_guard:
            logger.info(budget_guard.summary())

        total_elapsed = time.time() - start_time
        await progress.complete_council(total_elapsed)

        # Create run manifest
        manifest = RunManifest.create(
            question=question,
            config=config,
            stage1_count=len(stage1_results),
            valid_ballots=valid_ballots,
            total_ballots=total_ballots,
            elapsed_seconds=total_elapsed,
            estimated_tokens=cost_tracker.total_tokens,
        )

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
        await progress._cleanup()
