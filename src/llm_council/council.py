"""Main orchestrator for the LLM Council deliberation process."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from .aggregation import calculate_aggregate_rankings
from .cost import CouncilCostTracker
from .formatting import format_output
from .progress import ProgressManager
from .security import sanitize_user_input
from .stages import stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final

logger = logging.getLogger("llm-council")


def validate_config(config: dict) -> list[str]:
    """Validate council configuration. Returns list of error messages (empty = valid)."""
    errors: list[str] = []

    if "council_models" not in config:
        errors.append("Config missing 'council_models' list")
    elif not isinstance(config["council_models"], list) or len(config["council_models"]) == 0:
        errors.append("'council_models' must be a non-empty list")

    if "chairman" not in config:
        errors.append("Config missing 'chairman' entry")

    all_models = list(config.get("council_models", []))
    if "chairman" in config:
        all_models.append(config["chairman"])

    valid_providers = {"bedrock", "poe"}

    for i, model in enumerate(all_models):
        label = model.get("name", f"model[{i}]")

        if "name" not in model:
            errors.append(f"Model at index {i} missing 'name'")

        if "provider" not in model:
            errors.append(f"{label}: missing 'provider' field")
        elif model["provider"] not in valid_providers:
            errors.append(f"{label}: unknown provider '{model['provider']}' (must be one of: {valid_providers})")
        elif model["provider"] == "bedrock":
            if "model_id" not in model:
                errors.append(f"{label}: Bedrock provider requires 'model_id'")
            if "budget_tokens" in model:
                bt = model["budget_tokens"]
                if not isinstance(bt, int) or bt < 1024 or bt > 128000:
                    errors.append(f"{label}: 'budget_tokens' must be integer between 1024 and 128000")
        elif model["provider"] == "poe":
            if "bot_name" not in model:
                errors.append(f"{label}: Poe provider requires 'bot_name'")
            if "reasoning_effort" in model:
                valid_efforts = {"medium", "high", "Xhigh", "minimal", "low"}
                if model["reasoning_effort"] not in valid_efforts:
                    errors.append(f"{label}: 'reasoning_effort' must be one of: {valid_efforts}")

    # Check POE_API_KEY if any model uses poe
    poe_models = [m for m in config.get("council_models", []) if m.get("provider") == "poe"]
    if config.get("chairman", {}).get("provider") == "poe":
        poe_models.append(config["chairman"])
    if poe_models and not os.environ.get("POE_API_KEY"):
        errors.append("POE_API_KEY environment variable required for Poe provider models")

    return errors


async def run_council(question: str, config: dict[str, Any]) -> str:
    """Run the full 3-stage council process."""
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

    try:
        logger.info("Stage 1: Collecting responses from council...")
        stage1_results = await stage1_collect_responses(question, council_models, poe_api_key, progress=progress)

        if not stage1_results:
            return "Error: All models failed to respond. Please check your API credentials."

        logger.info(f"Stage 1 complete: {len(stage1_results)} responses collected")

        # Record stage 1 token usage
        for result in stage1_results:
            cost_tracker.record(result["model"], 1, question, result["response"])

        logger.info("Stage 2: Collecting peer rankings...")
        stage2_results, label_to_model = await stage2_collect_rankings(
            question, stage1_results, council_models, poe_api_key, progress=progress
        )

        logger.info(f"Stage 2 complete: {len(stage2_results)} rankings collected")

        # Record stage 2 token usage
        for result in stage2_results:
            cost_tracker.record(result["model"], 2, "[ranking prompt]", result.get("ranking", ""))

        # Calculate aggregate rankings with ballot validity tracking
        aggregate_rankings, valid_ballots, total_ballots = calculate_aggregate_rankings(
            stage2_results, label_to_model
        )

        logger.info(f"Valid ballots: {valid_ballots}/{total_ballots}")

        logger.info("Stage 3: Chairman synthesizing final answer...")
        stage3_result = await stage3_synthesize_final(
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

        # Record stage 3 token usage
        cost_tracker.record(stage3_result["model"], 3, "[synthesis prompt]", stage3_result["response"])

        logger.info(cost_tracker.summary())

        total_elapsed = time.time() - progress.total_start_time
        await progress.complete_council(total_elapsed)

        # Format output with ballot validity
        return format_output(aggregate_rankings, stage3_result, valid_ballots, total_ballots)
    finally:
        await progress._cleanup()
