"""Stage 1: Collect individual responses from all council models."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from ..context import CouncilContext
from ..models import ModelConfig, Stage1Result, coerce_model_config, get_model_identifier
from ..progress import ModelStatus
from .execution import query_models_parallel

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
