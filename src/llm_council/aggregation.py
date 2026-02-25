"""Ranking aggregation logic for LLM Council."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .models import AggregateRanking, Stage2Result


def bootstrap_confidence_intervals(
    model_positions: list[int],
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate bootstrap confidence intervals for average rank.

    Args:
        model_positions: List of ranking positions for a model
        n_resamples: Number of bootstrap resamples
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    if not model_positions:
        return (0.0, 0.0)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_resamples):
        resample = random.choices(model_positions, k=len(model_positions))
        bootstrap_means.append(sum(resample) / len(resample))

    # Sort bootstrap means
    bootstrap_means.sort()

    # Calculate percentiles for confidence interval
    alpha = 1 - confidence
    lower_percentile = alpha / 2
    upper_percentile = 1 - alpha / 2

    lower_idx = int(n_resamples * lower_percentile)
    upper_idx = int(n_resamples * upper_percentile)

    # Ensure indices are within bounds
    lower_idx = max(0, min(lower_idx, n_resamples - 1))
    upper_idx = max(0, min(upper_idx, n_resamples - 1))

    return (
        round(bootstrap_means[lower_idx], 2),
        round(bootstrap_means[upper_idx], 2),
    )


def calculate_borda_score(positions: list[int], n_candidates: int) -> float:
    """Calculate Borda count score.

    For N candidates, first place gets N-1 points, second gets N-2, etc.

    Args:
        positions: List of ranking positions (1-based)
        n_candidates: Total number of candidates

    Returns:
        Average Borda score
    """
    if not positions:
        return 0.0

    borda_points = []
    for position in positions:
        # Convert position to Borda points
        # Position 1 gets n_candidates-1 points, position 2 gets n_candidates-2, etc.
        points = max(0, n_candidates - position)
        borda_points.append(points)

    return round(sum(borda_points) / len(borda_points), 2)


def calculate_aggregate_rankings(
    stage2_results: list[Stage2Result] | list[dict[str, Any]],
    label_mappings: dict[str, str] | dict[str, dict[str, str]],
) -> tuple[list[AggregateRanking], int, int]:
    """Calculate aggregate rankings across all models.

    Args:
        stage2_results: List of ranking results from stage2 (can be Stage2Result or dict for backwards compatibility)
        label_mappings: Either a simple dict (old format) or per-ranker mappings (new format)

    Returns:
        Tuple of (aggregate rankings list, valid ballot count, total ballot count)
    """
    model_positions: dict[str, list[int]] = defaultdict(list)
    valid_ballots = 0
    total_ballots = len(stage2_results)

    # Determine if we have per-ranker mappings or a single mapping
    per_ranker_mappings = isinstance(label_mappings, dict) and any(isinstance(v, dict) for v in label_mappings.values())

    # Collect all unique models that received rankings
    all_models = set()

    for ranking in stage2_results:
        # Handle both Stage2Result and dict for backwards compatibility
        if isinstance(ranking, Stage2Result):
            ranker_name = ranking.model
            parsed_ranking = ranking.parsed_ranking
            is_valid = ranking.is_valid_ballot
        else:
            ranker_name = ranking.get("model", "unknown")
            parsed_ranking = ranking.get("parsed_ranking", [])
            is_valid = ranking.get("is_valid_ballot", False)

        if not is_valid:
            continue

        valid_ballots += 1

        # Get the appropriate label mapping for this ranker
        if per_ranker_mappings:
            # New format: per-ranker mappings
            ranker_labels = label_mappings.get(ranker_name, {})
        else:
            # Old format: single mapping for all rankers
            ranker_labels = label_mappings

        for position, label in enumerate(parsed_ranking, start=1):
            if label in ranker_labels:
                model_name = ranker_labels[label]
                model_positions[model_name].append(position)
                all_models.add(model_name)

    # Calculate the maximum number of candidates any ranker evaluated
    # (accounting for self-exclusion, different rankers may see different counts)
    max_candidates = 0
    for ranking in stage2_results:
        if isinstance(ranking, Stage2Result):
            max_candidates = max(max_candidates, len(ranking.parsed_ranking))
        else:
            max_candidates = max(max_candidates, len(ranking.get("parsed_ranking", [])))

    # Calculate aggregate metrics for each model
    aggregate: list[AggregateRanking] = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)

            # Calculate confidence intervals
            ci_lower, ci_upper = bootstrap_confidence_intervals(positions)

            # Calculate Borda score
            # Use max_candidates as the number of candidates for Borda scoring
            borda_score = calculate_borda_score(positions, max_candidates)

            aggregate.append(
                AggregateRanking(
                    model=model,
                    average_rank=round(avg_rank, 2),
                    rankings_count=len(positions),
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    borda_score=borda_score,
                )
            )

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x.average_rank)

    return (aggregate, valid_ballots, total_ballots)
