"""Ranking aggregation for Stage 2 peer-review ballots.

Provides ``calculate_aggregate_rankings`` to combine per-model ranking
ballots into a leaderboard using average rank, Borda count scoring, and
bootstrap confidence intervals. Called after Stage 2 to feed Stage 3.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from .models import AggregateRanking, Stage2Result

logger = logging.getLogger("llm-council")


def bootstrap_confidence_intervals(
    model_positions: list[int],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Calculate bootstrap confidence intervals for average rank.

    Args:
        model_positions: List of ranking positions for a model
        n_resamples: Number of bootstrap resamples
        confidence: Confidence level (default 0.95 for 95% CI)
        rng: Optional seeded Random instance for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    if not model_positions:
        return (0.0, 0.0)

    _rng = rng or random

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_resamples):
        resample = _rng.choices(model_positions, k=len(model_positions))
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
    stage2_results: list[Stage2Result],
    label_mappings: dict[str, dict[str, str]],
    seed: int | None = None,
    attempted_count: int | None = None,
) -> tuple[list[AggregateRanking], int, int]:
    """Calculate aggregate rankings across all models.

    Args:
        stage2_results: List of Stage2Result ranking results
        label_mappings: Per-ranker mappings: {ranker_name: {label: model_name}}
        seed: Optional seed for reproducible bootstrap confidence intervals
        attempted_count: Number of models that attempted ranking. When provided,
            used as total_ballots instead of len(stage2_results) to avoid
            undercounting when models fail before producing any result.

    Returns:
        Tuple of (aggregate rankings list, valid ballot count, total ballot count)
    """
    model_positions: dict[str, list[int]] = defaultdict(list)
    model_normalized_borda: dict[str, list[float]] = defaultdict(list)
    valid_ballots = 0
    total_ballots = attempted_count if attempted_count is not None else len(stage2_results)

    for ranking in stage2_results:
        if not ranking.is_valid_ballot:
            continue

        ranker_labels = label_mappings.get(ranking.model, {})

        # Track resolved vs phantom labels for this ballot
        resolved: list[tuple[str, int]] = []
        phantom_labels: list[str] = []
        for position, label in enumerate(ranking.parsed_ranking, start=1):
            if label in ranker_labels:
                resolved.append((ranker_labels[label], position))
            else:
                phantom_labels.append(label)

        # Item 12: Invalidate ballots with phantom (unresolvable) labels
        if phantom_labels:
            logger.warning(
                f"Phantom labels in ballot from {ranking.model}: {phantom_labels}. "
                f"Ballot invalidated ({len(resolved)}/{len(ranking.parsed_ranking)} labels resolved)."
            )
            continue

        valid_ballots += 1
        n_candidates = len(ranking.parsed_ranking)

        for model_name, position in resolved:
            model_positions[model_name].append(position)

            # Item 13: Accumulate per-ballot normalized Borda scores (0-1 scale)
            raw_borda = max(0, n_candidates - position)
            if n_candidates > 1:
                normalized = raw_borda / (n_candidates - 1)
            else:
                normalized = 1.0  # Single candidate always gets full score
            model_normalized_borda[model_name].append(normalized)

    # Create seeded RNG if seed provided
    rng = random.Random(seed) if seed is not None else None

    # Calculate aggregate metrics for each model
    aggregate: list[AggregateRanking] = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)

            # Calculate confidence intervals
            ci_lower, ci_upper = bootstrap_confidence_intervals(positions, rng=rng)

            # Item 13: Average normalized Borda scores across ballots
            borda_scores = model_normalized_borda.get(model, [])
            borda_score = round(sum(borda_scores) / len(borda_scores), 2) if borda_scores else 0.0

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

    # Sort by average rank (lower is better), break ties by higher Borda, more rankings, then name
    aggregate.sort(key=lambda x: (x.average_rank, -(x.borda_score or 0), -x.rankings_count, x.model))

    return (aggregate, valid_ballots, total_ballots)
