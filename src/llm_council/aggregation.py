"""Ranking aggregation logic for LLM Council."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def calculate_aggregate_rankings(
    stage2_results: list[dict[str, Any]],
    label_to_model: dict[str, str],
) -> tuple[list[dict[str, Any]], int, int]:
    """Calculate aggregate rankings across all models.

    Returns:
        Tuple of (aggregate rankings list, valid ballot count, total ballot count)
    """
    model_positions: dict[str, list[int]] = defaultdict(list)
    valid_ballots = 0
    total_ballots = len(stage2_results)

    for ranking in stage2_results:
        # Use pre-parsed ranking from stage2
        parsed_ranking = ranking.get("parsed_ranking", [])
        is_valid = ranking.get("is_valid_ballot", False)

        if is_valid:
            valid_ballots += 1

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate: list[dict[str, Any]] = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({"model": model, "average_rank": round(avg_rank, 2), "rankings_count": len(positions)})

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x["average_rank"])

    return (aggregate, valid_ballots, total_ballots)
