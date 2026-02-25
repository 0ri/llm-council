"""Output formatting for LLM Council results."""

from __future__ import annotations

from typing import Any

from .models import AggregateRanking, Stage1Result, Stage3Result


def format_output(
    aggregate_rankings: list[AggregateRanking] | list[dict[str, Any]],
    stage3_result: Stage3Result | dict[str, Any],
    valid_ballots: int,
    total_ballots: int,
) -> str:
    """Format the council results as markdown.

    Args:
        aggregate_rankings: List of aggregate rankings (AggregateRanking or dict for backwards compatibility)
        stage3_result: Chairman synthesis result (Stage3Result or dict for backwards compatibility)
        valid_ballots: Number of valid ballots
        total_ballots: Total number of ballots

    Returns:
        Formatted markdown output
    """
    output: list[str] = []
    output.append("## LLM Council Response\n")

    # Rankings table with confidence intervals and Borda scores
    output.append("### Model Rankings (by peer review)\n")
    output.append("| Rank | Model | Avg Position | 95% CI | Borda Score |")
    output.append("|------|-------|--------------|--------|-------------|")

    for i, ranking in enumerate(aggregate_rankings, start=1):
        # Handle both AggregateRanking and dict for backwards compatibility
        if isinstance(ranking, AggregateRanking):
            ci_str = f"[{ranking.ci_lower or 0}, {ranking.ci_upper or 0}]"
            borda = ranking.borda_score or 0
            output.append(f"| {i} | {ranking.model} | {ranking.average_rank} | {ci_str} | {borda} |")
        else:
            ci = ranking.get("confidence_interval", (0, 0))
            ci_str = f"[{ci[0]}, {ci[1]}]"
            borda = ranking.get("borda_score", 0)
            output.append(f"| {i} | {ranking['model']} | {ranking['average_rank']} | {ci_str} | {borda} |")

    # Ballot validity indicator
    if valid_ballots == total_ballots:
        ballot_status = f"*Rankings based on {valid_ballots}/{total_ballots} valid ballots (anonymous peer evaluation)*"
    else:
        ballot_status = (
            f"*Rankings based on {valid_ballots}/{total_ballots} "
            "valid ballots (some rankings could not be parsed reliably)*"
        )

    output.append(f"\n{ballot_status}\n")
    output.append("---\n")

    # Chairman synthesis
    output.append("### Synthesized Answer\n")
    # Handle both Stage3Result and dict for backwards compatibility
    if isinstance(stage3_result, Stage3Result):
        output.append(f"**Chairman:** {stage3_result.model}\n")
        output.append(stage3_result.response)
    else:
        output.append(f"**Chairman:** {stage3_result['model']}\n")
        output.append(stage3_result["response"])

    return "\n".join(output)


def format_stage1_output(
    stage1_results: list[Stage1Result] | list[dict[str, Any]],
) -> str:
    """Format Stage 1 results only (individual responses, no rankings).

    Args:
        stage1_results: List of Stage1 results

    Returns:
        Formatted markdown output
    """
    output: list[str] = []
    output.append("## LLM Council Response (Stage 1 only)\n")

    for result in stage1_results:
        if isinstance(result, Stage1Result):
            name, response = result.model, result.response
        else:
            name, response = result["model"], result["response"]
        output.append(f"### {name}\n")
        output.append(response)
        output.append("")

    return "\n".join(output)


def format_stage2_output(
    aggregate_rankings: list[AggregateRanking] | list[dict[str, Any]],
    stage1_results: list[Stage1Result] | list[dict[str, Any]],
    valid_ballots: int,
    total_ballots: int,
) -> str:
    """Format Stage 1 + Stage 2 results (responses and rankings, no synthesis).

    Args:
        aggregate_rankings: Sorted aggregate ranking data
        stage1_results: Individual model responses
        valid_ballots: Number of valid ballots
        total_ballots: Total number of ballots

    Returns:
        Formatted markdown output
    """
    output: list[str] = []
    output.append("## LLM Council Response (Stages 1-2, no synthesis)\n")

    # Rankings table
    output.append("### Model Rankings (by peer review)\n")
    output.append("| Rank | Model | Avg Position | 95% CI | Borda Score |")
    output.append("|------|-------|--------------|--------|-------------|")

    for i, ranking in enumerate(aggregate_rankings, start=1):
        if isinstance(ranking, AggregateRanking):
            ci_str = f"[{ranking.ci_lower or 0}, {ranking.ci_upper or 0}]"
            borda = ranking.borda_score or 0
            output.append(f"| {i} | {ranking.model} | {ranking.average_rank} | {ci_str} | {borda} |")
        else:
            ci = ranking.get("confidence_interval", (0, 0))
            ci_str = f"[{ci[0]}, {ci[1]}]"
            borda = ranking.get("borda_score", 0)
            output.append(f"| {i} | {ranking['model']} | {ranking['average_rank']} | {ci_str} | {borda} |")

    if valid_ballots == total_ballots:
        ballot_status = f"*Rankings based on {valid_ballots}/{total_ballots} valid ballots (anonymous peer evaluation)*"
    else:
        ballot_status = (
            f"*Rankings based on {valid_ballots}/{total_ballots} "
            "valid ballots (some rankings could not be parsed reliably)*"
        )
    output.append(f"\n{ballot_status}\n")
    output.append("---\n")

    # Individual responses
    output.append("### Individual Responses\n")
    for result in stage1_results:
        if isinstance(result, Stage1Result):
            name, response = result.model, result.response
        else:
            name, response = result["model"], result["response"]
        output.append(f"#### {name}\n")
        output.append(response)
        output.append("")

    return "\n".join(output)
