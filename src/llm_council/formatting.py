"""Output formatting for council results at each pipeline stage.

Exports ``format_output`` (full 3-stage markdown with rankings table and
chairman synthesis), ``format_stage1_output`` (responses only), and
``format_stage2_output`` (responses + rankings, no synthesis). Called by
``run_council`` to produce the final user-facing output.
"""

from __future__ import annotations

from .models import AggregateRanking, Stage1Result, Stage3Result


def format_output(
    aggregate_rankings: list[AggregateRanking],
    stage3_result: Stage3Result,
    valid_ballots: int,
    total_ballots: int,
) -> str:
    """Format the full 3-stage council results as a markdown document.

    Assembles a rankings table (average position, 95 % confidence
    interval, Borda score), a ballot-validity indicator, and the
    chairman's synthesis into a single markdown string ready for
    display.

    Args:
        aggregate_rankings: Aggregate ranking entries sorted from best
            to worst.  Each entry carries the model name, average rank,
            confidence-interval bounds, and Borda score.
        stage3_result: The chairman's Stage 3 synthesis containing the
            model name and synthesized response text.
        valid_ballots: Number of Stage 2 ballots that were successfully
            parsed.
        total_ballots: Total number of Stage 2 ballots attempted
            (including unparseable ones).

    Returns:
        A markdown-formatted string containing the rankings table,
        ballot validity note, and chairman synthesis, joined by
        newlines.
    """
    output: list[str] = []
    output.append("## LLM Council Response\n")

    # Rankings table with confidence intervals and Borda scores
    output.append("### Model Rankings (by peer review)\n")
    output.append("| Rank | Model | Avg Position | 95% CI | Borda Score |")
    output.append("|------|-------|--------------|--------|-------------|")

    for i, ranking in enumerate(aggregate_rankings, start=1):
        ci_str = f"[{ranking.ci_lower or 0}, {ranking.ci_upper or 0}]"
        borda = ranking.borda_score or 0
        output.append(f"| {i} | {ranking.model} | {ranking.average_rank} | {ci_str} | {borda} |")

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
    output.append(f"**Chairman:** {stage3_result.model}\n")
    output.append(stage3_result.response)

    return "\n".join(output)


def format_stage1_output(
    stage1_results: list[Stage1Result],
) -> str:
    """Format Stage 1 results only (individual responses, no rankings)."""
    output: list[str] = []
    output.append("## LLM Council Response (Stage 1 only)\n")

    for result in stage1_results:
        output.append(f"### {result.model}\n")
        output.append(result.response)
        output.append("")

    return "\n".join(output)


def format_stage2_output(
    aggregate_rankings: list[AggregateRanking],
    stage1_results: list[Stage1Result],
    valid_ballots: int,
    total_ballots: int,
) -> str:
    """Format Stage 1 + Stage 2 results (responses and rankings, no synthesis)."""
    output: list[str] = []
    output.append("## LLM Council Response (Stages 1-2, no synthesis)\n")

    # Rankings table
    output.append("### Model Rankings (by peer review)\n")
    output.append("| Rank | Model | Avg Position | 95% CI | Borda Score |")
    output.append("|------|-------|--------------|--------|-------------|")

    for i, ranking in enumerate(aggregate_rankings, start=1):
        ci_str = f"[{ranking.ci_lower or 0}, {ranking.ci_upper or 0}]"
        borda = ranking.borda_score or 0
        output.append(f"| {i} | {ranking.model} | {ranking.average_rank} | {ci_str} | {borda} |")

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
        output.append(f"#### {result.model}\n")
        output.append(result.response)
        output.append("")

    return "\n".join(output)
