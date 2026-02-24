"""Output formatting for LLM Council results."""
from __future__ import annotations

from typing import Any


def format_output(
    aggregate_rankings: list[dict[str, Any]],
    stage3_result: dict[str, Any],
    valid_ballots: int,
    total_ballots: int,
) -> str:
    """Format the council results as markdown."""
    output: list[str] = []
    output.append("## LLM Council Response\n")

    # Rankings table
    output.append("### Model Rankings (by peer review)\n")
    output.append("| Rank | Model | Avg Position |")
    output.append("|------|-------|--------------|")

    for i, ranking in enumerate(aggregate_rankings, start=1):
        output.append(f"| {i} | {ranking['model']} | {ranking['average_rank']} |")

    # Ballot validity indicator
    if valid_ballots == total_ballots:
        ballot_status = (
            f"*Rankings based on {valid_ballots}/{total_ballots} "
            "valid ballots (anonymous peer evaluation)*"
        )
    else:
        ballot_status = (
            f"*Rankings based on {valid_ballots}/{total_ballots} "
            "valid ballots (some rankings could not be parsed reliably)*"
        )

    output.append(f"\n{ballot_status}\n")
    output.append("---\n")

    # Chairman synthesis
    output.append("### Synthesized Answer\n")
    output.append(f"**Chairman:** {stage3_result['model']}\n")
    output.append(stage3_result["response"])

    return "\n".join(output)
