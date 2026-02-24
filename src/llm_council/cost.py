"""Token counting and cost estimation for LLM Council runs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("llm-council")


@dataclass
class ModelUsage:
    """Token usage for a single model call."""

    model_name: str
    stage: int
    input_chars: int = 0
    output_chars: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0


@dataclass
class CouncilCostTracker:
    """Tracks token usage and estimates costs across all council stages."""

    usages: list[ModelUsage] = field(default_factory=list)

    def record(self, model_name: str, stage: int, input_text: str, output_text: str) -> None:
        """Record a model interaction's token usage."""
        input_chars = len(input_text)
        output_chars = len(output_text)
        # Rough estimate: ~4 chars per token for English text
        usage = ModelUsage(
            model_name=model_name,
            stage=stage,
            input_chars=input_chars,
            output_chars=output_chars,
            estimated_input_tokens=input_chars // 4,
            estimated_output_tokens=output_chars // 4,
        )
        self.usages.append(usage)

    @property
    def total_input_tokens(self) -> int:
        return sum(u.estimated_input_tokens for u in self.usages)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.estimated_output_tokens for u in self.usages)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def summary(self) -> str:
        """Return a human-readable cost summary."""
        lines = ["", "--- Token Usage Estimate ---"]

        for stage_num in [1, 2, 3]:
            stage_usages = [u for u in self.usages if u.stage == stage_num]
            if not stage_usages:
                continue
            stage_in = sum(u.estimated_input_tokens for u in stage_usages)
            stage_out = sum(u.estimated_output_tokens for u in stage_usages)
            stage_total = stage_in + stage_out
            lines.append(
                f"  Stage {stage_num}: ~{stage_in:,} in + ~{stage_out:,} out"
                f" = ~{stage_total:,} tokens"
            )
            for u in stage_usages:
                lines.append(
                    f"    {u.model_name}: ~{u.estimated_input_tokens:,} in,"
                    f" ~{u.estimated_output_tokens:,} out"
                )

        total_in = self.total_input_tokens
        total_out = self.total_output_tokens
        lines.append(
            f"  Total: ~{total_in:,} in + ~{total_out:,} out"
            f" = ~{self.total_tokens:,} tokens"
        )
        lines.append("---")
        return "\n".join(lines)
