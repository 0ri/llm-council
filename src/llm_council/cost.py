"""Token counting and cost estimation for council runs.

Exports ``estimate_tokens`` (tiktoken or character heuristic),
``ModelUsage`` for per-call tracking, and ``CouncilCostTracker`` which
accumulates usage across all three pipeline stages and produces a
human-readable summary with actual vs. estimated token breakdowns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("llm-council")

_ENCODER = None
HAS_TIKTOKEN = False
_TIKTOKEN_LOADED = False


def _get_encoder():
    """Lazy-load tiktoken encoder on first use."""
    global _ENCODER, HAS_TIKTOKEN, _TIKTOKEN_LOADED
    if not _TIKTOKEN_LOADED:
        _TIKTOKEN_LOADED = True
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
            HAS_TIKTOKEN = True
        except ImportError:
            _ENCODER = None
            HAS_TIKTOKEN = False
    return _ENCODER


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses tiktoken cl100k_base encoding when available.
    Falls back to 4 chars/token heuristic.
    """
    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return len(text) // 4


@dataclass
class ModelUsage:
    """Token usage for a single model call."""

    model_name: str
    stage: int
    input_chars: int = 0
    output_chars: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None

    @property
    def input_tokens(self) -> int:
        """Return actual tokens if available, otherwise estimated."""
        return self.actual_input_tokens if self.actual_input_tokens is not None else self.estimated_input_tokens

    @property
    def output_tokens(self) -> int:
        """Return actual tokens if available, otherwise estimated."""
        return self.actual_output_tokens if self.actual_output_tokens is not None else self.estimated_output_tokens


@dataclass
class CouncilCostTracker:
    """Tracks token usage and estimates costs across all council stages."""

    usages: list[ModelUsage] = field(default_factory=list)

    def record(
        self,
        model_name: str,
        stage: int,
        input_text: str,
        output_text: str,
        actual_input_tokens: int | None = None,
        actual_output_tokens: int | None = None,
    ) -> None:
        """Record a model interaction's token usage.

        Args:
            model_name: Name of the model
            stage: Stage number (1, 2, or 3)
            input_text: Input text for character counting
            output_text: Output text for character counting
            actual_input_tokens: Actual input token count from API (if available)
            actual_output_tokens: Actual output token count from API (if available)
        """
        input_chars = len(input_text)
        output_chars = len(output_text)
        usage = ModelUsage(
            model_name=model_name,
            stage=stage,
            input_chars=input_chars,
            output_chars=output_chars,
            estimated_input_tokens=estimate_tokens(input_text),
            estimated_output_tokens=estimate_tokens(output_text),
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
        )
        self.usages.append(usage)

    def record_with_usage(
        self,
        model_name: str,
        stage: int,
        input_text: str,
        output_text: str,
        token_usage: dict[str, Any] | None,
    ) -> None:
        """Record a model interaction with optional token usage from API.

        Args:
            model_name: Name of the model
            stage: Stage number (1, 2, or 3)
            input_text: Input text for character counting
            output_text: Output text for character counting
            token_usage: Dict with 'input_tokens' and 'output_tokens' keys (if available)
        """
        actual_input = None
        actual_output = None
        if token_usage:
            actual_input = token_usage.get("input_tokens")
            actual_output = token_usage.get("output_tokens")

        self.record(
            model_name=model_name,
            stage=stage,
            input_text=input_text,
            output_text=output_text,
            actual_input_tokens=actual_input,
            actual_output_tokens=actual_output,
        )

    @property
    def total_input_tokens(self) -> int:
        """Sum input tokens across all recorded usages."""
        return sum(u.input_tokens for u in self.usages)

    @property
    def total_output_tokens(self) -> int:
        """Sum output tokens across all recorded usages."""
        return sum(u.output_tokens for u in self.usages)

    @property
    def total_tokens(self) -> int:
        """Sum all input and output tokens across all recorded usages."""
        return self.total_input_tokens + self.total_output_tokens

    def summary(self) -> str:
        """Return a human-readable cost summary."""
        lines = ["", "--- Token Usage ---"]

        # Track whether we have any actual token counts
        has_actual = any(u.actual_input_tokens is not None or u.actual_output_tokens is not None for u in self.usages)

        for stage_num in [1, 2, 3]:
            stage_usages = [u for u in self.usages if u.stage == stage_num]
            if not stage_usages:
                continue
            stage_in = sum(u.input_tokens for u in stage_usages)
            stage_out = sum(u.output_tokens for u in stage_usages)
            stage_total = stage_in + stage_out

            # Check if this stage has actual counts
            stage_has_actual = any(
                u.actual_input_tokens is not None or u.actual_output_tokens is not None for u in stage_usages
            )
            prefix = "" if stage_has_actual else "~"

            lines.append(
                f"  Stage {stage_num}: {prefix}{stage_in:,} in + {prefix}{stage_out:,} out"
                f" = {prefix}{stage_total:,} tokens"
            )
            for u in stage_usages:
                # Mark individual models as estimated or actual
                in_prefix = "" if u.actual_input_tokens is not None else "~"
                out_prefix = "" if u.actual_output_tokens is not None else "~"
                lines.append(
                    f"    {u.model_name}: {in_prefix}{u.input_tokens:,} in, {out_prefix}{u.output_tokens:,} out"
                )

        total_in = self.total_input_tokens
        total_out = self.total_output_tokens
        # Use ~ prefix only if some tokens are estimated
        prefix = "" if not has_actual else ("~" if any(u.actual_input_tokens is None for u in self.usages) else "")
        lines.append(
            f"  Total: {prefix}{total_in:,} in + {prefix}{total_out:,} out = {prefix}{self.total_tokens:,} tokens"
        )
        if has_actual:
            lines.append("  (~ indicates estimated tokens, actual counts used where available)")
        else:
            lines.append("  (All token counts are estimates based on character count)")
        lines.append("---")
        return "\n".join(lines)
