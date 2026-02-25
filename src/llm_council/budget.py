"""Budget controls for LLM Council to prevent excessive token usage and cost."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("llm-council")


class BudgetExceededError(Exception):
    """Raised when budget limits are exceeded."""

    pass


@dataclass
class BudgetGuard:
    """Tracks and enforces token usage and cost limits.

    Raises BudgetExceededError if limits are exceeded.
    """

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    # Default pricing per 1K tokens (configurable)
    input_cost_per_1k: float = 0.01
    output_cost_per_1k: float = 0.03

    # Tracking
    total_input_tokens: int = field(default=0, init=False)
    total_output_tokens: int = field(default=0, init=False)
    total_cost_usd: float = field(default=0.0, init=False)
    queries: list[dict[str, Any]] = field(default_factory=list, init=False)

    def can_afford(
        self,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        model_name: str,
    ) -> None:
        """Pre-flight check: would this query exceed the budget?

        Does NOT mutate state. Call ``commit()`` after a successful query.

        Raises:
            BudgetExceededError: If the query would exceed budget limits.
        """
        projected_input = self.total_input_tokens + estimated_input_tokens
        projected_output = self.total_output_tokens + estimated_output_tokens
        projected_total = projected_input + projected_output

        query_cost = (
            estimated_input_tokens / 1000 * self.input_cost_per_1k
            + estimated_output_tokens / 1000 * self.output_cost_per_1k
        )
        projected_cost = self.total_cost_usd + query_cost

        if self.max_tokens is not None and projected_total > self.max_tokens:
            logger.error(f"Budget exceeded: {projected_total} tokens > {self.max_tokens} limit. Model: {model_name}")
            raise BudgetExceededError(
                f"Token budget exceeded: {projected_total} > {self.max_tokens} (limit). "
                f"Query would add {estimated_input_tokens + estimated_output_tokens} tokens."
            )

        if self.max_cost_usd is not None and projected_cost > self.max_cost_usd:
            logger.error(
                f"Budget exceeded: ${projected_cost:.2f} > ${self.max_cost_usd:.2f} limit. Model: {model_name}"
            )
            raise BudgetExceededError(
                f"Cost budget exceeded: ${projected_cost:.2f} > ${self.max_cost_usd:.2f} (limit). "
                f"Query would add ${query_cost:.2f}."
            )

        logger.debug(
            f"Budget preflight OK for {model_name}: "
            f"{projected_total}/{self.max_tokens or 'unlimited'} tokens, "
            f"${projected_cost:.2f}/${self.max_cost_usd or 'unlimited'}"
        )

    def commit(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
    ) -> None:
        """Record actual token usage after a successful query."""
        query_cost = input_tokens / 1000 * self.input_cost_per_1k + output_tokens / 1000 * self.output_cost_per_1k

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += query_cost

        self.queries.append(
            {
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": query_cost,
            }
        )

        logger.debug(f"Budget committed for {model_name}: {input_tokens} in + {output_tokens} out, ${query_cost:.4f}")

    def check_and_update(
        self,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        model_name: str,
    ) -> None:
        """Legacy: preflight + immediate commit (kept for backwards compat)."""
        self.can_afford(estimated_input_tokens, estimated_output_tokens, model_name)
        self.commit(estimated_input_tokens, estimated_output_tokens, model_name)

    def reset(self) -> None:
        """Reset all tracking counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.queries.clear()

    def summary(self) -> str:
        """Return a human-readable budget summary."""
        lines = ["--- Budget Summary ---"]

        # Token usage
        total_tokens = self.total_input_tokens + self.total_output_tokens
        if self.max_tokens:
            pct_tokens = (total_tokens / self.max_tokens) * 100
            lines.append(f"Tokens: {total_tokens:,}/{self.max_tokens:,} ({pct_tokens:.1f}% used)")
        else:
            lines.append(f"Tokens: {total_tokens:,} (no limit)")

        lines.append(f"  Input: {self.total_input_tokens:,}")
        lines.append(f"  Output: {self.total_output_tokens:,}")

        # Cost
        if self.max_cost_usd:
            pct_cost = (self.total_cost_usd / self.max_cost_usd) * 100
            lines.append(f"Cost: ${self.total_cost_usd:.2f}/${self.max_cost_usd:.2f} ({pct_cost:.1f}% used)")
        else:
            lines.append(f"Cost: ${self.total_cost_usd:.2f} (no limit)")

        # Query count
        lines.append(f"Queries: {len(self.queries)}")

        return "\n".join(lines)


def create_budget_guard(config: dict[str, Any]) -> BudgetGuard | None:
    """Create a BudgetGuard from config, or None if no budget configured.

    Args:
        config: Council configuration dict

    Returns:
        BudgetGuard instance or None if no budget limits configured
    """
    budget_config = config.get("budget", {})
    if not budget_config:
        return None

    max_tokens = budget_config.get("max_tokens")
    max_cost_usd = budget_config.get("max_cost_usd")

    if max_tokens is None and max_cost_usd is None:
        return None

    # Use custom pricing if provided, otherwise defaults
    input_cost = budget_config.get("input_cost_per_1k", 0.01)
    output_cost = budget_config.get("output_cost_per_1k", 0.03)

    return BudgetGuard(
        max_tokens=max_tokens,
        max_cost_usd=max_cost_usd,
        input_cost_per_1k=input_cost,
        output_cost_per_1k=output_cost,
    )
