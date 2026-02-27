"""Pydantic data models for council configuration and pipeline results.

Defines ``CouncilConfig`` (top-level config with discriminated-union model
entries), provider-specific configs (``BedrockModelConfig``,
``PoeModelConfig``, ``OpenRouterModelConfig``), ``BudgetConfig``, and stage
result types (``Stage1Result``, ``Stage2Result``, ``Stage3Result``,
``AggregateRanking``).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BedrockModelConfig(BaseModel):
    """Configuration for an AWS Bedrock model."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    provider: Literal["bedrock"]
    model_id: str
    budget_tokens: int | None = Field(None, ge=1024, le=128000)


class PoeModelConfig(BaseModel):
    """Configuration for a Poe.com model."""

    name: str
    provider: Literal["poe"]
    bot_name: str
    web_search: bool = False
    reasoning_effort: Literal["medium", "high", "Xhigh", "minimal", "low"] | None = None


class OpenRouterModelConfig(BaseModel):
    """Configuration for an OpenRouter model."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    provider: Literal["openrouter"]
    model_id: str
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: Literal["xhigh", "high", "medium", "low", "minimal"] | None = None
    reasoning_max_tokens: int | None = None


# Discriminated union on the 'provider' field
ModelConfig = BedrockModelConfig | PoeModelConfig | OpenRouterModelConfig


class BudgetConfig(BaseModel):
    """Budget controls for limiting token usage and cost."""

    max_tokens: Annotated[int, Field(gt=0)] | None = None
    max_cost_usd: Annotated[float, Field(gt=0)] | None = None
    input_cost_per_1k: Annotated[float, Field(ge=0)] = 0.01
    output_cost_per_1k: Annotated[float, Field(ge=0)] = 0.03


class CouncilConfig(BaseModel):
    """Top-level council configuration for a deliberation run.

    Wraps the list of council member models, the designated chairman
    model, and operational settings (budget, caching, timeouts).
    Typically constructed from a ``council-config.json`` file via
    ``CouncilConfig(**json.load(f))``.

    Raises:
        pydantic.ValidationError: If *council_models* is empty, a
            required field is missing, or a provider-specific constraint
            is violated (e.g. ``budget_tokens`` out of range).
    """

    council_models: list[ModelConfig] = Field(min_length=1)
    chairman: ModelConfig
    budget: BudgetConfig | None = None
    cache_ttl: int = 86400
    soft_timeout: float = 300
    min_responses: int | None = None
    stage2_retries: int = 1

    @model_validator(mode="after")
    def _validate_council_models(self) -> CouncilConfig:
        if len(self.council_models) > 26:
            raise ValueError(
                "Council supports a maximum of 26 models (Response A-Z), "
                f"got {len(self.council_models)}"
            )
        names = [m.name for m in self.council_models]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate model names in council_models: {', '.join(duplicates)}"
            )
        return self


class Stage1Result(BaseModel):
    """Result from a single model in Stage 1."""

    model: str
    response: str


class Stage2Result(BaseModel):
    """Result from a single model in Stage 2."""

    model: str
    ranking: str
    parsed_ranking: list[str]
    is_valid_ballot: bool


class AggregateRanking(BaseModel):
    """Aggregate ranking for a single model."""

    model: str
    average_rank: float
    rankings_count: int
    ci_lower: float | None = None
    ci_upper: float | None = None
    borda_score: float | None = None

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Get confidence interval as a tuple."""
        return (self.ci_lower or 0, self.ci_upper or 0)


class Stage3Result(BaseModel):
    """Result from the chairman synthesis."""

    model: str
    response: str
