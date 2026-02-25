"""Pydantic configuration models for LLM Council."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


# Discriminated union on the 'provider' field
ModelConfig = BedrockModelConfig | PoeModelConfig | OpenRouterModelConfig


class CouncilConfig(BaseModel):
    """Top-level council configuration."""

    council_models: list[ModelConfig] = Field(min_length=1)
    chairman: ModelConfig


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
