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


# Discriminated union on the 'provider' field
ModelConfig = BedrockModelConfig | PoeModelConfig


class CouncilConfig(BaseModel):
    """Top-level council configuration."""

    council_models: list[ModelConfig] = Field(min_length=1)
    chairman: ModelConfig
