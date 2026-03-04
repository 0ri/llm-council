"""Pydantic data models for council configuration and pipeline results.

Defines ``CouncilConfig`` (top-level config with discriminated-union model
entries), provider-specific configs (``BedrockModelConfig``,
``PoeModelConfig``, ``OpenRouterModelConfig``), ``BudgetConfig``, and stage
result types (``Stage1Result``, ``Stage2Result``, ``Stage3Result``,
``AggregateRanking``).
"""

from __future__ import annotations

import os
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .defaults import DEFAULT_CACHE_TTL, DEFAULT_SOFT_TIMEOUT, DEFAULT_STAGE2_RETRIES


def generate_response_labels(count: int) -> list[str]:
    """Generate anonymous response labels: 'Response A', 'Response B', etc."""
    if count > 26:
        raise ValueError(f"Max 26 labels supported, got {count}")
    return [f"Response {chr(65 + i)}" for i in range(count)]


def generate_letter_labels(count: int) -> list[str]:
    """Generate single-letter labels: 'A', 'B', 'C', etc."""
    if count > 26:
        raise ValueError(f"Max 26 labels supported, got {count}")
    return [chr(65 + i) for i in range(count)]


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


def get_model_identifier(config: ModelConfig) -> str:
    """Extract the provider-specific model identifier from any ModelConfig type."""
    return getattr(config, "model_id", getattr(config, "bot_name", config.name))


_PROVIDER_TO_CONFIG: dict[str, type[BaseModel]] = {
    "bedrock": BedrockModelConfig,
    "poe": PoeModelConfig,
    "openrouter": OpenRouterModelConfig,
}


def coerce_model_config(config: ModelConfig | dict) -> ModelConfig:
    """Convert a dict to the appropriate ModelConfig subtype if needed.

    Passes through existing ModelConfig instances unchanged. Accepts
    plain dicts (as used by tests and JSON config) and returns the
    correct discriminated-union member based on the ``provider`` field.
    When ``provider`` is absent, infers it from available keys.
    """
    if isinstance(config, (BedrockModelConfig, PoeModelConfig, OpenRouterModelConfig)):
        return config
    if isinstance(config, dict):
        clean = {k: v for k, v in config.items() if not k.startswith("_")}
        provider = clean.get("provider")
        # Infer provider from available keys when not explicitly set
        if provider is None:
            if "bot_name" in clean:
                provider = "poe"
            elif "model_id" in clean:
                provider = "bedrock"
            else:
                provider = "poe"
            clean["provider"] = provider
        # Ensure name field exists (some test dicts omit it)
        if "name" not in clean:
            clean["name"] = clean.get("model_id", clean.get("bot_name", "unknown"))
        # Ensure required provider-specific identifiers exist
        if provider == "bedrock" and "model_id" not in clean:
            clean["model_id"] = clean["name"]
        elif provider == "poe" and "bot_name" not in clean:
            clean["bot_name"] = clean["name"]
        elif provider == "openrouter" and "model_id" not in clean:
            clean["model_id"] = clean["name"]
        cls = _PROVIDER_TO_CONFIG.get(provider)
        if cls is None:
            raise ValueError(f"Unknown provider: {provider}")
        return cls(**clean)
    return config


class BudgetConfig(BaseModel):
    """Budget controls for limiting token usage and cost."""

    max_tokens: Annotated[int, Field(gt=0)] | None = None
    max_cost_usd: Annotated[float, Field(gt=0)] | None = None
    input_cost_per_1k: Annotated[float, Field(ge=0)] = 0.01
    output_cost_per_1k: Annotated[float, Field(ge=0)] = 0.03


class PromptConfig(BaseModel):
    """Custom prompt templates for Stage 2 ranking and Stage 3 synthesis.

    All fields are optional — omitted fields use the built-in defaults.
    Templates must include the required ``{placeholders}`` documented in
    ``docs/examples/prompts/README.md``.

    Inline strings or file paths are both supported:
    - ``ranking_system``: inline template string
    - ``ranking_system_file``: path to a text file containing the template

    If both inline and file variants are set, the inline string wins.
    """

    ranking_system: str | None = None
    ranking_system_file: str | None = None
    ranking_user: str | None = None
    ranking_user_file: str | None = None
    synthesis_system: str | None = None
    synthesis_system_file: str | None = None
    synthesis_user: str | None = None
    synthesis_user_file: str | None = None

    def resolve(self, field: str) -> str | None:
        """Resolve a prompt field, preferring inline over file."""
        from pathlib import Path

        inline = getattr(self, field, None)
        if inline is not None:
            return inline
        file_path = getattr(self, f"{field}_file", None)
        if file_path is not None:
            return Path(file_path).read_text(encoding="utf-8").strip()
        return None


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
    chairman: ModelConfig | None = None
    budget: BudgetConfig | None = None
    prompts: PromptConfig | None = None
    cache_ttl: int = DEFAULT_CACHE_TTL
    soft_timeout: float = DEFAULT_SOFT_TIMEOUT
    min_responses: int | None = None
    stage2_retries: int = Field(default=DEFAULT_STAGE2_RETRIES, ge=0, le=5)
    min_valid_ballots: int | None = None
    strict_ballots: bool = False

    @classmethod
    def validate_from_dict(cls, config: dict) -> list[str]:
        """Validate a raw config dict and return human-readable error strings.

        Delegates structural validation to Pydantic, then performs
        environment-variable checks that Pydantic cannot handle.

        Returns:
            A list of error strings. Empty means the configuration is valid.
        """
        errors: list[str] = []

        # --- Pydantic structural validation ---
        try:
            cls(**config)
        except ValidationError as e:
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]

                if error["type"] == "missing":
                    if loc == "council_models":
                        errors.append("Config missing 'council_models' list")
                    else:
                        errors.append(f"Missing required field: {loc}")
                elif error["type"] in ["list_min_length", "too_short"] and "council_models" in loc:
                    errors.append("'council_models' must be a non-empty list")
                elif error["type"] == "literal_error" and "provider" in loc:
                    if "council_models" in loc:
                        idx = loc.split(".")[1] if "." in loc else "0"
                        idx_int = int(idx) if idx.isdigit() else 0
                        model_name = config.get("council_models", [{}])[idx_int].get("name", f"model[{idx}]")
                    else:
                        model_name = config.get("chairman", {}).get("name", "chairman")
                    errors.append(f"{model_name}: unknown provider (must be one of: bedrock, poe, openrouter)")
                elif "budget_tokens" in loc and error["type"] in ("greater_than_equal", "less_than_equal"):
                    if "council_models" in loc:
                        idx = int(loc.split(".")[1]) if "." in loc and loc.split(".")[1].isdigit() else 0
                        model_name = config.get("council_models", [{}])[idx].get("name", f"model[{idx}]")
                    else:
                        model_name = config.get("chairman", {}).get("name", "chairman")
                    errors.append(f"{model_name}: 'budget_tokens' must be integer between 1024 and 128000")
                elif error["type"] == "union_tag_invalid":
                    if "council_models" in loc:
                        idx = int(loc.split(".")[1]) if "." in loc and loc.split(".")[1].isdigit() else 0
                        model_name = config.get("council_models", [{}])[idx].get("name", f"model[{idx}]")
                    else:
                        model_name = config.get("chairman", {}).get("name", "chairman")
                    input_str = str(error.get("input", {}))
                    if "model_id" in input_str:
                        errors.append(f"{model_name}: Bedrock provider requires 'model_id'")
                    elif "bot_name" in input_str:
                        errors.append(f"{model_name}: Poe provider requires 'bot_name'")
                    else:
                        errors.append(f"{loc}: {msg}")
                else:
                    errors.append(f"{loc}: {msg}")

        # --- Check for missing name fields (union dispatch can mask this) ---
        all_models = list(config.get("council_models", []))
        if config.get("chairman"):
            all_models.append(config["chairman"])
        for i, model in enumerate(all_models):
            if "name" not in model:
                errors.append(f"Model at index {i} missing 'name'")

        # --- Environment variable checks (Pydantic can't do these) ---
        poe_models = [m for m in config.get("council_models", []) if m.get("provider") == "poe"]
        chairman_cfg = config.get("chairman") or {}
        if chairman_cfg.get("provider") == "poe":
            poe_models.append(config["chairman"])
        if poe_models and not os.environ.get("POE_API_KEY"):
            errors.append("POE_API_KEY environment variable required for Poe provider models")

        or_models = [m for m in config.get("council_models", []) if m.get("provider") == "openrouter"]
        if chairman_cfg.get("provider") == "openrouter":
            or_models.append(config["chairman"])
        if or_models and not os.environ.get("OPENROUTER_API_KEY"):
            errors.append("OPENROUTER_API_KEY environment variable required for OpenRouter provider models")

        # Deduplicate preserving order
        seen: set[str] = set()
        return [e for e in errors if not (e in seen or seen.add(e))]

    @model_validator(mode="after")
    def _validate_council_models(self) -> CouncilConfig:
        if len(self.council_models) > 26:
            raise ValueError(f"Council supports a maximum of 26 models (Response A-Z), got {len(self.council_models)}")
        names = [m.name for m in self.council_models]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Duplicate model names in council_models: {', '.join(duplicates)}")
        return self


class Stage1Result(BaseModel):
    """Result from a single model in Stage 1."""

    model: str
    response: str = Field(min_length=1)


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
    def confidence_interval(self) -> tuple[float, float] | None:
        """Get confidence interval as a tuple, or None if not computed."""
        if self.ci_lower is None or self.ci_upper is None:
            return None
        return (self.ci_lower, self.ci_upper)


class Stage3Result(BaseModel):
    """Result from the chairman synthesis."""

    model: str
    response: str
