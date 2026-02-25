"""Provider abstraction layer for LLM Council."""

from __future__ import annotations

import asyncio
import time as _time
import typing

from .bedrock import BedrockProvider
from .poe import PoeProvider

# Configurable defaults
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT = 360
DEFAULT_MAX_RETRIES = 2

# Timeout for individual model queries (seconds)
MODEL_TIMEOUT = DEFAULT_TIMEOUT
# Number of retry attempts for transient failures
MAX_RETRIES = DEFAULT_MAX_RETRIES


class Provider(typing.Protocol):
    """Protocol for LLM providers."""

    async def query(self, prompt: str, model_config: dict, timeout: int) -> str: ...


# ---------------------------------------------------------------------------
# Async semaphore for rate-limit protection
# ---------------------------------------------------------------------------

_semaphore: asyncio.Semaphore | None = None


# Deprecated: use CouncilContext instead
def get_semaphore(max_concurrent: int = 4) -> asyncio.Semaphore:
    """Get or create a semaphore for limiting concurrent provider calls."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max_concurrent)
    return _semaphore


# Deprecated: use CouncilContext instead
def reset_semaphore() -> None:
    """Reset the semaphore (for testing)."""
    global _semaphore
    _semaphore = None


# ---------------------------------------------------------------------------
# Circuit breaker pattern
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Simple circuit breaker for provider failure detection.

    After ``failure_threshold`` consecutive failures, the circuit opens
    and rejects calls for ``cooldown_seconds`` before allowing a retry.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._state: str = "closed"  # closed = normal, open = rejecting

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            # Check if cooldown has elapsed
            if _time.time() - self._last_failure_time >= self.cooldown_seconds:
                self._state = "half-open"
                return False
            return True
        return False

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = _time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = "open"


# Per-provider circuit breakers
_circuit_breakers: dict[str, CircuitBreaker] = {}


# Deprecated: use CouncilContext instead
def get_circuit_breaker(identifier: str) -> CircuitBreaker:
    """Get or create a circuit breaker for *identifier*.

    Args:
        identifier: Can be a provider name ("poe", "bedrock") for backward compatibility,
                   or a model-specific key ("poe:GPT-5.3-Codex", "bedrock:claude-opus")
    """
    if identifier not in _circuit_breakers:
        _circuit_breakers[identifier] = CircuitBreaker()
    return _circuit_breakers[identifier]


# Deprecated: use CouncilContext instead
def reset_circuit_breakers() -> None:
    """Reset all circuit breakers (for testing)."""
    _circuit_breakers.clear()


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_providers: dict[str, Provider] = {}


# Deprecated: use CouncilContext instead
def get_provider(provider_name: str, api_key: str | None = None) -> Provider:
    """Get or create a provider instance."""
    if provider_name not in _providers:
        if provider_name == "bedrock":
            _providers[provider_name] = BedrockProvider()
        elif provider_name == "poe":
            if not api_key:
                raise ValueError("POE_API_KEY required for Poe provider")
            _providers[provider_name] = PoeProvider(api_key)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    return _providers[provider_name]


# Deprecated: use CouncilContext instead
def reset_providers() -> None:
    """Clear the provider registry (useful for testing)."""
    _providers.clear()


__all__ = [
    "BedrockProvider",
    "CircuitBreaker",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_REGION",
    "DEFAULT_TIMEOUT",
    "MAX_RETRIES",
    "MODEL_TIMEOUT",
    "Provider",
    "PoeProvider",
    "get_circuit_breaker",
    "get_provider",
    "get_semaphore",
    "reset_circuit_breakers",
    "reset_providers",
    "reset_semaphore",
]
