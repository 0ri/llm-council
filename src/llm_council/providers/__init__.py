"""Provider abstraction layer for LLM Council."""

from __future__ import annotations

import time as _time
import typing

from .bedrock import BedrockProvider
from .openrouter import OpenRouterProvider
from .poe import PoeProvider

# Configurable defaults
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT = 360
DEFAULT_SOFT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 2

# Timeout for individual model queries (seconds)
MODEL_TIMEOUT = DEFAULT_TIMEOUT
# Soft timeout for parallel queries before proceeding with partial results
SOFT_TIMEOUT = DEFAULT_SOFT_TIMEOUT
# Number of retry attempts for transient failures
MAX_RETRIES = DEFAULT_MAX_RETRIES


class Provider(typing.Protocol):
    """Protocol for LLM providers."""

    async def query(self, prompt: str, model_config: dict, timeout: int) -> tuple[str, dict | None]: ...


class StreamResult:
    """Wrapper for streaming results that captures usage metadata."""

    def __init__(self, aiter: typing.AsyncIterator[str]):
        self._aiter = aiter
        self.usage: dict[str, typing.Any] | None = None
        self.accumulated: str = ""

    def __aiter__(self) -> StreamResult:
        return self

    async def __anext__(self) -> str:
        chunk = await self._aiter.__anext__()
        self.accumulated += chunk
        return chunk


@typing.runtime_checkable
class StreamingProvider(Provider, typing.Protocol):
    """Protocol for providers that support streaming."""

    def astream(self, prompt: str, model_config: dict, timeout: int) -> StreamResult: ...


def fallback_astream(provider: Provider, prompt: str, model_config: dict, timeout: int) -> StreamResult:
    """Create a StreamResult that falls back to query() for non-streaming providers."""

    async def _single_chunk() -> typing.AsyncIterator[str]:
        text, usage = await provider.query(prompt, model_config, timeout)
        yield text

    result = StreamResult(_single_chunk())
    return result


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


__all__ = [
    "BedrockProvider",
    "CircuitBreaker",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_REGION",
    "DEFAULT_TIMEOUT",
    "MAX_RETRIES",
    "MODEL_TIMEOUT",
    "SOFT_TIMEOUT",
    "OpenRouterProvider",
    "PoeProvider",
    "Provider",
    "StreamResult",
    "StreamingProvider",
    "fallback_astream",
]
