"""Provider abstraction layer for LLM Council.

Defines the ``Provider`` and ``StreamingProvider`` protocols that all LLM
backends must implement, along with shared infrastructure: ``CircuitBreaker``
for failure detection, ``StreamResult`` for streaming responses, configurable
timeout/retry defaults, and the ``fallback_astream`` helper. Concrete
implementations live in the ``bedrock``, ``openrouter``, and ``poe`` sub-modules.
"""

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
    """Define the interface that all LLM provider backends must implement.

    Every concrete provider (Bedrock, OpenRouter, Poe) satisfies this
    protocol by implementing the ``query`` method. The council pipeline
    dispatches Stage 1 and Stage 2 calls through this interface, passing
    a prompt string, provider-specific model configuration, and a timeout.

    Implementations should return the model's text response along with
    optional usage metadata (token counts, cost estimates). If the
    provider cannot fulfil the request, it should raise an appropriate
    exception rather than returning an empty string.
    """

    async def query(
        self, prompt: str, model_config: dict, timeout: int
    ) -> tuple[str, dict | None]:
        """Send a prompt to the language model and return its response.

        Args:
            prompt: The full prompt text to send to the model, including
                any system instructions and user content.
            model_config: Provider-specific configuration dictionary.
                Bedrock configs include ``model_id`` and ``budget_tokens``;
                Poe configs include ``bot_name``, ``web_search``, and
                ``reasoning_effort``; OpenRouter configs include
                ``model_id``, ``temperature``, and ``max_tokens``.
            timeout: Maximum number of seconds to wait for a response
                before raising a timeout error.

        Returns:
            A tuple of (response_text, usage_metadata). ``response_text``
            is the model's generated text. ``usage_metadata`` is an
            optional dictionary containing token counts and cost
            information (e.g. ``input_tokens``, ``output_tokens``), or
            ``None`` if the provider does not report usage data.

        Raises:
            TimeoutError: If the provider does not respond within the
                specified timeout.
            RuntimeError: If the provider encounters an unrecoverable
                error (e.g. invalid credentials, model not found).
        """
        ...


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
    """Extend ``Provider`` with token-by-token streaming support.

    Providers that implement this protocol can deliver Stage 3 synthesis
    output incrementally, allowing the CLI to display tokens as they
    arrive rather than waiting for the full response. The
    ``fallback_astream`` helper wraps any plain ``Provider`` as a
    single-chunk stream, so callers can always use the streaming
    interface regardless of provider capability.

    A class satisfies this protocol by implementing both ``query``
    (inherited from ``Provider``) and ``astream``.
    """

    def astream(
        self, prompt: str, model_config: dict, timeout: int
    ) -> StreamResult:
        """Stream a prompt response as an async iterator of text chunks.

        Args:
            prompt: The full prompt text to send to the model, including
                any system instructions and user content.
            model_config: Provider-specific configuration dictionary.
                See ``Provider.query`` for the expected keys per
                provider type.
            timeout: Maximum number of seconds to wait for the stream
                to begin producing chunks before raising a timeout
                error.

        Returns:
            A ``StreamResult`` instance that can be async-iterated to
            receive text chunks. The ``StreamResult.accumulated``
            attribute collects the full response, and
            ``StreamResult.usage`` is populated with token/cost
            metadata once the stream completes.

        Raises:
            TimeoutError: If the provider does not begin streaming
                within the specified timeout.
            RuntimeError: If the provider encounters an unrecoverable
                error (e.g. invalid credentials, model not found).
        """
        ...


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
