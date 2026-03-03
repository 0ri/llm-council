"""Provider abstraction layer for LLM Council.

Defines the ``Provider`` and ``StreamingProvider`` protocols that all LLM
backends must implement, along with shared infrastructure: ``CircuitBreaker``
for failure detection, ``StreamResult`` for streaming responses, configurable
timeout/retry defaults, and the ``fallback_astream`` helper. Concrete
implementations live in the ``bedrock``, ``openrouter``, and ``poe`` sub-modules.
"""

from __future__ import annotations

import dataclasses
import time as _time
import typing

if typing.TYPE_CHECKING:
    from ..models import ModelConfig

@dataclasses.dataclass(slots=True)
class ProviderRequest:
    """Typed request object passed alongside model_config to providers.

    Replaces the old pattern of injecting ``_messages``, ``_system_message``,
    and ``_skip_flags`` as private keys into the model config dict.
    """

    messages: list[dict[str, str]]
    system_message: str | None = None
    suppress_provider_flags: bool = False


# Configurable defaults
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TIMEOUT = 600
DEFAULT_SOFT_TIMEOUT = 480
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
    a prompt string, a typed ``ModelConfig`` with provider-specific
    parameters, a timeout, and a typed ``ProviderRequest`` containing
    messages and metadata.

    Implementations should return the model's text response along with
    optional usage metadata (token counts, cost estimates). If the
    provider cannot fulfil the request, it should raise an appropriate
    exception rather than returning an empty string.
    """

    async def query(
        self,
        prompt: str,
        model_config: ModelConfig,
        timeout: int,
        request: ProviderRequest | None = None,
    ) -> tuple[str, dict | None]:
        """Send a prompt to the language model and return its response.

        Args:
            prompt: The full prompt text to send to the model, including
                any system instructions and user content.
            model_config: Typed ``ModelConfig`` (discriminated union) with
                provider-specific attributes. Bedrock configs have
                ``model_id`` and ``budget_tokens``; Poe configs have
                ``bot_name``, ``web_search``, and ``reasoning_effort``;
                OpenRouter configs have ``model_id``, ``temperature``,
                and ``max_tokens``.
            timeout: Maximum number of seconds to wait for a response
                before raising a timeout error.
            request: Optional typed request containing messages,
                system_message, and flags. When provided, providers
                should use these instead of extracting from model_config.

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


class UsageTrackingStream:
    """Wraps an async generator and captures usage metadata on exhaustion.

    Unlike monkeypatching ``__anext__`` on an async generator (which is
    unreliable), this class IS the async iterator, so ``__anext__`` is a
    proper method. Providers call ``set_usage()`` as usage info arrives
    during streaming, and when the inner iterator is exhausted the usage
    is copied onto the ``StreamResult``.
    """

    def __init__(self, aiter: typing.AsyncIterator[str], result: StreamResult):
        self._aiter = aiter
        self._result = result
        self._usage: dict[str, typing.Any] | None = None

    def set_usage(self, usage: dict[str, typing.Any]) -> None:
        self._usage = usage

    def __aiter__(self) -> UsageTrackingStream:
        return self

    async def __anext__(self) -> str:
        try:
            return await self._aiter.__anext__()
        except StopAsyncIteration:
            if self._usage:
                self._result.usage = self._usage
            raise


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
        self,
        prompt: str,
        model_config: ModelConfig,
        timeout: int,
        request: ProviderRequest | None = None,
    ) -> StreamResult:
        """Stream a prompt response as an async iterator of text chunks.

        Args:
            prompt: The full prompt text to send to the model, including
                any system instructions and user content.
            model_config: Typed ``ModelConfig`` with provider-specific
                attributes. See ``Provider.query`` for details.
            timeout: Maximum number of seconds to wait for the stream
                to begin producing chunks before raising a timeout
                error.
            request: Optional typed request containing messages,
                system_message, and flags. When provided, providers
                should use these instead of extracting from model_config.

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


def fallback_astream(
    provider: Provider,
    prompt: str,
    model_config: ModelConfig,
    timeout: int,
    request: ProviderRequest | None = None,
) -> StreamResult:
    """Create a StreamResult that falls back to query() for non-streaming providers."""

    async def _single_chunk() -> typing.AsyncIterator[str]:
        text, usage = await provider.query(prompt, model_config, timeout, request=request)
        result.usage = usage
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
    "CircuitBreaker",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_REGION",
    "DEFAULT_TIMEOUT",
    "MAX_RETRIES",
    "MODEL_TIMEOUT",
    "SOFT_TIMEOUT",
    "Provider",
    "ProviderRequest",
    "StreamResult",
    "StreamingProvider",
    "UsageTrackingStream",
    "fallback_astream",
]
