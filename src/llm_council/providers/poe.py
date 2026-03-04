"""Poe provider for GPT, Gemini, and Grok models via Poe.com.

Implements ``PoeProvider`` with ``query()`` and ``astream()`` methods using the
``fastapi_poe`` SDK. Supports bot-specific flags such as ``web_search`` and
``reasoning_effort`` (mapped to Gemini's ``--thinking_level``). Poe does not
report token usage, so cost tracking falls back to estimation for this provider.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from ..models import ModelConfig
    from . import ProviderRequest, StreamResult

logger = logging.getLogger("llm-council")


def is_retryable_poe_error(exc: BaseException) -> bool:
    """Determine if a Poe error is retryable."""
    # Timeout is always retryable
    if isinstance(exc, asyncio.TimeoutError):
        return True

    # Check for HTTP status codes in exception message
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__

    # Non-retryable: auth errors, not found, bad request
    if any(x in exc_str for x in ["401", "403", "404", "400", "unauthorized", "forbidden", "not found"]):
        logger.warning(f"Non-retryable Poe error: {exc_str[:200]}")
        return False

    # Non-retryable: bot doesn't exist or similar
    if "bot" in exc_str and ("not exist" in exc_str or "not found" in exc_str or "invalid" in exc_str):
        logger.warning(f"Non-retryable Poe error - bot issue: {exc_str[:200]}")
        return False

    # Retryable: rate limit, server errors, connection issues
    if any(x in exc_str for x in ["429", "500", "502", "503", "504", "rate", "throttl"]):
        return True

    # Connection errors are retryable
    if "connection" in exc_str or "Connection" in exc_type:
        return True

    # Network errors are retryable
    if "timeout" in exc_str or "Timeout" in exc_type:
        return True

    # Default to retry for unknown errors
    return True


def _build_protocol_messages(
    messages: list[dict[str, str]],
    bot_name: str,
    system_message: str | None,
    web_search: bool,
    reasoning_effort: str | None,
) -> list:
    """Build ProtocolMessage list from messages, applying Poe-specific flags."""
    from fastapi_poe import ProtocolMessage

    protocol_messages = []

    if system_message:
        protocol_messages.append(ProtocolMessage(role="system", content=system_message))

    for i, msg in enumerate(messages):
        role = msg["role"]
        if role == "assistant":
            role = "bot"

        content = msg["content"]

        # Add flags to the first user message
        if i == 0 and role == "user":
            flags = []
            if web_search:
                if "Gemini" in bot_name:
                    flags.append("--web_search true")
                else:
                    flags.append("--web_search")

            if reasoning_effort:
                if "Gemini" in bot_name:
                    flags.append(f"--thinking_level {reasoning_effort}")
                else:
                    flags.append(f"--reasoning_effort {reasoning_effort}")

            if flags:
                content = content + "\n\n" + " ".join(flags)

        protocol_messages.append(ProtocolMessage(role=role, content=content))

    return protocol_messages


class PoeProvider:
    """Poe.com provider for GPT, Gemini, Grok models."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def query(
        self,
        model_config: ModelConfig,
        timeout: int,
        request: ProviderRequest | None = None,
    ) -> tuple[str, None]:
        """Query a Poe.com bot with retry logic and timeout.

        Returns:
            Tuple of (response text, None) - Poe doesn't provide token counts
        """
        from . import MAX_RETRIES

        bot_name = model_config.bot_name

        # Use typed request if provided
        if request is not None:
            suppress_flags = request.suppress_provider_flags
            messages = request.messages
            system_message = request.system_message
        else:
            suppress_flags = False
            messages = []
            system_message = None

        web_search = False if suppress_flags else getattr(model_config, "web_search", False)
        reasoning_effort = None if suppress_flags else getattr(model_config, "reasoning_effort", None)

        import fastapi_poe as fp

        @retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception(is_retryable_poe_error),
            reraise=True,
        )
        async def _query_poe_inner() -> str:
            protocol_messages = _build_protocol_messages(
                messages, bot_name, system_message, web_search, reasoning_effort
            )

            accumulated_text = ""
            async for partial in fp.get_bot_response(
                messages=protocol_messages,
                bot_name=bot_name,
                api_key=self.api_key,
            ):
                accumulated_text += partial.text

            return accumulated_text

        result = await _query_poe_inner()
        return result, None

    def astream(
        self,
        model_config: ModelConfig,
        timeout: int,
        request: ProviderRequest | None = None,
    ) -> StreamResult:
        """Stream a Poe.com bot response, yielding each partial text chunk.

        Returns:
            StreamResult wrapping an async generator of text chunks.
            Usage is always None for Poe.
        """
        from . import StreamResult

        bot_name = model_config.bot_name

        if request is not None:
            suppress_flags = request.suppress_provider_flags
            messages = request.messages
            system_message = request.system_message
        else:
            suppress_flags = False
            messages = []
            system_message = None

        web_search = False if suppress_flags else getattr(model_config, "web_search", False)
        reasoning_effort = None if suppress_flags else getattr(model_config, "reasoning_effort", None)

        import fastapi_poe as fp

        async def _generate():
            protocol_messages = _build_protocol_messages(
                messages, bot_name, system_message, web_search, reasoning_effort
            )

            async for partial in fp.get_bot_response(
                messages=protocol_messages,
                bot_name=bot_name,
                api_key=self.api_key,
            ):
                yield partial.text

        result = StreamResult(_generate())
        result.usage = None
        return result
