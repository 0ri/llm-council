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
    from . import StreamResult

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


class PoeProvider:
    """Poe.com provider for GPT, Gemini, Grok models."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def query(self, prompt: str, model_config: dict, timeout: int) -> tuple[str, None]:
        """Query a Poe.com bot with retry logic and timeout.

        Returns:
            Tuple of (response text, None) - Poe doesn't provide token counts
        """
        from . import MAX_RETRIES

        # Extract model-specific parameters
        bot_name = model_config["bot_name"]
        skip_flags = model_config.get("_skip_flags", False)
        web_search = False if skip_flags else model_config.get("web_search", False)
        reasoning_effort = None if skip_flags else model_config.get("reasoning_effort")

        # Parse the messages and system message from config
        messages = model_config.get("_messages", [{"role": "user", "content": prompt}])
        system_message = model_config.get("_system_message")

        import fastapi_poe as fp
        from fastapi_poe import ProtocolMessage

        @retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception(is_retryable_poe_error),
            reraise=True,
        )
        async def _query_poe_inner() -> str:
            # Convert messages to ProtocolMessage format
            # Poe uses 'bot' instead of 'assistant' for the role
            protocol_messages = []

            # Add system message as first user message if provided
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
                        # GPT uses --web_search, Gemini uses --web_search true
                        if "Gemini" in bot_name:
                            flags.append("--web_search true")
                        else:
                            flags.append("--web_search")

                    if reasoning_effort:
                        # GPT uses --reasoning_effort, Gemini uses --thinking_level
                        if "Gemini" in bot_name:
                            flags.append(f"--thinking_level {reasoning_effort}")
                        else:
                            flags.append(f"--reasoning_effort {reasoning_effort}")

                    # Flags go at the END of the message for Poe bots
                    if flags:
                        content = content + "\n\n" + " ".join(flags)

                protocol_messages.append(ProtocolMessage(role=role, content=content))

            # Accumulate response chunks
            accumulated_text = ""
            async for partial in fp.get_bot_response(
                messages=protocol_messages,
                bot_name=bot_name,
                api_key=self.api_key,
            ):
                accumulated_text += partial.text

            return accumulated_text

        result = await _query_poe_inner()
        # Poe doesn't provide token counts, return None for usage
        return result, None

    def astream(self, prompt: str, model_config: dict, timeout: int) -> StreamResult:
        """Stream a Poe.com bot response, yielding each partial text chunk.

        Returns:
            StreamResult wrapping an async generator of text chunks.
            Usage is always None for Poe.
        """
        from . import StreamResult

        bot_name = model_config["bot_name"]
        skip_flags = model_config.get("_skip_flags", False)
        web_search = False if skip_flags else model_config.get("web_search", False)
        reasoning_effort = None if skip_flags else model_config.get("reasoning_effort")

        messages = model_config.get("_messages", [{"role": "user", "content": prompt}])
        system_message = model_config.get("_system_message")

        import fastapi_poe as fp
        from fastapi_poe import ProtocolMessage

        async def _generate():
            protocol_messages = []

            if system_message:
                protocol_messages.append(ProtocolMessage(role="system", content=system_message))

            for i, msg in enumerate(messages):
                role = msg["role"]
                if role == "assistant":
                    role = "bot"

                content = msg["content"]

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

            async for partial in fp.get_bot_response(
                messages=protocol_messages,
                bot_name=bot_name,
                api_key=self.api_key,
            ):
                yield partial.text

        result = StreamResult(_generate())
        # Poe doesn't provide token counts
        result.usage = None
        return result
