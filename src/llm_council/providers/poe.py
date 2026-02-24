"""Poe.com provider for GPT, Gemini, Grok models."""
from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("llm-council")


class PoeProvider:
    """Poe.com provider for GPT, Gemini, Grok models."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def query(self, prompt: str, model_config: dict, timeout: int) -> str:
        """Query a Poe.com bot with retry logic and timeout."""
        from . import MAX_RETRIES

        # Extract model-specific parameters
        bot_name = model_config["bot_name"]
        web_search = model_config.get("web_search", False)
        reasoning_effort = model_config.get("reasoning_effort")

        # Parse the messages and system message from config
        messages = model_config.get("_messages", [{"role": "user", "content": prompt}])
        system_message = model_config.get("_system_message")

        import fastapi_poe as fp
        from fastapi_poe import ProtocolMessage

        @retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
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

        return await _query_poe_inner()
