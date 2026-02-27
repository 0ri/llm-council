"""Custom provider implementation for LLM Council.

Shows how to satisfy the ``Provider`` and ``StreamingProvider`` protocols
so that a new LLM backend can participate in council deliberations.
"""

from __future__ import annotations

import asyncio  # async entry point
import typing  # AsyncIterator for streaming

from llm_council.context import CouncilContext  # per-run dependency container
from llm_council.providers import StreamResult  # wrapper for streaming chunks

# ---------------------------------------------------------------------------
# 1. Implement the Provider protocol
# ---------------------------------------------------------------------------
# Any class with an ``async def query(...)`` matching the signature below
# satisfies ``Provider``.  No base class or decorator is required — the
# council pipeline uses structural (duck) typing.


class EchoProvider:
    """Minimal provider that echoes the prompt back as its response."""

    async def query(
        self, prompt: str, model_config: dict, timeout: int
    ) -> tuple[str, dict | None]:
        """Return the prompt text as the model response.

        Args:
            prompt: Full prompt text sent by the pipeline.
            model_config: Provider-specific config (ignored here).
            timeout: Max seconds to wait (unused for this demo).

        Returns:
            A tuple of (response_text, usage_metadata).  Usage metadata
            reports approximate token counts so cost tracking works.
        """
        # Simulate a short network delay
        await asyncio.sleep(0.05)

        # Build simple usage metadata so the cost tracker can record it
        usage = {"input_tokens": len(prompt.split()), "output_tokens": len(prompt.split())}
        return prompt, usage


# ---------------------------------------------------------------------------
# 2. Optionally implement StreamingProvider
# ---------------------------------------------------------------------------
# Adding an ``astream`` method upgrades the provider to support token-by-
# token output during Stage 3 synthesis.  ``astream`` must return a
# ``StreamResult`` wrapping an ``AsyncIterator[str]``.


class EchoStreamingProvider(EchoProvider):
    """Streaming variant that yields the response word-by-word."""

    def astream(
        self, prompt: str, model_config: dict, timeout: int
    ) -> StreamResult:
        """Stream the echo response one word at a time.

        Returns a ``StreamResult`` whose async iterator yields individual
        words.  The pipeline collects them into the final response and
        populates ``StreamResult.usage`` when iteration completes.
        """

        async def _word_iterator() -> typing.AsyncIterator[str]:
            words = prompt.split()
            for i, word in enumerate(words):
                # Yield each word with a leading space (except the first)
                yield (" " + word) if i else word
                await asyncio.sleep(0.01)  # simulate incremental generation

        return StreamResult(_word_iterator())


# ---------------------------------------------------------------------------
# 3. Register the custom provider in a CouncilContext
# ---------------------------------------------------------------------------
# CouncilContext stores providers in its ``providers`` dict keyed by name.
# Pre-populating this dict bypasses the built-in factory so the pipeline
# uses your provider for any model whose ``provider`` field matches the key.


def build_context() -> CouncilContext:
    """Create a CouncilContext with the custom echo provider registered."""
    ctx = CouncilContext(
        # Inject the custom provider under the name "echo"
        providers={"echo": EchoStreamingProvider()},
    )
    return ctx


# Quick smoke test when run directly
if __name__ == "__main__":

    async def _demo() -> None:
        provider = EchoStreamingProvider()
        text, usage = await provider.query("Hello, council!", {}, timeout=30)
        print(f"query  -> {text!r}  usage={usage}")

        stream = provider.astream("Hello, council!", {}, timeout=30)
        chunks = [chunk async for chunk in stream]
        print(f"stream -> {''.join(chunks)!r}")

    asyncio.run(_demo())
