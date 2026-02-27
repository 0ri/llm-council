"""Programmatic usage of the LLM Council deliberation pipeline.

Demonstrates how to call ``run_council()`` from Python with a custom
``CouncilContext`` and an inline configuration dictionary.
"""

import asyncio  # async event loop to drive the awaitable pipeline
import os  # read API keys from environment variables

from llm_council.context import CouncilContext  # per-run dependency container
from llm_council.council import run_council  # main 3-stage pipeline entry point


async def main() -> None:
    """Run a single council deliberation and print the result."""

    # Build a council configuration dictionary.
    # Three OpenRouter models act as council members; one doubles as chairman.
    config: dict = {
        "council_models": [
            {
                "name": "claude-sonnet",
                "provider": "openrouter",
                "model_id": "anthropic/claude-sonnet-4",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            {
                "name": "gpt-4o",
                "provider": "openrouter",
                "model_id": "openai/gpt-4o",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            {
                "name": "gemini-flash",
                "provider": "openrouter",
                "model_id": "google/gemini-2.0-flash-001",
                "temperature": 0.7,
                "max_tokens": 1024,
            },
        ],
        # The chairman synthesises the final answer in Stage 3
        "chairman": {
            "name": "claude-sonnet",
            "provider": "openrouter",
            "model_id": "anthropic/claude-sonnet-4",
            "temperature": 0.5,
            "max_tokens": 2048,
        },
    }

    # Create a context with the OpenRouter API key from the environment.
    # CouncilContext isolates provider instances, caching, and cost tracking.
    ctx = CouncilContext(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    # A factory callable lets run_council use our pre-configured context
    # instead of building its own from environment variables.
    context_factory = lambda: ctx  # noqa: E731

    # The question to present to the council
    question = "What are the key trade-offs between SQL and NoSQL databases?"

    # Execute the full 3-stage pipeline and capture the formatted output
    result = await run_council(question, config, context_factory=context_factory)

    # Print the council's deliberation result
    print(result)


# Standard async entry point
if __name__ == "__main__":
    asyncio.run(main())
