"""CLI entry point for LLM Council."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

from .council import run_council, validate_config

logger = logging.getLogger("llm-council")


def setup_logging(verbose: bool = False):
    """Configure logging for the council script."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.setLevel(level)
    logger.addHandler(handler)


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from JSON file."""
    # Default config path
    if config_path is None:
        # Look for config in common locations
        possible_paths = [
            os.path.join(os.getcwd(), ".claude", "council-config.json"),
            os.path.expanduser("~/.claude/council-config.json"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

    if config_path is None or not os.path.exists(config_path):
        # Return default config with enhanced capabilities
        return {
            "council_models": [
                {
                    "name": "Claude Opus 4.6",
                    "provider": "bedrock",
                    "model_id": "us.anthropic.claude-opus-4-6-v1:0",
                    "budget_tokens": 10000,
                },
                {
                    "name": "GPT-5.3-Codex",
                    "provider": "poe",
                    "bot_name": "GPT-5.3-Codex",
                    "web_search": True,
                    "reasoning_effort": "high",
                },
                {
                    "name": "Gemini-3.1-Pro",
                    "provider": "poe",
                    "bot_name": "Gemini-3.1-Pro",
                    "web_search": True,
                    "reasoning_effort": "high",
                },
                {"name": "Grok-4", "provider": "poe", "bot_name": "Grok-4"},
            ],
            "chairman": {
                "name": "Claude Opus 4.6",
                "provider": "bedrock",
                "model_id": "us.anthropic.claude-opus-4-6-v1:0",
                "budget_tokens": 10000,
            },
        }

    try:
        with open(config_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file {config_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config file {config_path}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    # Parse arguments
    args = sys.argv[1:]

    if not args:
        print('Usage: council.py [--config CONFIG_PATH] [-v|--verbose] "question"', file=sys.stderr)
        sys.exit(1)

    # Handle flags
    config_path = None
    question = None
    verbose = False

    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif args[i] in ["-v", "--verbose"]:
            verbose = True
            i += 1
        else:
            question = args[i]
            i += 1

    if not question:
        print("Error: No question provided", file=sys.stderr)
        sys.exit(1)

    # Setup logging
    setup_logging(verbose=verbose)

    # Load config
    config = load_config(config_path)

    # Validate config early for better error reporting
    errors = validate_config(config)
    if errors:
        logger.error("Configuration errors:")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)

    # Run council
    result = asyncio.run(run_council(question, config))

    # Output result
    print(result)
