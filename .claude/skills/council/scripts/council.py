#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["llm-council-skill"]
# ///
"""
LLM Council CLI - Multi-model deliberation with anonymized peer review.

This is a thin wrapper that delegates to the installed llm_council package.
Run `uv sync` or `pip install -e .` from the project root to install.
"""

import sys


def main():
    try:
        from llm_council.cli import main as council_main
    except ImportError:
        print(
            "Error: llm_council package not installed.\nRun 'uv sync' or 'pip install -e .' from the project root.",
            file=sys.stderr,
        )
        sys.exit(1)
    council_main()


if __name__ == "__main__":
    main()
