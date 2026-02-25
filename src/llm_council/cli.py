"""CLI entry point for LLM Council."""

from __future__ import annotations

import argparse
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
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
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


def _print_dry_run(config: dict, question: str) -> None:
    """Print dry-run summary: models, query counts, budget limits."""
    council_models = config.get("council_models", [])
    chairman = config.get("chairman", {})
    n = len(council_models)

    print("=== Dry Run ===", file=sys.stderr)
    print(f"Question: {question[:120]}{'...' if len(question) > 120 else ''}", file=sys.stderr)
    print(f"\nCouncil models ({n}):", file=sys.stderr)
    for m in council_models:
        print(f"  - {m.get('name', '?')} ({m.get('provider', '?')})", file=sys.stderr)
    print(f"Chairman: {chairman.get('name', '?')} ({chairman.get('provider', '?')})", file=sys.stderr)

    # Estimate query counts
    stage1_queries = n
    stage2_queries = n  # each model ranks (N-1) responses but it's still N queries
    stage3_queries = 1
    total = stage1_queries + stage2_queries + stage3_queries
    print(
        f"\nEstimated API calls: {total} "
        f"(Stage 1: {stage1_queries}, Stage 2: {stage2_queries}, Stage 3: {stage3_queries})",
        file=sys.stderr,
    )

    budget = config.get("budget")
    if budget:
        max_tok = budget.get("max_tokens", "unlimited")
        max_cost = budget.get("max_cost_usd", "unlimited")
        print(f"Budget limits: {max_tok} tokens, ${max_cost}", file=sys.stderr)
    else:
        print("Budget limits: none configured", file=sys.stderr)

    print("=== No API calls were made ===", file=sys.stderr)


async def _list_available_models() -> None:
    """List available models from all configured providers."""
    # Bedrock
    print("=== Bedrock ===", file=sys.stderr)
    try:
        import boto3

        def _list_bedrock():
            client = boto3.client("bedrock")
            return client.list_foundation_models()

        response = await asyncio.to_thread(_list_bedrock)
        models = response.get("modelSummaries", [])
        for m in sorted(models, key=lambda x: x.get("modelId", "")):
            model_id = m.get("modelId", "")
            name = m.get("modelName", "")
            provider = m.get("providerName", "")
            print(f"  {model_id}  ({provider}: {name})", file=sys.stderr)
        if not models:
            print("  (no models found — check AWS credentials/region)", file=sys.stderr)
    except Exception as e:
        print(f"  (unavailable: {e})", file=sys.stderr)

    # Poe
    print("\n=== Poe.com ===", file=sys.stderr)
    poe_key = os.environ.get("POE_API_KEY")
    if poe_key:
        print("  Common bots (Poe has no discovery API):", file=sys.stderr)
        for bot in [
            "GPT-5.3-Codex",
            "GPT-5.2",
            "GPT-4o",
            "Gemini-3.1-Pro",
            "Gemini-3-Flash",
            "Grok-4",
            "Grok-3",
            "Claude-3.5-Sonnet",
            "Claude-3-Opus",
            "Llama-3.3-70B",
            "Mixtral-8x7B",
        ]:
            print(f"  {bot}", file=sys.stderr)
    else:
        print("  (POE_API_KEY not set)", file=sys.stderr)

    # OpenRouter
    print("\n=== OpenRouter ===", file=sys.stderr)
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if or_key:
        try:
            from .providers.openrouter import OpenRouterProvider

            provider = OpenRouterProvider(or_key)
            models = await provider.list_models()
            for m in models[:50]:  # Cap at 50 to avoid flooding
                print(f"  {m['id']}  ({m['name']})", file=sys.stderr)
            if len(models) > 50:
                print(f"  ... and {len(models) - 50} more", file=sys.stderr)
            await provider.close()
        except Exception as e:
            print(f"  (error listing models: {e})", file=sys.stderr)
    else:
        print("  (OPENROUTER_API_KEY not set)", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-council",
        description="Multi-model LLM deliberation with anonymized peer review.",
    )
    parser.add_argument("question", nargs="?", default=None, help="The question to ask the council")
    parser.add_argument("--config", dest="config", metavar="PATH", help="Path to council config JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--manifest", action="store_true", help="Print run manifest JSON to stderr")
    parser.add_argument("--log-dir", dest="log_dir", metavar="DIR", help="Write JSONL run logs to this directory")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3, help="Maximum stage to run (default: 3)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Preview config and costs, no API calls")
    parser.add_argument("--list-models", dest="list_models", action="store_true", help="List available models and exit")
    parser.add_argument("--flatten", dest="flatten", metavar="PATH", help="Flatten a directory and prepend to question")
    parser.add_argument("--question-file", dest="question_file", metavar="FILE", help="Read question from file")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible bootstrap confidence intervals")
    parser.add_argument("--no-cache", dest="no_cache", action="store_true", help="Disable local response cache")
    return parser


def main():
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # --list-models works without a question
    if args.list_models:
        setup_logging(verbose=args.verbose)
        asyncio.run(_list_available_models())
        return

    # Read question from file if --question-file is given
    if args.question_file:
        try:
            with open(args.question_file) as f:
                question = f.read()
        except Exception as e:
            parser.error(f"Cannot read question file: {e}")
    elif args.question:
        question = args.question
    else:
        parser.error("No question provided (use positional arg or --question-file)")

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Load config
    config = load_config(args.config)

    # Validate config early for better error reporting
    errors = validate_config(config)
    if errors:
        logger.error("Configuration errors:")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)

    # --dry-run: print summary and exit without API calls
    if args.dry_run:
        _print_dry_run(config, question)
        return

    # --flatten: prepend flattened directory content to question
    if args.flatten:
        from .flattener import flatten_directory

        flattened = flatten_directory(args.flatten)
        print(
            f"Flattened {flattened.file_count} files (est. ~{flattened.estimated_tokens:,} tokens)",
            file=sys.stderr,
        )
        if flattened.estimated_tokens > 100_000:
            print(
                f"Warning: flattened content is ~{flattened.estimated_tokens:,} tokens, "
                "may exceed model context limits",
                file=sys.stderr,
            )
        question = f"<project>\n{flattened.markdown}\n</project>\n\n{question}"

    # Run council
    result = asyncio.run(
        run_council(
            question,
            config,
            print_manifest=args.manifest,
            log_dir=args.log_dir,
            max_stage=args.stage,
            seed=args.seed,
            use_cache=not args.no_cache,
        )
    )

    # Output result
    print(result)
