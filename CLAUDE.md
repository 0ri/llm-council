# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM Council runs multi-model deliberation with anonymized peer review. It queries multiple LLMs, has them rank each other's responses (using anonymous labels to prevent bias), and synthesizes a final answer. Available as both a Claude Code skill and an OpenClaw skill.

## Skill Usage

### Normal Mode
```
/council "What's the best approach for building a REST API?"
```

### Config Mode
```
/council --config
```

## Architecture

### 3-Stage Deliberation

1. **Stage 1**: All council models answer the question independently
2. **Stage 2**: Models rank responses using anonymous labels (Response A/B/C)
3. **Stage 3**: Chairman synthesizes final answer based on rankings (auto-selected from #1 ranked model if not configured, with fallback on failure)

### Provider Strategy

The default config uses **OpenRouter** for all models (single API key, real token usage reporting). Alternative providers:

- **AWS Bedrock**: Native AWS auth, extended thinking via `budget_tokens`
- **Poe.com**: Single API key covers GPT, Gemini, Grok, and community bots
- **OpenRouter**: OpenAI-compatible API, model discovery via `--list-models`

### Provider Abstraction

Providers (Bedrock, Poe, OpenRouter) are implemented as classes behind a common interface. New providers can be added by implementing the Provider protocol.

### Key Files

See [Package Structure](README.md#package-structure) in the README for the full module listing. Core files:

```
src/llm_council/
├── cli.py                  # CLI entry point (thin dispatcher → _cmd_* functions)
├── council.py              # Orchestrator: _RunState, stage helpers, run_council
├── run_options.py           # RunOptions dataclass for run_council parameters
├── models.py               # Pydantic config models and result types
├── defaults.py             # Shared default constants (cache TTL, timeouts, retries)
├── context.py              # Per-run DI container (lazy provider imports)
├── _token_estimation.py    # Shared tiktoken/heuristic token estimator
├── stages/                 # 3-stage pipeline package
│   ├── __init__.py         # Re-exports for backward compatibility
│   ├── execution.py        # query_model, stream_model, parallel dispatch
│   ├── stage1.py           # Collect individual responses (with caching)
│   ├── stage2.py           # Anonymized peer ranking with retry
│   └── stage3.py           # Chairman synthesis (query or streaming)
├── providers/              # Provider implementations (Bedrock, Poe, OpenRouter)
├── aggregation.py          # Borda count, bootstrap confidence intervals
├── budget.py               # Token and cost budget guards
├── cache.py                # SQLite response cache for Stage 1
├── persistence.py          # Buffered JSONL run logger
├── security.py             # Input sanitization, injection detection, nonce fencing
├── prompts.py              # Prompt templates for ranking and synthesis
└── formatting.py           # Markdown output formatting

.claude/
├── commands/council.md     # Claude Code skill command definition
├── skills/council/scripts/council.py  # Skill wrapper script
└── council-config.json     # Model configuration

skills/council/             # OpenClaw skill (symlinks to shared script/config)
```

## Configuration

Edit `.claude/council-config.json` to change models. The `chairman` field is optional — if omitted, the #1 ranked model from Stage 2 is automatically selected as chairman. The default config uses OpenRouter for all models:

```json
{
  "council_models": [
    {
      "name": "Claude Opus 4.6",
      "provider": "openrouter",
      "model_id": "anthropic/claude-opus-4.6",
      "reasoning_max_tokens": 16000,
      "max_tokens": 32000
    },
    {
      "name": "GPT-5.3-Codex",
      "provider": "openrouter",
      "model_id": "openai/gpt-5.3-codex",
      "reasoning_effort": "high",
      "max_tokens": 32000
    },
    {
      "name": "Gemini-3.1-Pro",
      "provider": "openrouter",
      "model_id": "google/gemini-3.1-pro-preview",
      "reasoning_effort": "high",
      "max_tokens": 32000
    },
    {
      "name": "Grok 4",
      "provider": "openrouter",
      "model_id": "x-ai/grok-4",
      "reasoning_effort": "high",
      "max_tokens": 32000
    }
  ],
  "chairman": {
    "name": "Claude Opus 4.6",
    "provider": "openrouter",
    "model_id": "anthropic/claude-opus-4.6",
    "reasoning_max_tokens": 16000,
    "max_tokens": 32000
  }
}
```

For the full configuration reference (all provider-specific fields, budget controls, cache settings, resilience tuning), see the [Configuration Reference](README.md#configuration-reference) in the README.

## Security

### Injection Hardening
- Model responses are wrapped in fenced blocks during peer review to prevent prompt injection
- System messages instruct ranking/synthesis models to ignore manipulation attempts in responses
- Anonymous labels (Response A/B/C) prevent model name bias
- Nonce-based XML wrapping prevents response boundary confusion
- Input sanitization removes potentially malicious patterns

## Requirements

- **OPENROUTER_API_KEY** for OpenRouter models (default config — single key covers all models)
- **POE_API_KEY** for Poe.com models (if using Poe provider)
- **AWS credentials** for Bedrock models (if using Bedrock provider)

You only need keys for the providers in your config. Run `llm-council --list-models` to discover available models. See the [README](README.md#available-models) for the full model list.

## Direct Usage

The council is available as a Python package:

```bash
# Install the package
uv sync  # or: pip install -e .

# Run via CLI entry point (full 3-stage deliberation)
llm-council "What is 2+2?"
llm-council --config /path/to/config.json "question"

# Run only Stage 1 (individual responses, no ranking/synthesis)
llm-council --stage 1 "question"

# Run Stages 1-2 (responses + rankings, no synthesis)
llm-council --stage 2 "question"

# Dry run (preview cost/model list, no API calls)
llm-council --dry-run "question"

# List available models from all providers
llm-council --list-models

# Flatten a codebase and ask for review
llm-council --flatten ./src "Review this code for bugs"

# Or run the skill script directly
python .claude/skills/council/scripts/council.py "What is 2+2?"
```

## OpenClaw Skill

The OpenClaw-compatible skill lives in `skills/council/`. It shares the same Python script and config via symlinks.

### Installation

Copy or symlink `skills/council/` into your OpenClaw workspace skills directory, or install via ClawHub:

```bash
clawhub install council
```

### Configuration

Set your API key via OpenClaw's skill config injection. The default config uses OpenRouter:

```json5
{
  skills: {
    entries: {
      council: {
        enabled: true,
        apiKey: "YOUR_OPENROUTER_API_KEY",
      },
    },
  },
}
```

For Poe models, set `POE_API_KEY` instead. For Bedrock, configure AWS credentials via `aws configure` or environment variables.

## Key Design Decisions

### Anonymized Peer Review
Models receive "Response A", "Response B", etc. instead of model names. This prevents bias where models might favor their own family or disfavor competitors.

### Multi-Provider Support
OpenRouter as default (single key, real token usage, broadest model access). Bedrock for direct AWS integration and extended thinking. Poe for access to community bots and web search augmentation.

### Subagent Execution
The skill spawns a subagent to run the council, preserving the main conversation's context window.

### Graceful Degradation
If a model fails, the council continues with remaining models rather than failing entirely. Circuit breakers skip consistently failing models. See [Troubleshooting](docs/TROUBLESHOOTING.md) for common errors and fixes.
