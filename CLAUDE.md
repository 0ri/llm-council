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
3. **Stage 3**: Chairman synthesizes final answer based on rankings

### Hybrid Provider Strategy

- **Bedrock**: Claude Opus 4.6 (council member + chairman)
- **Poe.com**: GPT-5.3-Codex, Gemini-3.1-Pro, Grok-4
- **OpenRouter**: Any model via OpenAI-compatible API (real token usage reporting)

### Provider Abstraction

Providers (Bedrock, Poe, OpenRouter) are implemented as classes behind a common interface. New providers can be added by implementing the Provider protocol.

### Key Files

```
src/llm_council/                      # Main package
├── __init__.py                       # Package initialization
├── cli.py                            # CLI entry point
├── council.py                        # Council orchestration
├── stages.py                         # 3-stage deliberation logic
├── providers/                        # Provider implementations
│   ├── __init__.py
│   ├── base.py                       # Provider protocol
│   ├── bedrock.py                    # AWS Bedrock provider
│   ├── poe.py                        # Poe.com provider
│   └── openrouter.py                 # OpenRouter provider
├── aggregation.py                    # Ranking aggregation algorithms
├── flattener.py                      # Project directory flattener
├── parsing.py                        # Response parsing utilities
├── security.py                       # Injection hardening
├── formatting.py                     # Output formatting
├── progress.py                       # Progress visualization
├── cost.py                           # Cost tracking
└── models.py                         # Data models

tests/                                 # Test suite
├── test_aggregation.py
├── test_config.py
├── test_council_integration.py
├── test_parsing.py
├── test_resilience.py
└── test_security.py

.claude/                              # Claude Code skill
├── skills/council/
│   ├── commands/
│   │   └── council.md                # Skill command definition
│   └── scripts/
│       └── council.py                # Skill wrapper script
└── council-config.json               # Model configuration

skills/                               # OpenClaw skill
└── council/
    ├── SKILL.md                      # OpenClaw skill manifest
    ├── scripts/
    │   └── council.py → (symlink)    # Shared script
    └── config/
        └── council-config.json → (symlink)  # Shared config
```

## Configuration

Edit `.claude/council-config.json` to change models:

```json
{
  "council_models": [
    {
      "name": "Claude Opus 4.6",
      "provider": "bedrock",
      "model_id": "us.anthropic.claude-opus-4-6-v1:0",
      "budget_tokens": 10000
    },
    {
      "name": "GPT-5.3-Codex",
      "provider": "poe",
      "bot_name": "GPT-5.3-Codex",
      "web_search": true,
      "reasoning_effort": "high"
    },
    {
      "name": "Gemini-3.1-Pro",
      "provider": "poe",
      "bot_name": "Gemini-3.1-Pro",
      "web_search": true,
      "reasoning_effort": "high"
    },
    {"name": "Grok-4", "provider": "poe", "bot_name": "Grok-4"}
  ],
  "chairman": {
    "name": "Claude Opus 4.6",
    "provider": "bedrock",
    "model_id": "us.anthropic.claude-opus-4-6-v1:0",
    "budget_tokens": 10000
  }
}
```

### Enhanced Model Parameters

| Provider | Parameter | Description |
|----------|-----------|-------------|
| Bedrock | `budget_tokens` | Extended thinking token budget (e.g., 10000) |
| Poe | `web_search` | Enable web search (true/false) |
| Poe | `reasoning_effort` | GPT: "medium"/"high"/"Xhigh", Gemini: "minimal"/"low"/"high" |
| OpenRouter | `model_id` | OpenRouter model ID (e.g., "openai/gpt-4o") |
| OpenRouter | `temperature` | Optional temperature (0.0-2.0) |
| OpenRouter | `max_tokens` | Optional max output tokens |

## Security

### Injection Hardening
- Model responses are wrapped in fenced blocks during peer review to prevent prompt injection
- System messages instruct ranking/synthesis models to ignore manipulation attempts in responses
- Anonymous labels (Response A/B/C) prevent model name bias
- Nonce-based XML wrapping prevents response boundary confusion
- Input sanitization removes potentially malicious patterns

## Requirements

- **AWS credentials** configured for Bedrock (Claude models)
- **POE_API_KEY** environment variable for Poe.com models
- **OPENROUTER_API_KEY** environment variable for OpenRouter models

## Available Models

Any model available on Bedrock, Poe.com, or OpenRouter can be used. Run `llm-council --list-models` to discover available models.

### Bedrock
Any model in your AWS Bedrock region (Anthropic, Meta, Mistral, Cohere, etc.). Examples:
- `us.anthropic.claude-opus-4-6-v1:0` - Claude Opus 4.6 (supports extended thinking)
- `us.anthropic.claude-sonnet-4-20250514-v1:0` - Claude Sonnet 4

### Poe.com
Any bot on Poe's API (hundreds of models including open-source). Examples:
- `GPT-5.3-Codex`, `GPT-5.2` - supports web_search, reasoning_effort
- `Gemini-3.1-Pro`, `Gemini-3-Flash` - supports web_search, thinking_level
- `Grok-4`

### OpenRouter
Any model on OpenRouter's API (stable, OpenAI-compatible, real token usage). Examples:
- `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`, `google/gemini-pro`
- `meta-llama/llama-3-70b`, `mistralai/mixtral-8x7b`

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

Set `POE_API_KEY` via OpenClaw's skill config injection:

```json5
{
  skills: {
    entries: {
      council: {
        enabled: true,
        apiKey: "YOUR_POE_API_KEY",
      },
    },
  },
}
```

AWS credentials for Bedrock should be configured via `aws configure` or environment variables as usual.

## Key Design Decisions

### Anonymized Peer Review
Models receive "Response A", "Response B", etc. instead of model names. This prevents bias where models might favor their own family or disfavor competitors.

### Hybrid Providers
Bedrock for Anthropic models (already configured, no extra API key). Poe.com for others (single API key covers GPT, Gemini, Grok). OpenRouter as an alternative stable API with real token usage reporting.

### Subagent Execution
The skill spawns a subagent to run the council, preserving the main conversation's context window.

### Graceful Degradation
If a model fails, the council continues with remaining models rather than failing entirely.
