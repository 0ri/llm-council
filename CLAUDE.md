# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM Council is a Claude Code skill that runs multi-model deliberation with anonymized peer review. It queries multiple LLMs, has them rank each other's responses (using anonymous labels to prevent bias), and synthesizes a final answer.

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

- **Bedrock**: Claude Opus 4.5 (council member + chairman)
- **Poe.com**: GPT-5, Gemini-2.5-Pro, Grok-4

### Key Files

```
.claude/
├── skills/council/
│   ├── SKILL.md              # Skill manifest and instructions
│   └── scripts/
│       └── council.py        # Self-contained CLI script
└── council-config.json       # Model configuration

backend/
├── council.py                # Reference implementation (not used by skill)
└── poe.py                    # Reference Poe client (not used by skill)
```

## Configuration

Edit `.claude/council-config.json` to change models:

```json
{
  "council_models": [
    {
      "name": "Claude Opus 4.5",
      "provider": "bedrock",
      "model_id": "us.anthropic.claude-opus-4-5-20251101-v1:0",
      "budget_tokens": 10000
    },
    {
      "name": "GPT-5.2-Pro",
      "provider": "poe",
      "bot_name": "GPT-5.2-Pro",
      "web_search": true,
      "reasoning_effort": "high"
    },
    {
      "name": "Gemini-3-Flash",
      "provider": "poe",
      "bot_name": "Gemini-3-Flash",
      "web_search": true,
      "reasoning_effort": "high"
    },
    {"name": "Grok-4", "provider": "poe", "bot_name": "Grok-4"}
  ],
  "chairman": {
    "name": "Claude Opus 4.5",
    "provider": "bedrock",
    "model_id": "us.anthropic.claude-opus-4-5-20251101-v1:0",
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

## Requirements

- **AWS credentials** configured for Bedrock (Claude models)
- **POE_API_KEY** environment variable for Poe.com models

## Available Models

### Bedrock (Anthropic)
- `us.anthropic.claude-opus-4-5-20251101-v1:0` - Claude Opus 4.5 (supports extended thinking)
- `us.anthropic.claude-sonnet-4-20250514-v1:0` - Claude Sonnet 4

### Poe.com
- `GPT-5.2-Pro` - supports web_search, reasoning_effort (medium/high/Xhigh)
- `GPT-5`, `GPT-4o`
- `Gemini-3-Flash`, `Gemini-3-Pro` - supports web_search, thinking_level (minimal/low/high)
- `Grok-4` - no enhanced features via API

## Direct Script Usage

The skill script can be run directly for testing:

```bash
# Run with default config
python .claude/skills/council/scripts/council.py "What is 2+2?"

# Run with custom config
python .claude/skills/council/scripts/council.py --config /path/to/config.json "question"
```

## Key Design Decisions

### Anonymized Peer Review
Models receive "Response A", "Response B", etc. instead of model names. This prevents bias where models might favor their own family or disfavor competitors.

### Hybrid Providers
Bedrock for Anthropic models (already configured, no extra API key). Poe.com for others (single API key covers GPT, Gemini, Grok).

### Subagent Execution
The skill spawns a subagent to run the council, preserving the main conversation's context window.

### Graceful Degradation
If a model fails, the council continues with remaining models rather than failing entirely.
