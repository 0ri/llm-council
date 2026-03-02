---
name: council
description: Multi-model LLM council with anonymized peer review
user-invocable: true
homepage: https://github.com/0ri/llm-council
metadata: {"openclaw": {"emoji": "🏛️", "requires": {"bins": ["uv", "python3"], "env": ["OPENROUTER_API_KEY"]}, "primaryEnv": "OPENROUTER_API_KEY", "install": [{"id": "uv", "kind": "brew", "formula": "uv", "bins": ["uv"], "label": "Install uv (brew)"}]}}
---

# LLM Council

Query a council of LLMs that deliberate through 3 stages:
1. **Stage 1**: All models answer the question independently
2. **Stage 2**: Models anonymously rank each other's responses (Response A/B/C)
3. **Stage 3**: Chairman synthesizes the best answer based on rankings (auto-selected from #1 ranked model if not configured)

The anonymization prevents models from playing favorites with their own family or against competitors.

## Execution Instructions

**User's input:** $ARGUMENTS

### If `$ARGUMENTS` is a question (not --config):

Execute this command to run the LLM council:

```bash
uv run {baseDir}/scripts/council.py --config {baseDir}/config/council-config.json "$ARGUMENTS"
```

Return the complete markdown output from the council to display to the user.

The output will include:
- Model rankings determined by anonymous peer review
- A synthesized answer from the chairman model
- Details about which models participated

### If `$ARGUMENTS` is `--config`:

1. **Read the current configuration** from `{baseDir}/config/council-config.json`

2. **Ask the user which models to include** in the council (multi-select):

   **OpenRouter** (hundreds of models via single API key — requires OPENROUTER_API_KEY):
   - `anthropic/claude-opus-4.6`, `anthropic/claude-sonnet-4` - supports reasoning_max_tokens
   - `openai/gpt-5.3-codex`, `openai/gpt-4o` - supports reasoning_effort
   - `google/gemini-3.1-pro-preview` - supports reasoning_effort
   - `x-ai/grok-4` - supports reasoning_effort
   - Plus hundreds more (use `llm-council --list-models` to discover)

   **Bedrock** (any model in your region — requires AWS credentials):
   - Claude Opus 4.6 (`us.anthropic.claude-opus-4-6-v1:0`) - supports budget_tokens
   - Claude Sonnet 4 (`us.anthropic.claude-sonnet-4-20250514-v1:0`)
   - Plus any other Bedrock model (Meta, Mistral, Cohere, etc.)

   **Poe.com** (any bot on Poe — requires POE_API_KEY):
   - GPT-5.3-Codex, GPT-5.2 - supports web_search, reasoning_effort (medium/high/Xhigh)
   - Gemini-3.1-Pro, Gemini-3-Pro - supports web_search, reasoning_effort (minimal/low/high; mapped to thinking_level internally)
   - Grok-4
   - Plus hundreds of other models and community bots

3. **Ask which model should be chairman** (single select from the chosen council models, or leave blank for auto-select — the #1 ranked model from Stage 2 will be used)

4. **Ask about enhanced parameters** for selected models:
   - For OpenRouter Claude models: reasoning_max_tokens, max_tokens
   - For OpenRouter GPT/Gemini/Grok models: reasoning_effort (minimal/low/medium/high/xhigh), max_tokens
   - For Bedrock Claude models: budget_tokens (e.g., 10000 for extended thinking)
   - For Poe GPT models: web_search (true/false), reasoning_effort (medium/high/Xhigh)
   - For Poe Gemini models: web_search (true/false), reasoning_effort (minimal/low/high; sent as thinking_level)

5. **Save the configuration** to `{baseDir}/config/council-config.json` with this format:

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

Bedrock and Poe models can also be used. See the project README for alternative config examples.

6. **Confirm** the configuration was saved successfully

## Output Format

The council returns a markdown summary with:

```markdown
## LLM Council Response

### Model Rankings (by peer review)

| Rank | Model | Avg Position | 95% CI | Borda Score |
|------|-------|--------------|--------|-------------|
| 1 | Claude Opus 4.6 | 1.33 | [1.0, 1.67] | 2.67 |
| 2 | GPT-5.3-Codex | 2.0 | [1.33, 2.67] | 2.0 |
| 3 | Gemini-3.1-Pro | 2.67 | [2.0, 3.33] | 1.33 |
| 4 | Grok-4 | 3.67 | [3.33, 4.0] | 0.33 |

*Rankings based on 3/3 valid ballots (anonymous peer evaluation)*

---

### Synthesized Answer

**Chairman:** Claude Opus 4.6

[Chairman's synthesis based on all responses and rankings...]
```

## Enhanced Model Parameters

| Provider | Parameter | Description |
|----------|-----------|-------------|
| OpenRouter | `model_id` | OpenRouter model ID (e.g., `anthropic/claude-opus-4.6`) |
| OpenRouter | `reasoning_effort` | `minimal` / `low` / `medium` / `high` / `xhigh` |
| OpenRouter | `reasoning_max_tokens` | Token budget for reasoning (Anthropic, Qwen models) |
| OpenRouter | `max_tokens` | Max output tokens |
| OpenRouter | `temperature` | Sampling temperature (0.0-2.0) |
| Bedrock | `budget_tokens` | Extended thinking token budget (1024-128000) |
| Poe | `web_search` | Enable web search (true/false) |
| Poe | `reasoning_effort` | GPT: "medium"/"high"/"Xhigh", Gemini: "minimal"/"low"/"high" |

## Requirements

- **OPENROUTER_API_KEY** environment variable for OpenRouter models (default config)
  - This is injected by OpenClaw via skill configuration
- **POE_API_KEY** environment variable for Poe.com models (if using Poe provider)
- **AWS credentials** for Bedrock models (if using Bedrock provider)

You only need keys for providers in your config.

## Error Handling

- If a model fails during deliberation, the council continues with remaining models
- If config file is missing or corrupted, use default configuration with fallback models
- If credentials are missing, show clear setup instructions for AWS CLI and POE_API_KEY

## Direct Testing

The skill script can be run directly for testing:

```bash
# Run with default config
python {baseDir}/scripts/council.py "What is 2+2?"

# Run with custom config path
python {baseDir}/scripts/council.py --config /path/to/config.json "question"
```