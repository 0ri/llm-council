---
name: council
description: Multi-model LLM council with anonymized peer review
user-invocable: true
homepage: https://github.com/0ri/llm-council
metadata: {"openclaw": {"emoji": "🏛️", "requires": {"bins": ["uv", "python3"], "env": ["POE_API_KEY"]}, "primaryEnv": "POE_API_KEY", "install": [{"id": "uv", "kind": "brew", "formula": "uv", "bins": ["uv"], "label": "Install uv (brew)"}]}}
---

# LLM Council

Query a council of LLMs that deliberate through 3 stages:
1. **Stage 1**: All models answer the question independently
2. **Stage 2**: Models anonymously rank each other's responses (Response A/B/C)
3. **Stage 3**: Chairman synthesizes the best answer based on rankings

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

   **Bedrock** (any model in your region — requires AWS credentials):
   - Claude Opus 4.6 (`us.anthropic.claude-opus-4-6-v1:0`) - supports budget_tokens
   - Claude Sonnet 4 (`us.anthropic.claude-sonnet-4-20250514-v1:0`)
   - Plus any other Bedrock model (Meta, Mistral, Cohere, etc.)

   **Poe.com** (any bot on Poe — requires POE_API_KEY):
   - GPT-5.2-Pro, GPT-5.2 - supports web_search, reasoning_effort (medium/high/Xhigh)
   - Gemini-3.1-Pro, Gemini-3-Pro - supports web_search, thinking_level (minimal/low/high)
   - Grok-4
   - Plus hundreds of other models and community bots

3. **Ask which model should be chairman** (single select from the chosen council models)

4. **Ask about enhanced parameters** for selected models:
   - For Claude models: budget_tokens (e.g., 10000 for extended thinking)
   - For GPT models: web_search (true/false), reasoning_effort (medium/high/Xhigh)
   - For Gemini models: web_search (true/false), thinking_level (minimal/low/high)

5. **Save the configuration** to `{baseDir}/config/council-config.json` with this format:

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
      "name": "GPT-5.2-Pro",
      "provider": "poe",
      "bot_name": "GPT-5.2-Pro",
      "web_search": true,
      "reasoning_effort": "high"
    },
    {
      "name": "Gemini-3.1-Pro",
      "provider": "poe",
      "bot_name": "Gemini-3.1-Pro",
      "web_search": true,
      "thinking_level": "high"
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

6. **Confirm** the configuration was saved successfully

## Output Format

The council returns a markdown summary with:

```markdown
## LLM Council Response

### Model Rankings (by peer review)

| Rank | Model | Avg Position |
|------|-------|--------------|
| 1 | Claude Opus 4.6 | 1.5 |
| 2 | GPT-5.2-Pro | 2.0 |
| 3 | Gemini-3.1-Pro | 2.8 |
| 4 | Grok-4 | 3.7 |

*Rankings determined by anonymous peer evaluation*

---

### Synthesized Answer

**Chairman:** Claude Opus 4.6

[Chairman's synthesis based on all responses and rankings...]
```

## Enhanced Model Parameters

| Provider | Parameter | Description |
|----------|-----------|-------------|
| Bedrock | `budget_tokens` | Extended thinking token budget (e.g., 10000) |
| Poe | `web_search` | Enable web search (true/false) |
| Poe | `reasoning_effort` | GPT models: "medium"/"high"/"Xhigh" |
| Poe | `thinking_level` | Gemini models: "minimal"/"low"/"high" |

## Requirements

- **AWS credentials** must be configured for Bedrock access (Claude models)
- **POE_API_KEY** environment variable required for Poe.com models (GPT, Gemini, Grok)
  - This is injected by OpenClaw via skill configuration

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