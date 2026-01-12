# LLM Council

A Claude Code skill for multi-model LLM deliberation with anonymized peer review.

## What It Does

Instead of asking a question to a single LLM, the council queries multiple models, has them anonymously rank each other's responses, and synthesizes a final answer. The anonymization prevents models from playing favorites.

**3-Stage Process:**

1. **Stage 1: First opinions** - All models answer independently
2. **Stage 2: Peer review** - Models rank responses using anonymous labels (Response A/B/C)
3. **Stage 3: Synthesis** - Chairman compiles the final answer based on rankings

## Usage

### As a Claude Code Skill

```
/council "What's the best approach for building a REST API?"
```

### Configure Models

```
/council --config
```

## Setup

### 1. Install the Skill

Copy the `.claude/skills/council/` directory to your project or home directory.

### 2. Configure API Keys

**For Bedrock (Claude models):**
- AWS credentials must be configured (`~/.aws/credentials` or environment variables)

**For Poe.com (GPT, Gemini, Grok):**
```bash
export POE_API_KEY=your-poe-api-key-here
```

Get your Poe API key at [poe.com/api_key](https://poe.com/api_key).

### 3. Model Configuration

Edit `.claude/council-config.json`:

```json
{
  "council_models": [
    {"name": "Claude Opus 4.5", "provider": "bedrock", "model_id": "us.anthropic.claude-opus-4-5-20251101-v1:0"},
    {"name": "GPT-5", "provider": "poe", "bot_name": "GPT-5"},
    {"name": "Gemini-2.5-Pro", "provider": "poe", "bot_name": "Gemini-2.5-Pro"},
    {"name": "Grok-4", "provider": "poe", "bot_name": "Grok-4"}
  ],
  "chairman": {
    "name": "Claude Opus 4.5",
    "provider": "bedrock",
    "model_id": "us.anthropic.claude-opus-4-5-20251101-v1:0"
  }
}
```

## Available Models

### Bedrock (Anthropic)
- Claude Opus 4.5: `us.anthropic.claude-opus-4-5-20251101-v1:0`
- Claude Sonnet 4: `us.anthropic.claude-sonnet-4-20250514-v1:0`

### Poe.com
- OpenAI: `GPT-5`, `GPT-4o`, `GPT-4o-Mini`
- Google: `Gemini-2.5-Pro`, `Gemini-2.0-Flash`
- xAI: `Grok-4`

## Direct Script Usage

The council script can be run directly for testing:

```bash
python .claude/skills/council/scripts/council.py "What is the capital of France?"
```

## Output Format

```markdown
## LLM Council Response

### Model Rankings (by peer review)

| Rank | Model | Avg Position |
|------|-------|--------------|
| 1 | Claude Opus 4.5 | 1.5 |
| 2 | GPT-5 | 2.0 |
| 3 | Gemini-2.5-Pro | 2.8 |
| 4 | Grok-4 | 3.7 |

*Rankings determined by anonymous peer evaluation*

---

### Synthesized Answer

**Chairman:** Claude Opus 4.5

[Final synthesized response...]
```

## Background

This project was inspired by the desire to evaluate multiple LLMs side by side and see their cross-opinions on each other's outputs. The key innovation is the anonymized peer review - models evaluate "Response A", "Response B", etc. without knowing which model produced each response.
