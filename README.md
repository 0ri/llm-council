# LLM Council

![CI](https://github.com/0ri/llm-council/actions/workflows/ci.yml/badge.svg)

Multi-model LLM deliberation with anonymized peer review. Available as a Claude Code skill and an OpenClaw skill.

## What It Does

Instead of asking a question to a single LLM, the council queries multiple models, has them anonymously rank each other's responses, and synthesizes a final answer. The anonymization prevents models from playing favorites.

**3-Stage Process:**

1. **Stage 1: First opinions** - All models answer independently
2. **Stage 2: Peer review** - Models rank responses using anonymous labels (Response A/B/C)
3. **Stage 3: Synthesis** - Chairman compiles the final answer based on rankings

## Architecture

```mermaid
flowchart TD
    Q["User Query"]

    subgraph Stage1["Stage 1: First Opinions"]
        direction LR
        B["Bedrock Provider"]
        P["Poe Provider"]
        M1["Claude Opus 4.6"]
        M2["GPT-5.3-Codex"]
        M3["Gemini-3.1-Pro"]
        M4["Grok-4"]
        B --> M1
        P --> M2
        P --> M3
        P --> M4
    end

    A["Anonymization<br/>(Response A/B/C/D)"]

    subgraph Stage2["Stage 2: Peer Review"]
        direction LR
        R1["Model 1 ranks"]
        R2["Model 2 ranks"]
        R3["Model 3 ranks"]
        R4["Model 4 ranks"]
    end

    AGG["Ranking Aggregation"]

    subgraph Stage3["Stage 3: Synthesis"]
        CH["Chairman<br/>(Claude Opus 4.6)"]
    end

    OUT["Final Output"]

    Q --> Stage1
    Stage1 --> A
    A --> Stage2
    Stage2 --> AGG
    AGG --> Stage3
    Stage3 --> OUT
```

## Usage

```
/council "What's the best approach for building a REST API?"
/council --config
```

Works in both Claude Code and OpenClaw.

## Setup

### Claude Code

Copy the `.claude/skills/council/` directory to your project.

### OpenClaw

Copy `skills/council/` into your OpenClaw workspace, or install via ClawHub:

```bash
clawhub install council
```

Then configure the API key in `~/.openclaw/openclaw.json`:

```json5
{
  skills: {
    entries: {
      council: { enabled: true, apiKey: "YOUR_POE_API_KEY" }
    }
  }
}
```

### API Keys

**Bedrock (Claude models):** AWS credentials must be configured (`~/.aws/credentials` or environment variables)

**Poe.com (GPT, Gemini, Grok):**
```bash
export POE_API_KEY=your-poe-api-key-here
```

Get your Poe API key at [poe.com/api_key](https://poe.com/api_key).

### Model Configuration

Edit the council config (`council-config.json`):

```json
{
  "council_models": [
    {"name": "Claude Opus 4.6", "provider": "bedrock", "model_id": "us.anthropic.claude-opus-4-6-v1:0", "budget_tokens": 10000},
    {"name": "GPT-5.3-Codex", "provider": "poe", "bot_name": "GPT-5.3-Codex", "web_search": true, "reasoning_effort": "high"},
    {"name": "Gemini-3.1-Pro", "provider": "poe", "bot_name": "Gemini-3.1-Pro", "web_search": true},
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

## Available Models

You can use **any model** available on Bedrock or Poe.com — both providers offer hundreds of options including open-source models, smaller/faster models, and specialized models. The config examples above show current state-of-the-art choices, but you can swap in whatever models you want.

### Bedrock
Any model available in your AWS Bedrock region. Examples:
- Anthropic: Claude Opus 4.6, Claude Sonnet 4
- Meta, Mistral, Cohere, and others available through Bedrock

### Poe.com
Any bot available on Poe's API. Examples:
- OpenAI: GPT-5.3-Codex, GPT-5.2, GPT-4o
- Google: Gemini-3.1-Pro, Gemini-3-Flash
- xAI: Grok-4
- Plus hundreds of other models and community bots

## Direct Usage

Install and run the council as a Python package:

```bash
# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .

# Run the council
llm-council "What is the capital of France?"

# Or use the skill script directly
python .claude/skills/council/scripts/council.py "What is the capital of France?"
```

## Output Format

```markdown
## LLM Council Response

### Model Rankings (by peer review)

| Rank | Model | Avg Position |
|------|-------|--------------|
| 1 | Claude Opus 4.6 | 1.5 |
| 2 | GPT-5.3-Codex | 2.0 |
| 3 | Gemini-3.1-Pro | 2.8 |
| 4 | Grok-4 | 3.7 |

*Rankings determined by anonymous peer evaluation*

---

### Synthesized Answer

**Chairman:** Claude Opus 4.6

[Final synthesized response...]
```

## Background

This project was inspired by the desire to evaluate multiple LLMs side by side and see their cross-opinions on each other's outputs. The key innovation is the anonymized peer review - models evaluate "Response A", "Response B", etc. without knowing which model produced each response.

## Known Limitations

- **No built-in caching** - repeated queries re-query all models, increasing latency and API costs
- **Ranking parser relies on models following JSON output format** - if a model returns malformed JSON rankings, parsing may fail
- **No streaming output** - waits for full responses from all models before displaying results
- **Limited provider support** - currently only Bedrock and Poe providers implemented, though the architecture supports adding more
- **No persistence layer** - council sessions and results are not saved for later analysis
- **Fixed deliberation stages** - the 3-stage process is hardcoded, not configurable for different workflows

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and add tests
4. Submit a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
