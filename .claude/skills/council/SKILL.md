---
name: council
description: Multi-model LLM council with anonymized peer review
invoke: /council
args: question or --config
---

# LLM Council Skill

Query a council of LLMs that deliberate through 3 stages:
1. **Stage 1**: All models answer the question independently
2. **Stage 2**: Models anonymously rank each other's responses (Response A/B/C)
3. **Stage 3**: Chairman synthesizes the best answer based on rankings

The anonymization prevents models from playing favorites.

## Invocation

### Normal Mode
```
/council "What's the best approach for building a REST API?"
```

### Config Mode
```
/council --config
```

## Execution Instructions

### When user runs `/council "question"`:

1. **Read config** from the project's `.claude/council-config.json`
   - If config doesn't exist, create it with defaults first

2. **Spawn orchestrator subagent** using the Task tool:
   ```
   Task(
     subagent_type="general-purpose",
     description="Run LLM council",
     prompt="Execute the LLM council script to answer this question: [QUESTION]

     Run: python /path/to/.claude/skills/council/scripts/council.py '[QUESTION]'

     The script will query multiple models and return a markdown summary.
     Return the full output to the user."
   )
   ```

3. **Display the results** inline in the conversation

### When user runs `/council --config`:

1. **Use AskUserQuestion** to let user configure:
   - Which models to include in the council (multi-select)
   - Which model should be chairman

2. **Available models**:

   **Bedrock (Anthropic)**:
   - Claude Opus 4.5 (`us.anthropic.claude-opus-4-5-20251101-v1:0`)
   - Claude Sonnet 4 (`us.anthropic.claude-sonnet-4-20250514-v1:0`)

   **Poe.com** (requires POE_API_KEY):
   - GPT-5
   - GPT-4o
   - Gemini-2.5-Pro
   - Gemini-2.0-Flash
   - Grok-4

3. **Save selections** to `.claude/council-config.json`

4. **Confirm** the configuration was saved

## Config File Format

Location: `.claude/council-config.json`

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

## Output Format

The council returns a markdown summary:

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

[Chairman's synthesis based on all responses and rankings...]
```

## Requirements

- **AWS credentials** configured for Bedrock access (Claude models)
- **POE_API_KEY** environment variable for Poe.com models (GPT, Gemini, Grok)

## Error Handling

- If a model fails, the council continues with remaining models
- If config file is missing, create default configuration
- If credentials are missing, show clear setup instructions
