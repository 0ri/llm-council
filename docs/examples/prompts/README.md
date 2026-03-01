# Council Prompt Templates

The LLM Council uses 4 prompt templates across its 3 stages:

| Stage | Template | File | Purpose |
|-------|----------|------|---------|
| Stage 2 | Ranking System | [ranking-system.md](ranking-system.md) | System message for peer evaluators |
| Stage 2 | Ranking User | [ranking-user.md](ranking-user.md) | Presents responses, asks for ranking |
| Stage 3 | Synthesis System | [synthesis-system.md](synthesis-system.md) | System message for Chairman |
| Stage 3 | Synthesis User | [synthesis-user.md](synthesis-user.md) | Presents responses + rankings, asks for synthesis |

Stage 1 uses no special prompt template — each model simply receives the user's question directly.

## Customizing Prompts

Add a `prompts` section to your config JSON:

```json
{
  "council_models": [...],
  "chairman": {...},
  "prompts": {
    "ranking_system": "You are an expert {{domain}} evaluator...",
    "ranking_user": "Rate these responses on accuracy and depth...\n\n{question}\n\n{responses_text}",
    "synthesis_system": "You are a senior technical lead synthesizing team input...",
    "synthesis_user": "Synthesize these into an actionable decision document...\n\n{stage1_text}\n\n{ranking_summary}"
  }
}
```

Or load from files:

```json
{
  "prompts": {
    "ranking_system_file": "./my-prompts/ranking-system.txt",
    "ranking_user_file": "./my-prompts/ranking-user.txt",
    "synthesis_system_file": "./my-prompts/synthesis-system.txt",
    "synthesis_user_file": "./my-prompts/synthesis-user.txt"
  }
}
```

### Required Template Variables

Your custom prompts **must** include these placeholders (the council fills them at runtime):

**Ranking prompts:**
- `{question}` — Original question
- `{responses_text}` — Anonymized responses in XML tags
- `{manipulation_resistance_msg}` — Security instructions (system message only)

**Synthesis prompts:**
- `{question}` — Original question
- `{stage1_text}` — All Stage 1 responses
- `{ranking_summary}` — Aggregate rankings from Stage 2
- `{manipulation_resistance_msg}` — Security instructions (system message only)

### Tips

- **Domain expertise**: "You are a [role] evaluating [domain] responses..."
- **Evaluation criteria**: "Prioritize: correctness (40%), depth (30%), actionability (30%)"
- **Output format**: "End with a structured recommendation, not just analysis"
- **Brevity**: "Keep synthesis under N words" (default has no length limit)
- **Perspective**: "Write for an audience of senior engineers" or "Write for a non-technical executive"
