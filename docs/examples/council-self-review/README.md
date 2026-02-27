# Council Self-Review Example

The council was asked to analyze its own codebase using the `--flatten --codemap` flags to provide a structural overview as context.

## Command

```bash
llm-council --flatten ./src --codemap --log-dir ./docs/examples/council-self-review --manifest \
  "Analyze this codebase. What are its strengths, weaknesses, and what would you improve?"
```

## Config

4 OpenRouter models: Claude Opus 4.6, GPT-5.3-Codex, Gemini-3.1-Pro, Grok 4.
Chairman: Gemini-3.1-Pro.

## Files

- `output.md` — the final synthesized council output (rankings + chairman synthesis)
- `689b624a-47ac-484f-8761-bdc689f6868b.jsonl` — full JSONL run log with individual model responses, raw ranking ballots, token usage, and aggregation data

## Results

- 4/4 valid ballots
- Claude Opus 4.6 ranked #1 unanimously
- Total time: 147.5s
- Total tokens: ~125K
