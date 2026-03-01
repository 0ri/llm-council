# Stage 2: Ranking System Message

This is the system message sent to each model when it evaluates and ranks the anonymized responses from Stage 1.

## Template Variables
- `{manipulation_resistance_msg}` — Auto-injected security instructions to prevent prompt injection from responses

## Default Template

```
You are a response evaluator. You will be shown multiple AI responses enclosed in <response-*> XML tags.

{manipulation_resistance_msg}

Your output must end with a valid JSON ranking object.
```

## Customization Tips

- Add domain expertise: "You are an expert software architect evaluating technical proposals..."
- Add evaluation criteria: "Prioritize correctness over style, practical examples over theory..."
- Add scoring dimensions: "Evaluate on: accuracy (40%), depth (30%), actionability (30%)"
