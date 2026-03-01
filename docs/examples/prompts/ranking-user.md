# Stage 2: Ranking User Prompt

This is the user prompt that presents the anonymized Stage 1 responses and asks the evaluator to rank them.

## Template Variables
- `{question}` — The original question asked
- `{responses_text}` — All Stage 1 responses, anonymized and wrapped in XML nonce tags
- `{nonce}` — Random nonce used in response XML tags (for injection resistance)

## Default Template

```
Evaluate these responses to the following question:

Question: {question}

Here are the responses from different models (anonymized, enclosed in <response-{nonce}> XML tags):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, provide your final ranking as a JSON object.

IMPORTANT: Your response MUST end with a JSON object in exactly this format:
```json
{"ranking": ["Response X", "Response Y", "Response Z"]}
```

Where the array lists responses from BEST to WORST. Include all responses in the ranking.

Example evaluation format:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

```json
{"ranking": ["Response C", "Response A", "Response B"]}
```

Now provide your evaluation and ranking:
```

## Customization Tips

- Add rubric: "Score each response 1-10 on these dimensions before ranking..."
- Add context: "The user asking this question is a senior engineer working on distributed systems..."
- Require justification: "For each ranking decision, cite specific quotes from the responses..."
