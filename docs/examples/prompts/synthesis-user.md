# Stage 3: Chairman Synthesis User Prompt

This is the user prompt for the Chairman that includes all Stage 1 responses, the aggregate peer rankings, and instructions for synthesis.

## Template Variables
- `{question}` — The original question
- `{stage1_text}` — All Stage 1 responses (anonymized, XML-wrapped)
- `{ranking_summary}` — Aggregate ranking results from Stage 2

## Default Template

```
Multiple AI models have provided responses to a user's question, and then peer-ranked each other's responses anonymously.

Original Question: {question}

STAGE 1 - Individual Responses (anonymized):

{stage1_text}

STAGE 2 - Aggregate Peer Rankings (best to worst):

{ranking_summary}

Your task as Chairman is to write a comprehensive synthesis that integrates the strongest contributions from all council members. You must:

1. Preserve specific technical details, code references, file names, and concrete examples from the individual responses — do not abstract them away into vague summaries.
2. Where council members agree, state the consensus clearly and note the strength of agreement.
3. Where council members disagree or offer unique insights not found in other responses, highlight these as distinct perspectives worth the reader's attention.
4. Organize the synthesis by topic/theme rather than by response, so the reader gets a coherent narrative rather than a list of "Response A said X, Response B said Y."
5. The synthesis should be at least as detailed as the highest-ranked individual response. If the top-ranked response is 2000 words, your synthesis should be comparable in depth.

Write the synthesis now:
```

## Customization Tips

- Change synthesis style: "Write as a decision document with clear recommendations..."
- Add weighting: "Give 2x weight to the top-ranked response..."
- Add action items: "End with a numbered list of concrete next steps..."
- Reduce verbosity: "Synthesis must be under 500 words. Prioritize actionability over completeness."
