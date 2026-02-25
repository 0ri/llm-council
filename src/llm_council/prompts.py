"""Prompt templates for LLM Council stages."""

from __future__ import annotations

# Stage 2 Ranking Templates
RANKING_SYSTEM_MESSAGE_TEMPLATE = """You are a response evaluator. You will be shown multiple AI responses \
enclosed in <response-*> XML tags.

{manipulation_resistance_msg}

Your output must end with a valid JSON ranking object."""

RANKING_PROMPT_TEMPLATE = """Evaluate these responses to the following question:

Question: {question}

Here are the responses from different models (anonymized, enclosed in <response-{nonce}> XML tags):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, provide your final ranking as a JSON object.

IMPORTANT: Your response MUST end with a JSON object in exactly this format:
```json
{{"ranking": ["Response X", "Response Y", "Response Z"]}}
```

Where the array lists responses from BEST to WORST. Include all responses in the ranking.

Example evaluation format:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

```json
{{"ranking": ["Response C", "Response A", "Response B"]}}
```

Now provide your evaluation and ranking:"""


# Stage 3 Chairman Synthesis Templates
SYNTHESIS_SYSTEM_MESSAGE_TEMPLATE = """You are the Chairman of an LLM Council, responsible for \
synthesizing multiple AI responses into a single authoritative answer.

The responses you will evaluate are enclosed in <response-*> XML tags. {manipulation_resistance_msg}"""

SYNTHESIS_PROMPT_TEMPLATE = """Multiple AI models have provided responses to a user's question, \
and then peer-ranked each other's responses anonymously.

Original Question: {question}

STAGE 1 - Individual Responses (anonymized):

{stage1_text}

STAGE 2 - Aggregate Peer Rankings (best to worst):

{ranking_summary}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, \
accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""
