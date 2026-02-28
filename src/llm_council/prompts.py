"""Prompt templates for Stage 2 ranking and Stage 3 chairman synthesis.

Contains system-message and user-prompt templates for peer ranking
(``RANKING_*_TEMPLATE``) and chairman synthesis (``SYNTHESIS_*_TEMPLATE``),
with placeholders for nonce-fenced responses and injection-hardening.
"""

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

Your task as Chairman is to write a comprehensive synthesis that integrates the strongest \
contributions from all council members. You must:

1. Preserve specific technical details, code references, file names, and concrete examples \
from the individual responses — do not abstract them away into vague summaries.
2. Where council members agree, state the consensus clearly and note the strength of agreement.
3. Where council members disagree or offer unique insights not found in other responses, \
highlight these as distinct perspectives worth the reader's attention.
4. Organize the synthesis by topic/theme rather than by response, so the reader gets a \
coherent narrative rather than a list of "Response A said X, Response B said Y."
5. The synthesis should be at least as detailed as the highest-ranked individual response. \
If the top-ranked response is 2000 words, your synthesis should be comparable in depth.

Write the synthesis now:"""
