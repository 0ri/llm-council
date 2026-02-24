"""3-stage deliberation logic for LLM Council."""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from typing import Any

from .parsing import parse_ranking_from_text
from .progress import ModelStatus, ProgressManager
from .providers import MODEL_TIMEOUT, get_circuit_breaker, get_provider, get_semaphore
from .security import build_manipulation_resistance_msg, format_anonymized_responses

logger = logging.getLogger("llm-council")


async def query_model(
    model_config: dict[str, Any],
    messages: list[dict[str, str]],
    poe_api_key: str | None = None,
    system_message: str | None = None,
) -> dict[str, Any] | None:
    """Query a model through its provider, with timeout.

    Args:
        model_config: Dict with 'provider', 'model_id' or 'bot_name', and optional:
            - budget_tokens: int (Bedrock extended thinking)
            - web_search: bool (Poe web search)
            - reasoning_effort: str (Poe reasoning level)
        messages: List of message dicts
        poe_api_key: Poe API key (required for poe provider)
        system_message: Optional system message for injection hardening

    Returns:
        Response dict with 'content' key, or None if failed
    """
    provider_name = model_config.get("provider", "poe")
    model_name = model_config.get("name", "unknown")

    # Check circuit breaker
    cb = get_circuit_breaker(provider_name)
    if cb.is_open:
        logger.warning(f"Circuit breaker open for {provider_name}, skipping {model_name}")
        return None

    # Add messages and system_message to config for provider access
    config_with_context = model_config.copy()
    config_with_context["_messages"] = messages
    config_with_context["_system_message"] = system_message

    try:
        provider = get_provider(provider_name, poe_api_key)
        sem = get_semaphore()
        async with sem:
            result = await asyncio.wait_for(
                provider.query("", config_with_context, MODEL_TIMEOUT),
                timeout=MODEL_TIMEOUT,
            )
        cb.record_success()
        return {"content": result}
    except asyncio.TimeoutError:
        cb.record_failure()
        logger.warning(f"Timeout querying {model_name} after {MODEL_TIMEOUT}s")
        return None
    except Exception as e:
        cb.record_failure()
        logger.error(f"Error querying {model_name}: {e}")
        return None


async def query_models_parallel(
    model_configs: list[dict[str, Any]],
    messages: list[dict[str, str]],
    poe_api_key: str | None = None,
    system_message: str | None = None,
    progress: ProgressManager | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Query multiple models in parallel via hybrid providers.

    Args:
        model_configs: List of model config dicts
        messages: Messages to send to each model
        poe_api_key: Poe API key
        system_message: Optional system message for injection hardening
        progress: Optional progress manager for live status updates

    Returns:
        Dict mapping model name to response (or None if failed)
    """

    async def safe_query(config: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        name = config.get("name", "unknown")
        start = time.time()
        if progress:
            await progress.update_model(name, ModelStatus.QUERYING)
        try:
            result = await query_model(config, messages, poe_api_key, system_message)
            elapsed = time.time() - start
            if progress:
                await progress.update_model(
                    name,
                    ModelStatus.DONE if result else ModelStatus.FAILED,
                    elapsed,
                )
            return (name, result)
        except Exception as e:
            elapsed = time.time() - start
            if progress:
                await progress.update_model(name, ModelStatus.FAILED, elapsed)
            logger.error(f"Error querying {name}: {e}")
            return (name, None)

    # Create tasks for all models
    tasks = [safe_query(config) for config in model_configs]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Convert to dict
    return dict(results)


async def stage1_collect_responses(
    user_query: str,
    council_models: list[dict[str, Any]],
    poe_api_key: str | None,
    progress: ProgressManager | None = None,
) -> list[dict[str, Any]]:
    """Stage 1: Collect individual responses from all council models."""
    if progress:
        model_names = [m.get("name", "unknown") for m in council_models]
        await progress.start_stage(1, "Collecting responses", model_names)

    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel
    responses = await query_models_parallel(council_models, messages, poe_api_key, progress=progress)

    # Format results
    stage1_results: list[dict[str, Any]] = []
    for model_name, response in responses.items():
        if response is not None:
            stage1_results.append({"model": model_name, "response": response.get("content", "")})

    if progress:
        await progress.complete_stage(f"{len(stage1_results)}/{len(council_models)} responses")

    return stage1_results


def build_ranking_prompt(question: str, responses: list[tuple[str, str]], labels: list[str]) -> str:
    """Construct the prompt for peer ranking with anonymized responses.

    Args:
        question: The original user question
        responses: List of (model_name, response_text) tuples
        labels: List of single-character labels (A, B, C...)

    Returns:
        The formatted ranking prompt
    """
    # Generate a random nonce for XML delimiters to prevent fence-breaking
    nonce = secrets.token_hex(8)

    # Build responses with randomized XML delimiters for injection hardening
    responses_text = format_anonymized_responses(responses, nonce=nonce)

    ranking_prompt = f"""Evaluate these responses to the following question:

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

    return ranking_prompt


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: list[dict[str, Any]],
    council_models: list[dict[str, Any]],
    poe_api_key: str | None,
    progress: ProgressManager | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Stage 2: Each model ranks the anonymized responses.

    Uses injection hardening: fenced blocks + system message + structured JSON output.
    """
    if progress:
        model_names = [m.get("name", "unknown") for m in council_models]
        await progress.start_stage(2, f"Peer ranking ({len(stage1_results)} responses)", model_names)

    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result["model"] for label, result in zip(labels, stage1_results, strict=True)
    }

    # Build responses tuples for prompt construction
    response_tuples = [(result["model"], result["response"]) for result in stage1_results]

    # Build the ranking prompt
    ranking_prompt = build_ranking_prompt(user_query, response_tuples, labels)

    # System message for injection hardening
    base_resistance_msg = build_manipulation_resistance_msg()
    system_message = (
        "You are a response evaluator. You will be shown multiple AI responses "
        "enclosed in <response-*> XML tags.\n\n"
        f"{base_resistance_msg}\n\n"
        "Your output must end with a valid JSON ranking object."
    )

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel with system message
    responses = await query_models_parallel(council_models, messages, poe_api_key, system_message, progress=progress)

    # Format results
    stage2_results: list[dict[str, Any]] = []
    for model_name, response in responses.items():
        if response is not None:
            full_text = response.get("content", "")
            parsed_ranking, is_valid = parse_ranking_from_text(full_text, num_responses=len(stage1_results))
            stage2_results.append({
                "model": model_name,
                "ranking": full_text,
                "parsed_ranking": parsed_ranking,
                "is_valid_ballot": is_valid,
            })

    if progress:
        valid = sum(1 for r in stage2_results if r.get("is_valid_ballot"))
        await progress.complete_stage(f"{len(stage2_results)} rankings, {valid} valid ballots")

    return stage2_results, label_to_model


def build_synthesis_prompt(
    question: str,
    responses: list[tuple[str, str]],
    rankings: dict[str, str],
    labels: list[str],
    aggregate_rankings: list[dict[str, Any]],
) -> str:
    """Construct the chairman synthesis prompt with anonymized context.

    Args:
        question: The original user question
        responses: List of (model_name, response_text) tuples
        rankings: Mapping of labels to model names
        labels: List of response labels
        aggregate_rankings: Sorted aggregate ranking data

    Returns:
        The formatted synthesis prompt
    """
    # Generate a random nonce for XML delimiters to prevent fence-breaking
    nonce = secrets.token_hex(8)

    # Build anonymized context for chairman with randomized XML delimiters
    stage1_text = format_anonymized_responses(responses, nonce=nonce)

    # Create reverse mapping: model name -> label
    model_to_label = {v: k for k, v in rankings.items()}

    # Summarize rankings anonymously (which responses were ranked highest)
    ranking_summary_lines = []
    for rank_info in aggregate_rankings:
        model = rank_info["model"]
        label = model_to_label.get(model, "Unknown")
        ranking_summary_lines.append(f"- {label}: Average position {rank_info['average_rank']}")
    ranking_summary = "\n".join(ranking_summary_lines)

    chairman_prompt = (
        "Multiple AI models have provided responses to a user's question, "
        "and then peer-ranked each other's responses anonymously.\n\n"
        f"Original Question: {question}\n\n"
        "STAGE 1 - Individual Responses (anonymized):\n\n"
        f"{stage1_text}\n\n"
        "STAGE 2 - Aggregate Peer Rankings (best to worst):\n\n"
        f"{ranking_summary}\n\n"
        "Your task as Chairman is to synthesize all of this information into "
        "a single, comprehensive, accurate answer to the user's original "
        "question. Consider:\n"
        "- The individual responses and their insights\n"
        "- The peer rankings and what they reveal about response quality\n"
        "- Any patterns of agreement or disagreement\n\n"
        "Provide a clear, well-reasoned final answer that represents "
        "the council's collective wisdom:"
    )

    return chairman_prompt


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: list[dict[str, Any]],
    stage2_results: list[dict[str, Any]],
    label_to_model: dict[str, str],
    aggregate_rankings: list[dict[str, Any]],
    chairman_config: dict[str, Any],
    poe_api_key: str | None,
    progress: ProgressManager | None = None,
) -> dict[str, Any]:
    """Stage 3: Chairman synthesizes final response.

    Uses anonymized labels to prevent bias toward specific models.
    """
    chairman_name = chairman_config.get("name", "Chairman")

    if progress:
        await progress.start_stage(3, f"Chairman ({chairman_name}) synthesizing", [chairman_name])
        await progress.update_model(chairman_name, ModelStatus.QUERYING)

    # Build responses tuples and labels for prompt construction
    response_tuples = [(result["model"], result["response"]) for result in stage1_results]
    labels = [f"Response {chr(65 + i)}" for i in range(len(stage1_results))]

    # Build the synthesis prompt
    chairman_prompt = build_synthesis_prompt(
        user_query,
        response_tuples,
        label_to_model,
        labels,
        aggregate_rankings,
    )

    # System message for injection hardening
    base_resistance_msg = build_manipulation_resistance_msg()
    system_message = (
        "You are the Chairman of an LLM Council, responsible for "
        "synthesizing multiple AI responses into a single authoritative "
        "answer.\n\n"
        "The responses you will evaluate are enclosed in <response-*> "
        f"XML tags. {base_resistance_msg}"
    )

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model with system message
    start = time.time()
    response = await query_model(chairman_config, messages, poe_api_key, system_message)
    elapsed = time.time() - start

    if progress:
        status = ModelStatus.DONE if response else ModelStatus.FAILED
        await progress.update_model(chairman_name, status, elapsed)
        await progress.complete_stage()

    if response is None:
        return {"model": chairman_name, "response": "Error: Unable to generate final synthesis."}

    return {"model": chairman_name, "response": response.get("content", "")}
