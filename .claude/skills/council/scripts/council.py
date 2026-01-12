#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi-poe>=0.0.79", "boto3>=1.34"]
# ///
"""
LLM Council CLI - Multi-model deliberation with anonymized peer review.

Usage:
    python council.py "What is the best approach for X?"
    python council.py --config /path/to/config.json "question"
"""

import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# PROVIDERS: Bedrock and Poe.com
# ============================================================================

def query_bedrock(model_id: str, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """
    Query Claude via AWS Bedrock.

    Args:
        model_id: Bedrock model ID (e.g., us.anthropic.claude-opus-4-5-20251101-v1:0)
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Response dict with 'content' key, or None if failed
    """
    try:
        import boto3

        client = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Convert to Bedrock message format
        bedrock_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8192,
                "messages": bedrock_messages
            })
        )

        result = json.loads(response["body"].read())
        return {"content": result["content"][0]["text"]}

    except Exception as e:
        print(f"Bedrock error for {model_id}: {e}", file=sys.stderr)
        return None


async def query_poe(bot_name: str, messages: List[Dict[str, str]], api_key: str) -> Optional[Dict[str, Any]]:
    """
    Query a Poe.com bot.

    Args:
        bot_name: Poe bot display name (e.g., "GPT-5")
        messages: List of message dicts with 'role' and 'content'
        api_key: Poe API key

    Returns:
        Response dict with 'content' key, or None if failed
    """
    try:
        import fastapi_poe as fp
        from fastapi_poe import ProtocolMessage

        # Convert messages to ProtocolMessage format
        # Poe uses 'bot' instead of 'assistant' for the role
        protocol_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                role = "bot"
            protocol_messages.append(
                ProtocolMessage(role=role, content=msg["content"])
            )

        # Accumulate response chunks
        accumulated_text = ""
        async for partial in fp.get_bot_response(
            messages=protocol_messages,
            bot_name=bot_name,
            api_key=api_key,
        ):
            accumulated_text += partial.text

        return {"content": accumulated_text}

    except Exception as e:
        print(f"Poe error for {bot_name}: {e}", file=sys.stderr)
        return None


async def query_model_hybrid(
    model_config: Dict[str, Any],
    messages: List[Dict[str, str]],
    poe_api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Route to appropriate provider based on config.

    Args:
        model_config: Dict with 'provider', 'model_id' or 'bot_name'
        messages: List of message dicts
        poe_api_key: Poe API key (required for poe provider)

    Returns:
        Response dict with 'content' key, or None if failed
    """
    provider = model_config.get("provider")

    if provider == "bedrock":
        # Bedrock is sync, run in thread pool
        return await asyncio.to_thread(
            query_bedrock, model_config["model_id"], messages
        )
    elif provider == "poe":
        if not poe_api_key:
            print(f"POE_API_KEY not set, cannot query {model_config.get('name')}", file=sys.stderr)
            return None
        return await query_poe(model_config["bot_name"], messages, poe_api_key)
    else:
        print(f"Unknown provider: {provider}", file=sys.stderr)
        return None


async def query_models_parallel(
    model_configs: List[Dict[str, Any]],
    messages: List[Dict[str, str]],
    poe_api_key: Optional[str] = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel via hybrid providers.

    Args:
        model_configs: List of model config dicts
        messages: Messages to send to each model
        poe_api_key: Poe API key

    Returns:
        Dict mapping model name to response (or None if failed)
    """
    async def safe_query(config: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        name = config.get("name", "unknown")
        try:
            result = await query_model_hybrid(config, messages, poe_api_key)
            return (name, result)
        except Exception as e:
            print(f"Error querying {name}: {e}", file=sys.stderr)
            return (name, None)

    # Create tasks for all models
    tasks = [safe_query(config) for config in model_configs]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Convert to dict
    return {name: response for name, response in results}


# ============================================================================
# COUNCIL LOGIC: 3-Stage Deliberation
# ============================================================================

async def stage1_collect_responses(
    user_query: str,
    council_models: List[Dict[str, Any]],
    poe_api_key: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.
    """
    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel
    responses = await query_models_parallel(council_models, messages, poe_api_key)

    # Format results
    stage1_results = []
    for model_name, response in responses.items():
        if response is not None:
            stage1_results.append({
                "model": model_name,
                "response": response.get('content', '')
            })

    return stage1_results


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    council_models: List[Dict[str, Any]],
    poe_api_key: Optional[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(council_models, messages, poe_api_key)

    # Format results
    stage2_results = []
    for model_name, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model_name,
                "ranking": full_text,
                "parsed_ranking": parsed
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    chairman_config: Dict[str, Any],
    poe_api_key: Optional[str]
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.
    """
    chairman_name = chairman_config.get("name", "Chairman")

    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    response = await query_model_hybrid(chairman_config, messages, poe_api_key)

    if response is None:
        return {
            "model": chairman_name,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": chairman_name,
        "response": response.get('content', '')
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.
    """
    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.
    """
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_output(
    aggregate_rankings: List[Dict[str, Any]],
    stage3_result: Dict[str, Any]
) -> str:
    """
    Format the council results as markdown.
    """
    output = []
    output.append("## LLM Council Response\n")

    # Rankings table
    output.append("### Model Rankings (by peer review)\n")
    output.append("| Rank | Model | Avg Position |")
    output.append("|------|-------|--------------|")

    for i, ranking in enumerate(aggregate_rankings, start=1):
        output.append(f"| {i} | {ranking['model']} | {ranking['average_rank']} |")

    output.append("\n*Rankings determined by anonymous peer evaluation (Response A/B/C labels)*\n")
    output.append("---\n")

    # Chairman synthesis
    output.append("### Synthesized Answer\n")
    output.append(f"**Chairman:** {stage3_result['model']}\n")
    output.append(stage3_result['response'])

    return "\n".join(output)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def run_council(question: str, config: Dict[str, Any]) -> str:
    """
    Run the full 3-stage council process.
    """
    council_models = config.get("council_models", [])
    chairman_config = config.get("chairman", {})
    poe_api_key = os.environ.get("POE_API_KEY")

    # Validate config
    if not council_models:
        return "Error: No council models configured."

    if not chairman_config:
        return "Error: No chairman model configured."

    # Check for required API keys
    needs_poe = any(m.get("provider") == "poe" for m in council_models)
    needs_poe = needs_poe or chairman_config.get("provider") == "poe"

    if needs_poe and not poe_api_key:
        return "Error: POE_API_KEY environment variable not set. Get one at poe.com/api_key"

    print("Stage 1: Collecting responses from council...", file=sys.stderr)
    stage1_results = await stage1_collect_responses(question, council_models, poe_api_key)

    if not stage1_results:
        return "Error: All models failed to respond. Please check your API credentials."

    print(f"Stage 1 complete: {len(stage1_results)} responses collected", file=sys.stderr)

    print("Stage 2: Collecting peer rankings...", file=sys.stderr)
    stage2_results, label_to_model = await stage2_collect_rankings(
        question, stage1_results, council_models, poe_api_key
    )

    print(f"Stage 2 complete: {len(stage2_results)} rankings collected", file=sys.stderr)

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    print("Stage 3: Chairman synthesizing final answer...", file=sys.stderr)
    stage3_result = await stage3_synthesize_final(
        question, stage1_results, stage2_results, chairman_config, poe_api_key
    )

    print("Stage 3 complete!", file=sys.stderr)

    # Format output
    return format_output(aggregate_rankings, stage3_result)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    """
    # Default config path
    if config_path is None:
        # Look for config in common locations
        possible_paths = [
            os.path.join(os.getcwd(), ".claude", "council-config.json"),
            os.path.expanduser("~/.claude/council-config.json"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

    if config_path is None or not os.path.exists(config_path):
        # Return default config
        return {
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

    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point."""
    # Parse arguments
    args = sys.argv[1:]

    if not args:
        print("Usage: council.py [--config CONFIG_PATH] \"question\"", file=sys.stderr)
        sys.exit(1)

    # Handle --config flag
    config_path = None
    question = None

    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        else:
            question = args[i]
            i += 1

    if not question:
        print("Error: No question provided", file=sys.stderr)
        sys.exit(1)

    # Load config
    config = load_config(config_path)

    # Run council
    result = asyncio.run(run_council(question, config))

    # Output result
    print(result)


if __name__ == "__main__":
    main()
