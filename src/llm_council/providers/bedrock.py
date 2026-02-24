"""AWS Bedrock provider for Anthropic models."""
from __future__ import annotations

import asyncio
import json

from tenacity import retry, stop_after_attempt, wait_exponential


class BedrockProvider:
    """AWS Bedrock provider for Anthropic models."""

    def __init__(self, region: str | None = None):
        from . import DEFAULT_REGION

        self.region = region or DEFAULT_REGION
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    async def query(self, prompt: str, model_config: dict, timeout: int) -> str:
        """Query Claude via AWS Bedrock with timeout and retry logic."""
        from . import DEFAULT_MAX_TOKENS, MAX_RETRIES

        # Extract model-specific parameters
        model_id = model_config["model_id"]
        budget_tokens = model_config.get("budget_tokens")

        # Parse the prompt as messages (it's already formatted)
        messages = model_config.get("_messages", [{"role": "user", "content": prompt}])
        system_message = model_config.get("_system_message")

        @retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def query_bedrock_inner():
            client = self._get_client()

            # Convert to Bedrock message format
            bedrock_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": DEFAULT_MAX_TOKENS if budget_tokens else 8192,
                "messages": bedrock_messages,
            }

            if system_message:
                request_body["system"] = system_message

            # Enable extended thinking if budget_tokens specified
            if budget_tokens:
                request_body["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}

            response = client.invoke_model(modelId=model_id, body=json.dumps(request_body))

            result = json.loads(response["body"].read())

            # Handle extended thinking response format (multiple content blocks)
            content_blocks = result.get("content", [])
            text_content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    text_content = block.get("text", "")
                    break
                elif isinstance(block, dict) and "text" in block:
                    text_content = block["text"]
                    break

            # Fallback for simple response format
            if not text_content and content_blocks:
                if isinstance(content_blocks[0], str):
                    text_content = content_blocks[0]
                elif isinstance(content_blocks[0], dict):
                    text_content = content_blocks[0].get("text", "")

            return text_content

        # Run synchronous Bedrock call in thread pool with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(query_bedrock_inner),
            timeout=timeout,
        )
        return result
