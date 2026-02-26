"""AWS Bedrock provider for Anthropic models."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from . import StreamResult

logger = logging.getLogger("llm-council")


def is_retryable_bedrock_error(exc: BaseException) -> bool:
    """Determine if a Bedrock error is retryable."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, ClientError):
        error_code = exc.response.get("Error", {}).get("Code", "")
        # Retryable: rate limit, throttling, server errors
        retryable_codes = {
            "ThrottlingException",
            "TooManyRequestsException",
            "ServiceUnavailableException",
            "InternalServerException",
            "RequestTimeoutException",
        }
        # Non-retryable: auth, validation, not found
        non_retryable_codes = {
            "AccessDeniedException",
            "UnauthorizedException",
            "ValidationException",
            "ResourceNotFoundException",
            "ModelNotReadyException",
        }
        if error_code in retryable_codes:
            return True
        if error_code in non_retryable_codes:
            logger.warning(f"Non-retryable Bedrock error: {error_code}")
            return False
        # For 5xx status codes
        status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        if 500 <= status_code < 600:
            return True
        if status_code == 429:  # Rate limit
            return True
    # Connection errors are retryable
    if "Connection" in str(type(exc).__name__):
        return True
    return False


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

    async def query(self, prompt: str, model_config: dict, timeout: int) -> tuple[str, dict | None]:
        """Query Claude via AWS Bedrock with timeout and retry logic.

        Returns:
            Tuple of (response text, token usage dict or None)
        """
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
            retry=retry_if_exception(is_retryable_bedrock_error),
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

            # Extract token usage from response
            token_usage = None
            if "usage" in result:
                usage = result["usage"]
                token_usage = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                }

            # Handle extended thinking response format (multiple content blocks)
            # Skip "thinking" blocks — concatenate all "text" type blocks
            content_blocks = result.get("content", [])
            text_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            text_content = "\n\n".join(text_parts) if text_parts else ""

            return text_content, token_usage

        # Run synchronous Bedrock call in thread pool with timeout
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(query_bedrock_inner),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Bedrock timeout for model {model_config.get('name', 'unknown')}")
            raise

    def astream(self, prompt: str, model_config: dict, timeout: int) -> StreamResult:
        """Stream Claude via AWS Bedrock using invoke_model_with_response_stream.

        Returns:
            StreamResult wrapping an async generator of text chunks.
        """
        from . import DEFAULT_MAX_TOKENS, MAX_RETRIES, StreamResult

        model_id = model_config["model_id"]
        budget_tokens = model_config.get("budget_tokens")
        messages = model_config.get("_messages", [{"role": "user", "content": prompt}])
        system_message = model_config.get("_system_message")

        @retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception(is_retryable_bedrock_error),
            reraise=True,
        )
        def stream_bedrock_inner():
            client = self._get_client()
            bedrock_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": DEFAULT_MAX_TOKENS if budget_tokens else 8192,
                "messages": bedrock_messages,
            }

            if system_message:
                request_body["system"] = system_message

            if budget_tokens:
                request_body["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}

            response = client.invoke_model_with_response_stream(modelId=model_id, body=json.dumps(request_body))
            return response

        usage_info: dict[str, int] = {}

        async def _generate():
            response = await asyncio.wait_for(
                asyncio.to_thread(stream_bedrock_inner),
                timeout=timeout,
            )
            stream = response.get("body", [])
            for event in stream:
                chunk = event.get("chunk")
                if not chunk:
                    continue
                data = json.loads(chunk.get("bytes", b"{}"))
                event_type = data.get("type", "")

                if event_type == "message_start":
                    msg = data.get("message", {})
                    msg_usage = msg.get("usage", {})
                    if "input_tokens" in msg_usage:
                        usage_info["input_tokens"] = msg_usage["input_tokens"]

                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text

                elif event_type == "message_delta":
                    delta_usage = data.get("usage", {})
                    if "output_tokens" in delta_usage:
                        usage_info["output_tokens"] = delta_usage["output_tokens"]

        gen = _generate()
        result = StreamResult(gen)

        original_anext = gen.__anext__

        async def patched_anext():
            try:
                return await original_anext()
            except StopAsyncIteration:
                if usage_info:
                    result.usage = usage_info
                raise

        result._aiter.__anext__ = patched_anext  # type: ignore[attr-defined]

        return result
