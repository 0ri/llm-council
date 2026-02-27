"""OpenRouter provider — OpenAI-compatible API with hundreds of models.

Implements ``OpenRouterProvider`` with ``query()``, ``astream()`` (SSE-based
streaming), and ``list_models()`` for model discovery. Communicates over HTTPS
using ``httpx``, with retry logic for rate-limit and server errors. Unlike Poe,
OpenRouter returns real token usage data, enabling accurate cost tracking in the
council pipeline.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from typing import TYPE_CHECKING, Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from . import ProviderRequest, StreamResult, UsageTrackingStream

logger = logging.getLogger("llm-council")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def is_retryable_openrouter_error(exc: BaseException) -> bool:
    """Determine if an OpenRouter error is retryable."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, OpenRouterAPIError):
        # 429 rate limit and 5xx server errors are retryable
        if exc.status_code == 429 or 500 <= exc.status_code < 600:
            return True
        # 401/403/400 are not retryable
        if exc.status_code in (400, 401, 403):
            logger.warning(f"Non-retryable OpenRouter error: {exc.status_code} {exc.message}")
            return False
    if isinstance(exc, httpx.ConnectError):
        return True
    return False


class OpenRouterAPIError(Exception):
    """Error from the OpenRouter API."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"OpenRouter API error {status_code}: {message}")


class OpenRouterProvider:
    """OpenRouter provider — OpenAI-compatible API with real token usage reporting."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the httpx async client, creating it on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(600.0, connect=30.0),
            )
        return self._client

    @staticmethod
    def _build_body(model_config: dict, api_messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build the request body from model config and messages."""
        body: dict[str, Any] = {
            "model": model_config["model_id"],
            "messages": api_messages,
            "max_tokens": model_config.get("max_tokens", 16384),
        }
        if model_config.get("temperature") is not None:
            body["temperature"] = model_config["temperature"]

        # Build reasoning config if either effort or max_tokens is set
        reasoning_effort = model_config.get("reasoning_effort")
        reasoning_max_tokens = model_config.get("reasoning_max_tokens")
        if reasoning_effort or reasoning_max_tokens:
            reasoning: dict[str, Any] = {}
            if reasoning_max_tokens:
                reasoning["max_tokens"] = reasoning_max_tokens
            elif reasoning_effort:
                reasoning["effort"] = reasoning_effort
            body["reasoning"] = reasoning

        return body

    async def query(
        self,
        prompt: str,
        model_config: dict,
        timeout: int,
        request: ProviderRequest | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Query a model via OpenRouter.

        Returns:
            Tuple of (response text, token usage dict or None).
            OpenRouter provides real usage data: {"input_tokens": N, "output_tokens": M}.
        """
        from . import MAX_RETRIES

        # Use typed request if provided, fall back to legacy model_config keys
        if request is not None:
            messages = request.messages
            system_message = request.system_message
        else:
            messages = model_config.get("_messages", [{"role": "user", "content": prompt}])
            system_message = model_config.get("_system_message")

        @retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception(is_retryable_openrouter_error),
            reraise=True,
        )
        async def _query_inner() -> tuple[str, dict[str, Any] | None]:
            client = self._get_client()

            # Build messages in OpenAI format
            api_messages: list[dict[str, str]] = []
            if system_message:
                api_messages.append({"role": "system", "content": system_message})
            for msg in messages:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

            body = self._build_body(model_config, api_messages)

            resp = await client.post("/chat/completions", json=body, timeout=timeout)

            if resp.status_code != 200:
                error_msg = resp.text[:500]
                raise OpenRouterAPIError(resp.status_code, error_msg)

            data = resp.json()

            # Extract text
            text = ""
            choices = data.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")

            # Extract token usage — OpenRouter provides real counts
            token_usage = None
            usage = data.get("usage")
            if usage:
                token_usage = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                }

            return text, token_usage

        return await _query_inner()

    def astream(
        self,
        prompt: str,
        model_config: dict,
        timeout: int,
        request: ProviderRequest | None = None,
    ) -> StreamResult:
        """Stream a model response via OpenRouter using SSE.

        Returns:
            StreamResult wrapping an async generator of text chunks.
        """
        from . import MAX_RETRIES, StreamResult, UsageTrackingStream

        # Use typed request if provided, fall back to legacy model_config keys
        if request is not None:
            messages = request.messages
            system_message = request.system_message
        else:
            messages = model_config.get("_messages", [{"role": "user", "content": prompt}])
            system_message = model_config.get("_system_message")

        usage_info: dict[str, int] = {}

        async def _generate():
            @retry(
                stop=stop_after_attempt(MAX_RETRIES),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                retry=retry_if_exception(is_retryable_openrouter_error),
                reraise=True,
            )
            async def _stream_inner():
                client = self._get_client()

                api_messages: list[dict[str, str]] = []
                if system_message:
                    api_messages.append({"role": "system", "content": system_message})
                for msg in messages:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})

                body = self._build_body(model_config, api_messages)
                body["stream"] = True

                return await client.send(
                    client.build_request("POST", "/chat/completions", json=body),
                    stream=True,
                )

            resp = await asyncio.wait_for(_stream_inner(), timeout=timeout)

            try:
                if resp.status_code != 200:
                    error_text = ""
                    async for chunk in resp.aiter_text():
                        error_text += chunk
                        if len(error_text) > 500:
                            break
                    raise OpenRouterAPIError(resp.status_code, error_text[:500])

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        data = _json.loads(payload)
                    except _json.JSONDecodeError:
                        continue

                    # Extract usage from the final chunk if present
                    chunk_usage = data.get("usage")
                    if chunk_usage:
                        usage_info["input_tokens"] = chunk_usage.get("prompt_tokens", 0)
                        usage_info["output_tokens"] = chunk_usage.get("completion_tokens", 0)

                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
            finally:
                await resp.aclose()

            # Set usage on the wrapper when the generator finishes normally
            if usage_info:
                wrapper.set_usage(usage_info)

        gen = _generate()
        result = StreamResult(gen)
        wrapper = UsageTrackingStream(gen, result)
        result._aiter = wrapper

        return result

    async def list_models(self) -> list[dict[str, str]]:
        """List available models from OpenRouter.

        Returns:
            List of dicts with 'id' and 'name' keys.
        """
        client = self._get_client()
        resp = await client.get("/models")
        if resp.status_code != 200:
            raise OpenRouterAPIError(resp.status_code, resp.text[:500])

        data = resp.json()
        models = []
        for model in data.get("data", []):
            models.append(
                {
                    "id": model.get("id", ""),
                    "name": model.get("name", model.get("id", "")),
                }
            )
        return models

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
