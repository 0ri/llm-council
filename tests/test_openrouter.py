"""Tests for the OpenRouter provider."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_council.models import OpenRouterModelConfig
from llm_council.providers import ProviderRequest
from llm_council.providers.openrouter import (
    OpenRouterAPIError,
    OpenRouterProvider,
    is_retryable_openrouter_error,
)


class TestOpenRouterProvider:
    """Test the OpenRouter provider implementation."""

    @pytest.mark.asyncio
    async def test_basic_query_success(self):
        """Test successful OpenRouter query with mocked httpx."""
        provider = OpenRouterProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from OpenRouter"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8},
        }

        with patch.object(provider, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            config = OpenRouterModelConfig(name="GPT-4o", provider="openrouter", model_id="openai/gpt-4o")
            result, usage = await provider.query("Test prompt", config, timeout=30)

        assert result == "Hello from OpenRouter"
        assert usage == {"input_tokens": 15, "output_tokens": 8}

    @pytest.mark.asyncio
    async def test_token_usage_extraction(self):
        """Test that token usage is properly extracted from response."""
        provider = OpenRouterProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        with patch.object(provider, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            config = OpenRouterModelConfig(name="test", provider="openrouter", model_id="test")
            _, usage = await provider.query("Test", config, timeout=30)

        assert usage is not None
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_no_usage_in_response(self):
        """Test handling when response has no usage data."""
        provider = OpenRouterProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
        }

        with patch.object(provider, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            config = OpenRouterModelConfig(name="test", provider="openrouter", model_id="test")
            result, usage = await provider.query("Test", config, timeout=30)

        assert result == "Response"
        assert usage is None

    @pytest.mark.asyncio
    async def test_system_message_handling(self):
        """Test that system messages are sent correctly."""
        provider = OpenRouterProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
        }

        with patch.object(provider, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            config = OpenRouterModelConfig(name="test", provider="openrouter", model_id="test")
            request = ProviderRequest(
                messages=[{"role": "user", "content": "Hello"}],
                system_message="You are a helpful assistant",
            )
            await provider.query("", config, timeout=30, request=request)

            call_args = mock_client.post.call_args
            body = call_args[1]["json"]
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][0]["content"] == "You are a helpful assistant"
            assert body["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens(self):
        """Test that temperature and max_tokens are passed correctly."""
        provider = OpenRouterProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
        }

        with patch.object(provider, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            config = OpenRouterModelConfig(
                name="test", provider="openrouter", model_id="test", temperature=0.7, max_tokens=4096
            )
            await provider.query("Test", config, timeout=30)

            call_args = mock_client.post.call_args
            body = call_args[1]["json"]
            assert body["temperature"] == 0.7
            assert body["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        """Test that non-200 responses raise OpenRouterAPIError."""
        provider = OpenRouterProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(provider, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            config = OpenRouterModelConfig(name="test", provider="openrouter", model_id="test")
            with pytest.raises(OpenRouterAPIError) as exc_info:
                await provider.query("Test", config, timeout=30)

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test listing available models."""
        provider = OpenRouterProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "openai/gpt-4o", "name": "GPT-4o"},
                {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
            ]
        }

        with patch.object(provider, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            models = await provider.list_models()

        assert len(models) == 2
        assert models[0]["id"] == "openai/gpt-4o"
        assert models[1]["name"] == "Claude 3.5 Sonnet"

    @pytest.mark.asyncio
    async def test_list_models_error(self):
        """Test that list_models raises on API error."""
        provider = OpenRouterProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(provider, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            with pytest.raises(OpenRouterAPIError):
                await provider.list_models()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test that close properly shuts down the client."""
        provider = OpenRouterProvider(api_key="test-key")

        mock_client = AsyncMock()
        provider._client = mock_client

        await provider.close()

        mock_client.aclose.assert_awaited_once()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_empty_choices(self):
        """Test handling of empty choices array."""
        provider = OpenRouterProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}

        with patch.object(provider, "_get_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            config = OpenRouterModelConfig(name="test", provider="openrouter", model_id="test")
            result, _ = await provider.query("Test", config, timeout=30)

        assert result == ""


class TestRetryableErrors:
    """Test the is_retryable_openrouter_error function."""

    def test_timeout_is_retryable(self):
        assert is_retryable_openrouter_error(asyncio.TimeoutError()) is True

    def test_httpx_timeout_is_retryable(self):
        assert is_retryable_openrouter_error(httpx.TimeoutException("timeout")) is True

    def test_connect_error_is_retryable(self):
        assert is_retryable_openrouter_error(httpx.ConnectError("connection failed")) is True

    def test_429_is_retryable(self):
        err = OpenRouterAPIError(429, "Rate limited")
        assert is_retryable_openrouter_error(err) is True

    def test_500_is_retryable(self):
        err = OpenRouterAPIError(500, "Internal error")
        assert is_retryable_openrouter_error(err) is True

    def test_502_is_retryable(self):
        err = OpenRouterAPIError(502, "Bad gateway")
        assert is_retryable_openrouter_error(err) is True

    def test_401_is_not_retryable(self):
        err = OpenRouterAPIError(401, "Unauthorized")
        assert is_retryable_openrouter_error(err) is False

    def test_403_is_not_retryable(self):
        err = OpenRouterAPIError(403, "Forbidden")
        assert is_retryable_openrouter_error(err) is False

    def test_400_is_not_retryable(self):
        err = OpenRouterAPIError(400, "Bad request")
        assert is_retryable_openrouter_error(err) is False

    def test_unknown_error_not_retryable(self):
        assert is_retryable_openrouter_error(ValueError("some error")) is False
