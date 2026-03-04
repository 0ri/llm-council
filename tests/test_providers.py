"""Tests for LLM provider implementations."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from llm_council.models import BedrockModelConfig, PoeModelConfig
from llm_council.providers import ProviderRequest
from llm_council.providers.bedrock import BedrockProvider
from llm_council.providers.poe import PoeProvider

# Check if fastapi_poe is available
try:
    import fastapi_poe  # noqa: F401

    HAS_FASTAPI_POE = True
except ImportError:
    HAS_FASTAPI_POE = False


class TestBedrockProvider:
    """Test the Bedrock provider implementation."""

    @pytest.mark.asyncio
    async def test_basic_query_success(self):
        """Test successful Bedrock query with mocked boto3 client."""
        with patch("boto3.client") as mock_boto3_client:
            # Setup mock client
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client

            # Mock response
            mock_response = {
                "body": Mock(
                    read=Mock(
                        return_value=json.dumps(
                            {
                                "content": [{"type": "text", "text": "Test response"}],
                                "usage": {"input_tokens": 10, "output_tokens": 5},
                            }
                        ).encode()
                    )
                )
            }
            mock_client.invoke_model.return_value = mock_response

            # Test query
            provider = BedrockProvider()
            config = BedrockModelConfig(name="Test Model", provider="bedrock", model_id="test-model")
            result, usage = await provider.query(config, timeout=30)

            assert result == "Test response"
            assert usage == {"input_tokens": 10, "output_tokens": 5}
            mock_client.invoke_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_extended_thinking_response(self):
        """Test parsing extended thinking response with multiple content blocks."""
        with patch("boto3.client") as mock_boto3_client:
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client

            # Mock extended thinking response format
            mock_response = {
                "body": Mock(
                    read=Mock(
                        return_value=json.dumps(
                            {
                                "content": [
                                    {"type": "thinking", "text": "Let me think about this..."},
                                    {"type": "text", "text": "Here is my answer"},
                                ],
                                "usage": {"input_tokens": 100, "output_tokens": 50},
                            }
                        ).encode()
                    )
                )
            }
            mock_client.invoke_model.return_value = mock_response

            provider = BedrockProvider()
            config = BedrockModelConfig(
                name="claude-opus", provider="bedrock", model_id="claude-opus", budget_tokens=10000
            )
            result, usage = await provider.query(config, timeout=60)

            # Should return the text block, not the thinking block
            assert result == "Here is my answer"
            assert usage == {"input_tokens": 100, "output_tokens": 50}

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that timeout is properly handled."""
        with patch("boto3.client") as mock_boto3_client:
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client

            # Make invoke_model block forever
            def slow_invoke(*args, **kwargs):
                import time

                time.sleep(10)  # Sleep longer than timeout

            mock_client.invoke_model = slow_invoke

            provider = BedrockProvider()
            config = BedrockModelConfig(name="test-model", provider="bedrock", model_id="test-model")
            with pytest.raises(asyncio.TimeoutError):
                await provider.query(config, timeout=0.1)

    @pytest.mark.asyncio
    async def test_retryable_error(self):
        """Test that retryable errors trigger retry logic."""
        with patch("boto3.client") as mock_boto3_client:
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client

            # First call fails with retryable error, second succeeds
            error_response = {"Error": {"Code": "ThrottlingException"}, "ResponseMetadata": {"HTTPStatusCode": 429}}
            success_body = json.dumps({"content": [{"type": "text", "text": "Success"}]}).encode()
            mock_client.invoke_model.side_effect = [
                ClientError(error_response, "invoke_model"),
                {"body": Mock(read=Mock(return_value=success_body))},
            ]

            provider = BedrockProvider()
            config = BedrockModelConfig(name="test", provider="bedrock", model_id="test")
            result, _ = await provider.query(config, timeout=10)

            assert result == "Success"
            assert mock_client.invoke_model.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test that non-retryable errors fail immediately."""
        with patch("boto3.client") as mock_boto3_client:
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client

            # Non-retryable error
            error_response = {"Error": {"Code": "ValidationException"}, "ResponseMetadata": {"HTTPStatusCode": 400}}
            mock_client.invoke_model.side_effect = ClientError(error_response, "invoke_model")

            provider = BedrockProvider()
            config = BedrockModelConfig(name="test", provider="bedrock", model_id="test")
            with pytest.raises(ClientError):
                await provider.query(config, timeout=10)

            # Should not retry
            assert mock_client.invoke_model.call_count == 1

    @pytest.mark.asyncio
    async def test_model_id_extraction(self):
        """Test that model_id is properly extracted and used."""
        with patch("boto3.client") as mock_boto3_client:
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client
            mock_client.invoke_model.return_value = {
                "body": Mock(read=Mock(return_value=json.dumps({"content": [{"type": "text", "text": "OK"}]}).encode()))
            }

            provider = BedrockProvider()
            config = BedrockModelConfig(
                name="claude-opus", provider="bedrock", model_id="us.anthropic.claude-opus-4-6-v1:0"
            )
            await provider.query(config, timeout=10)

            # Check model_id was passed correctly
            call_args = mock_client.invoke_model.call_args
            assert call_args[1]["modelId"] == "us.anthropic.claude-opus-4-6-v1:0"

    @pytest.mark.asyncio
    async def test_budget_tokens_enables_thinking(self):
        """Test that budget_tokens enables extended thinking mode."""
        with patch("boto3.client") as mock_boto3_client:
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client
            mock_client.invoke_model.return_value = {
                "body": Mock(read=Mock(return_value=json.dumps({"content": [{"type": "text", "text": "OK"}]}).encode()))
            }

            provider = BedrockProvider()
            config = BedrockModelConfig(name="test", provider="bedrock", model_id="test", budget_tokens=5000)
            await provider.query(config, timeout=10)

            # Check that thinking mode was enabled
            call_args = mock_client.invoke_model.call_args
            body = json.loads(call_args[1]["body"])
            assert "thinking" in body
            assert body["thinking"]["type"] == "enabled"
            assert body["thinking"]["budget_tokens"] == 5000

    def test_region_configuration(self):
        """Test that region can be configured."""
        with patch("boto3.client") as mock_boto3_client:
            # Test default region
            provider1 = BedrockProvider()
            assert provider1.region == "us-east-1"

            # Test custom region
            provider2 = BedrockProvider(region="eu-west-1")
            assert provider2.region == "eu-west-1"

            # When client is created, it should use the region
            provider2._get_client()
            mock_boto3_client.assert_called_with("bedrock-runtime", region_name="eu-west-1")


@pytest.mark.skipif(not HAS_FASTAPI_POE, reason="fastapi_poe not installed")
class TestPoeProvider:
    """Test the Poe provider implementation."""

    @pytest.mark.asyncio
    async def test_basic_query_success(self):
        """Test successful Poe query with mocked API."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            # Mock streaming response
            async def mock_stream(*args, **kwargs):
                from fastapi_poe import PartialResponse

                yield PartialResponse(text="Hello ")
                yield PartialResponse(text="world")

            mock_get_bot.return_value = mock_stream()

            provider = PoeProvider(api_key="test-key")
            config = PoeModelConfig(name="TestBot", provider="poe", bot_name="TestBot")
            result, usage = await provider.query(config, timeout=30)

            assert result == "Hello world"
            assert usage is None  # Poe doesn't provide token counts
            mock_get_bot.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_accumulation(self):
        """Test that streaming responses are properly accumulated."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            # Mock longer streaming response
            async def mock_stream(*args, **kwargs):
                from fastapi_poe import PartialResponse

                chunks = ["This ", "is ", "a ", "streaming ", "response."]
                for chunk in chunks:
                    yield PartialResponse(text=chunk)

            mock_get_bot.return_value = mock_stream()

            provider = PoeProvider(api_key="test-key")
            config = PoeModelConfig(name="TestBot", provider="poe", bot_name="TestBot")
            result, _ = await provider.query(config, timeout=30)

            assert result == "This is a streaming response."

    @pytest.mark.asyncio
    async def test_gpt_web_search_flag(self):
        """Test that web_search flag is properly injected for GPT models."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            mock_get_bot.return_value = self._mock_empty_stream()

            provider = PoeProvider(api_key="test-key")
            config = PoeModelConfig(name="GPT-5.3-Codex", provider="poe", bot_name="GPT-5.3-Codex", web_search=True)
            req = ProviderRequest(messages=[{"role": "user", "content": "Hello"}])
            await provider.query(config, timeout=30, request=req)

            # Check that the flag was added to the message
            call_args = mock_get_bot.call_args
            messages = call_args[1]["messages"]
            assert "--web_search" in messages[0].content

    @pytest.mark.asyncio
    async def test_gemini_web_search_flag(self):
        """Test that web_search flag is properly formatted for Gemini models."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            mock_get_bot.return_value = self._mock_empty_stream()

            provider = PoeProvider(api_key="test-key")
            config = PoeModelConfig(name="Gemini-3.1-Pro", provider="poe", bot_name="Gemini-3.1-Pro", web_search=True)
            req = ProviderRequest(messages=[{"role": "user", "content": "Hello"}])
            await provider.query(config, timeout=30, request=req)

            call_args = mock_get_bot.call_args
            messages = call_args[1]["messages"]
            assert "--web_search true" in messages[0].content

    @pytest.mark.asyncio
    async def test_gpt_reasoning_effort(self):
        """Test reasoning_effort flag for GPT models."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            mock_get_bot.return_value = self._mock_empty_stream()

            provider = PoeProvider(api_key="test-key")
            config = PoeModelConfig(
                name="GPT-5.3-Codex", provider="poe", bot_name="GPT-5.3-Codex", reasoning_effort="high"
            )
            req = ProviderRequest(messages=[{"role": "user", "content": "Hello"}])
            await provider.query(config, timeout=30, request=req)

            call_args = mock_get_bot.call_args
            messages = call_args[1]["messages"]
            assert "--reasoning_effort high" in messages[0].content

    @pytest.mark.asyncio
    async def test_gemini_thinking_level(self):
        """Test that Gemini uses thinking_level instead of reasoning_effort."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            mock_get_bot.return_value = self._mock_empty_stream()

            provider = PoeProvider(api_key="test-key")
            config = PoeModelConfig(
                name="Gemini-3.1-Pro", provider="poe", bot_name="Gemini-3.1-Pro", reasoning_effort="high"
            )
            req = ProviderRequest(messages=[{"role": "user", "content": "Hello"}])
            await provider.query(config, timeout=30, request=req)

            call_args = mock_get_bot.call_args
            messages = call_args[1]["messages"]
            assert "--thinking_level high" in messages[0].content
            assert "--reasoning_effort" not in messages[0].content

    @pytest.mark.asyncio
    async def test_multiple_flags(self):
        """Test multiple flags are properly combined."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            mock_get_bot.return_value = self._mock_empty_stream()

            provider = PoeProvider(api_key="test-key")
            config = PoeModelConfig(
                name="GPT-5.3-Codex",
                provider="poe",
                bot_name="GPT-5.3-Codex",
                web_search=True,
                reasoning_effort="high",
            )
            req = ProviderRequest(messages=[{"role": "user", "content": "Hello"}])
            await provider.query(config, timeout=30, request=req)

            call_args = mock_get_bot.call_args
            messages = call_args[1]["messages"]
            content = messages[0].content
            assert "--web_search" in content
            assert "--reasoning_effort high" in content

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test that rate limit errors trigger retry."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            # First call fails with rate limit, second succeeds
            mock_get_bot.side_effect = [
                Exception("429 Too Many Requests"),
                self._mock_simple_stream("Success"),
            ]

            provider = PoeProvider(api_key="test-key")
            config = PoeModelConfig(name="TestBot", provider="poe", bot_name="TestBot")
            result, _ = await provider.query(config, timeout=30)

            assert result == "Success"
            assert mock_get_bot.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_auth_error(self):
        """Test that auth errors fail immediately without retry."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            mock_get_bot.side_effect = Exception("401 Unauthorized")

            provider = PoeProvider(api_key="test-key")
            config = PoeModelConfig(name="TestBot", provider="poe", bot_name="TestBot")
            with pytest.raises(Exception, match="401"):
                await provider.query(config, timeout=30)

            # Should not retry
            assert mock_get_bot.call_count == 1

    @pytest.mark.asyncio
    async def test_bot_not_found_error(self):
        """Test that bot not found errors fail immediately."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            mock_get_bot.side_effect = Exception("Bot 'InvalidBot' does not exist")

            provider = PoeProvider(api_key="test-key")
            config = PoeModelConfig(name="InvalidBot", provider="poe", bot_name="InvalidBot")
            with pytest.raises(Exception, match="Bot"):
                await provider.query(config, timeout=30)

            assert mock_get_bot.call_count == 1

    @pytest.mark.asyncio
    async def test_system_message_handling(self):
        """Test that system messages are properly included via ProviderRequest."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            mock_get_bot.return_value = self._mock_empty_stream()

            provider = PoeProvider(api_key="test-key")
            request = ProviderRequest(
                messages=[{"role": "user", "content": "Hello"}],
                system_message="You are a helpful assistant",
            )
            config = PoeModelConfig(name="TestBot", provider="poe", bot_name="TestBot")
            await provider.query(config, timeout=30, request=request)

            call_args = mock_get_bot.call_args
            messages = call_args[1]["messages"]
            assert messages[0].role == "system"
            assert messages[0].content == "You are a helpful assistant"
            assert messages[1].role == "user"

    @pytest.mark.asyncio
    async def test_assistant_to_bot_role_conversion(self):
        """Test that 'assistant' role is converted to 'bot' for Poe."""
        with patch("fastapi_poe.get_bot_response") as mock_get_bot:
            mock_get_bot.return_value = self._mock_empty_stream()

            provider = PoeProvider(api_key="test-key")
            request = ProviderRequest(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                ],
            )
            config = PoeModelConfig(name="TestBot", provider="poe", bot_name="TestBot")
            await provider.query(config, timeout=30, request=request)

            call_args = mock_get_bot.call_args
            messages = call_args[1]["messages"]
            assert messages[0].role == "user"
            assert messages[1].role == "bot"  # Converted from 'assistant'
            assert messages[2].role == "user"

    async def _mock_empty_stream(self):
        """Helper to create an empty async generator."""
        from fastapi_poe import PartialResponse

        yield PartialResponse(text="")

    def _mock_simple_stream(self, text):
        """Helper to create a simple streaming response."""

        async def stream():
            from fastapi_poe import PartialResponse

            yield PartialResponse(text=text)

        return stream()
