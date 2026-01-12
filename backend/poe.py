"""Poe.com API client for making LLM requests using fastapi-poe."""

import asyncio
import logging
from typing import List, Dict, Any, Optional

import fastapi_poe as fp
from fastapi_poe import ProtocolMessage

logger = logging.getLogger(__name__)


class PoeAPIError(Exception):
    """Base exception for Poe API errors."""
    pass


class PoeAuthenticationError(PoeAPIError):
    """Raised when API key is invalid."""
    pass


class PoeBotNotFoundError(PoeAPIError):
    """Raised when the requested bot is not found."""
    pass


class PoeRateLimitError(PoeAPIError):
    """Raised when rate limit is exceeded."""
    pass


class PoeServiceUnavailableError(PoeAPIError):
    """Raised when Poe service is unavailable."""
    pass


def _sanitize_error_message(error_msg: str, api_key: str) -> str:
    """Remove API key from error messages to prevent exposure."""
    if api_key and api_key in error_msg:
        return error_msg.replace(api_key, "[REDACTED]")
    return error_msg


def _classify_and_raise_error(error: Exception, bot_name: str, api_key: str) -> None:
    """
    Classify a Poe API error and raise the appropriate exception.
    
    Args:
        error: The original exception
        bot_name: The bot that was being queried
        api_key: The API key (for sanitization)
    """
    error_str = str(error).lower()
    sanitized_msg = _sanitize_error_message(str(error), api_key)
    
    # Check for authentication errors
    if any(term in error_str for term in ["unauthorized", "invalid api key", "authentication", "401"]):
        raise PoeAuthenticationError(
            f"Invalid Poe API key. Get one at poe.com/api_key"
        ) from error
    
    # Check for bot not found errors
    if any(term in error_str for term in ["bot not found", "not found", "404", "unknown bot"]):
        raise PoeBotNotFoundError(
            f"Bot '{bot_name}' not found on Poe.com"
        ) from error
    
    # Check for rate limit errors
    if any(term in error_str for term in ["rate limit", "too many requests", "429"]):
        raise PoeRateLimitError(
            "Rate limit exceeded. Please try again later."
        ) from error
    
    # Check for service unavailable / connection errors
    if any(term in error_str for term in ["connection", "timeout", "unavailable", "503", "502"]):
        raise PoeServiceUnavailableError(
            "Poe.com service unavailable"
        ) from error
    
    # Re-raise as generic PoeAPIError with sanitized message
    raise PoeAPIError(sanitized_msg) from error


async def query_model(
    bot_name: str,
    messages: List[Dict[str, str]],
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Query a single Poe bot and return accumulated response.

    Args:
        bot_name: Poe bot display name (e.g., "GPT-5", "Claude-Sonnet-4.5")
        messages: List of message dicts with 'role' and 'content'
        api_key: Poe API key from poe.com/api_key

    Returns:
        Response dict with 'content' key, or None if failed
    """
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

    try:
        # Accumulate response chunks
        accumulated_text = ""
        async for partial in fp.get_bot_response(
            messages=protocol_messages,
            bot_name=bot_name,
            api_key=api_key,
        ):
            # Each PartialResponse.text contains just the next token
            accumulated_text += partial.text

        return {"content": accumulated_text}

    except (PoeAPIError, PoeAuthenticationError, PoeBotNotFoundError, 
            PoeRateLimitError, PoeServiceUnavailableError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Classify and handle the error appropriately
        try:
            _classify_and_raise_error(e, bot_name, api_key)
        except PoeAPIError as classified_error:
            # Log the classified error
            logger.error(f"Error querying bot {bot_name}: {classified_error}")
            return None


async def query_models_parallel(
    bot_names: List[str],
    messages: List[Dict[str, str]],
    api_key: str,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple Poe bots in parallel.

    Args:
        bot_names: List of Poe bot display names
        messages: List of message dicts to send to each bot
        api_key: Poe API key from poe.com/api_key

    Returns:
        Dict mapping bot_name to response dict (or None if failed)
    """
    async def safe_query(bot_name: str) -> Optional[Dict[str, Any]]:
        """Query a bot and catch exceptions to allow other queries to continue."""
        try:
            return await query_model(bot_name, messages, api_key)
        except PoeAPIError as e:
            logger.error(f"Error querying bot {bot_name}: {e}")
            return None
        except Exception as e:
            sanitized_msg = _sanitize_error_message(str(e), api_key)
            logger.error(f"Unexpected error querying bot {bot_name}: {sanitized_msg}")
            return None

    # Create tasks for all bots
    tasks = [safe_query(bot_name) for bot_name in bot_names]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map bot names to their responses
    return {bot_name: response for bot_name, response in zip(bot_names, responses)}
