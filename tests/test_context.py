"""Tests for the CouncilContext dependency injection container."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from llm_council.context import CouncilContext
from llm_council.cost import CouncilCostTracker
from llm_council.progress import ProgressManager
from llm_council.providers import CircuitBreaker
from llm_council.providers.bedrock import BedrockProvider
from llm_council.providers.poe import PoeProvider


class TestGetProvider:
    """Test CouncilContext.get_provider."""

    def test_bedrock_provider_creation(self):
        """get_provider('bedrock') returns a BedrockProvider."""
        with patch("boto3.client"):
            ctx = CouncilContext()
            provider = ctx.get_provider("bedrock")
            assert isinstance(provider, BedrockProvider)

    def test_poe_provider_requires_api_key(self):
        """get_provider('poe') raises ValueError without api_key."""
        ctx = CouncilContext(poe_api_key=None)
        with pytest.raises(ValueError, match="POE_API_KEY required"):
            ctx.get_provider("poe")

    def test_poe_provider_with_api_key(self):
        """get_provider('poe') works when api_key is set."""
        ctx = CouncilContext(poe_api_key="test-key")
        provider = ctx.get_provider("poe")
        assert isinstance(provider, PoeProvider)

    def test_provider_caching(self):
        """Same provider instance returned on repeat call."""
        with patch("boto3.client"):
            ctx = CouncilContext()
            p1 = ctx.get_provider("bedrock")
            p2 = ctx.get_provider("bedrock")
            assert p1 is p2

    def test_unknown_provider_raises(self):
        """get_provider with unknown name raises ValueError."""
        ctx = CouncilContext()
        with pytest.raises(ValueError, match="Unknown provider"):
            ctx.get_provider("nonexistent")


class TestGetCircuitBreaker:
    """Test CouncilContext.get_circuit_breaker."""

    def test_creates_new_circuit_breaker(self):
        """get_circuit_breaker creates a new CircuitBreaker for a new key."""
        ctx = CouncilContext()
        cb = ctx.get_circuit_breaker("key1")
        assert isinstance(cb, CircuitBreaker)
        assert not cb.is_open

    def test_circuit_breaker_caching(self):
        """Same CircuitBreaker returned on repeat call with same key."""
        ctx = CouncilContext()
        cb1 = ctx.get_circuit_breaker("key1")
        cb2 = ctx.get_circuit_breaker("key1")
        assert cb1 is cb2

    def test_different_keys_different_breakers(self):
        """Different keys produce different CircuitBreaker instances."""
        ctx = CouncilContext()
        cb1 = ctx.get_circuit_breaker("key1")
        cb2 = ctx.get_circuit_breaker("key2")
        assert cb1 is not cb2


class TestGetSemaphore:
    """Test CouncilContext.get_semaphore."""

    def test_lazy_creation(self):
        """Semaphore is None until get_semaphore() is called."""
        ctx = CouncilContext()
        assert ctx.semaphore is None
        sem = ctx.get_semaphore()
        assert isinstance(sem, asyncio.Semaphore)
        assert ctx.semaphore is sem

    def test_returns_same_instance(self):
        """Repeat calls return the same Semaphore."""
        ctx = CouncilContext()
        s1 = ctx.get_semaphore()
        s2 = ctx.get_semaphore()
        assert s1 is s2


class TestInstanceIsolation:
    """Two CouncilContext instances must not share state."""

    def test_providers_isolated(self):
        """Providers dict is not shared between instances."""
        with patch("boto3.client"):
            ctx1 = CouncilContext()
            ctx2 = CouncilContext()
            ctx1.get_provider("bedrock")
            assert "bedrock" not in ctx2.providers

    def test_circuit_breakers_isolated(self):
        """Circuit breakers dict is not shared between instances."""
        ctx1 = CouncilContext()
        ctx2 = CouncilContext()
        ctx1.get_circuit_breaker("x")
        assert "x" not in ctx2.circuit_breakers

    def test_semaphore_isolated(self):
        """Semaphore is not shared between instances."""
        ctx1 = CouncilContext()
        ctx2 = CouncilContext()
        s1 = ctx1.get_semaphore()
        s2 = ctx2.get_semaphore()
        assert s1 is not s2
