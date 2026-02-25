"""Dependency-injection container for per-run council state.

CouncilContext replaces the module-level global caches in
``providers/__init__.py`` with an instance-scoped container so that
concurrent council runs do not share mutable state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .budget import BudgetGuard
from .cache import ResponseCache
from .cost import CouncilCostTracker
from .progress import ProgressManager
from .providers import CircuitBreaker, Provider
from .providers.bedrock import BedrockProvider
from .providers.openrouter import OpenRouterProvider
from .providers.poe import PoeProvider


@dataclass
class CouncilContext:
    """Per-run state container for a single council execution.

    Every council run should receive its own ``CouncilContext`` so that
    provider caches, circuit breakers, and progress tracking are isolated.
    """

    poe_api_key: str | None = None
    openrouter_api_key: str | None = None
    max_concurrent: int = 4
    stage2_max_retries: int = 1
    providers: dict[str, Provider] = field(default_factory=dict)
    circuit_breakers: dict[str, CircuitBreaker] = field(default_factory=dict)
    semaphore: asyncio.Semaphore | None = None
    cost_tracker: CouncilCostTracker = field(default_factory=CouncilCostTracker)
    budget_guard: BudgetGuard | None = None
    progress: ProgressManager = field(default_factory=ProgressManager)
    cache: ResponseCache | None = None

    # ------------------------------------------------------------------
    # Lazy accessor helpers
    # ------------------------------------------------------------------

    def get_semaphore(self) -> asyncio.Semaphore:
        """Return the concurrency semaphore, creating it lazily."""
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_concurrent)
        return self.semaphore

    def get_provider(self, provider_name: str) -> Provider:
        """Return a cached provider instance, creating it lazily.

        Raises ``ValueError`` for unknown provider names or if the Poe
        provider is requested without an API key.
        """
        if provider_name not in self.providers:
            if provider_name == "bedrock":
                self.providers[provider_name] = BedrockProvider()
            elif provider_name == "poe":
                if not self.poe_api_key:
                    raise ValueError("POE_API_KEY required for Poe provider")
                self.providers[provider_name] = PoeProvider(self.poe_api_key)
            elif provider_name == "openrouter":
                if not self.openrouter_api_key:
                    raise ValueError("OPENROUTER_API_KEY required for OpenRouter provider")
                self.providers[provider_name] = OpenRouterProvider(self.openrouter_api_key)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")
        return self.providers[provider_name]

    async def close(self) -> None:
        """Close all providers and the cache."""
        for provider in self.providers.values():
            close = getattr(provider, "close", None)
            if close is not None and asyncio.iscoroutinefunction(close):
                await close()
        if self.cache is not None:
            self.cache.close()

    def get_circuit_breaker(self, identifier: str) -> CircuitBreaker:
        """Return a cached circuit breaker for *identifier*, creating it lazily."""
        if identifier not in self.circuit_breakers:
            self.circuit_breakers[identifier] = CircuitBreaker()
        return self.circuit_breakers[identifier]
