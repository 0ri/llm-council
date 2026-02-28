"""Dependency-injection container for per-run council state.

CouncilContext replaces the module-level global caches in
``providers/__init__.py`` with an instance-scoped container so that
concurrent council runs do not share mutable state.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from .budget import BudgetGuard
from .cache import ResponseCache
from .cost import CouncilCostTracker
from .progress import ProgressManager
from .providers import CircuitBreaker, Provider
from .providers.bedrock import BedrockProvider
from .providers.openrouter import OpenRouterProvider
from .providers.poe import PoeProvider

logger = logging.getLogger(__name__)

# Item 21: Provider registry mapping provider names to their classes.
# Keeps the if/elif dispatch in get_provider data-driven and extensible.
PROVIDER_REGISTRY: dict[str, type[Provider]] = {
    "bedrock": BedrockProvider,
    "poe": PoeProvider,
    "openrouter": OpenRouterProvider,
}

# API key requirements per provider (field name on CouncilContext -> display name)
_PROVIDER_API_KEYS: dict[str, str] = {
    "poe": "poe_api_key",
    "openrouter": "openrouter_api_key",
}


@dataclass
class CouncilContext:
    """Dependency-injection container holding per-run council state.

    Every council run should receive its own ``CouncilContext`` so that
    provider instances, circuit breakers, concurrency limits, and
    progress tracking are fully isolated between concurrent runs.

    Providers are created lazily via ``get_provider`` and cached for the
    lifetime of the context.  Call ``shutdown`` (or use as an async
    context manager) to release all resources when the run is complete.

    Args:
        poe_api_key: API key for the Poe provider.  Required if any
            council model uses the ``poe`` provider; otherwise ``None``.
        openrouter_api_key: API key for the OpenRouter provider.
            Required if any council model uses the ``openrouter``
            provider; otherwise ``None``.
        max_concurrent: Maximum number of concurrent provider requests
            enforced by the internal semaphore.  Defaults to 4.
        stage2_max_retries: Number of retry attempts for Stage 2 ranking
            requests that fail or produce invalid ballots.  Defaults to 1.
        providers: Pre-populated provider cache.  Normally left empty so
            that providers are created lazily by ``get_provider``.
        circuit_breakers: Pre-populated circuit-breaker cache.  Normally
            left empty so breakers are created lazily.
        semaphore: Explicit ``asyncio.Semaphore`` override.  When
            ``None``, a semaphore is created lazily from *max_concurrent*.
        cost_tracker: ``CouncilCostTracker`` instance used to accumulate
            token and cost metrics across all stages.
        budget_guard: Optional ``BudgetGuard`` that enforces token and
            cost limits.  ``None`` means no budget enforcement.
        progress: ``ProgressManager`` for emitting human-readable
            progress updates during the run.
        cache: Optional ``ResponseCache`` for Stage 1 response caching.
            ``None`` disables caching.

    Raises:
        ValueError: From ``get_provider`` if an unknown provider name is
            requested or if a required API key is missing.
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
    _shutdown_called: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Async context manager (Item 16)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> CouncilContext:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()

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

        Uses ``PROVIDER_REGISTRY`` for dispatch.  Raises ``ValueError``
        for unknown provider names or if a required API key is missing.
        """
        if provider_name not in self.providers:
            provider_cls = PROVIDER_REGISTRY.get(provider_name)
            if provider_cls is None:
                raise ValueError(f"Unknown provider: {provider_name}")

            # Check API key requirements
            api_key_attr = _PROVIDER_API_KEYS.get(provider_name)
            if api_key_attr is not None:
                api_key = getattr(self, api_key_attr, None)
                if not api_key:
                    env_var = api_key_attr.upper()  # e.g. poe_api_key -> POE_API_KEY
                    raise ValueError(f"{env_var} required for {provider_name.title()} provider")
                self.providers[provider_name] = provider_cls(api_key)
            else:
                self.providers[provider_name] = provider_cls()

        return self.providers[provider_name]

    async def close(self) -> None:
        """Close all providers and the cache."""
        for provider in self.providers.values():
            close = getattr(provider, "close", None)
            if close is not None and asyncio.iscoroutinefunction(close):
                await close()
        if self.cache is not None:
            self.cache.close()

    async def shutdown(self) -> None:
        """Unified shutdown: close providers, cache, and progress manager.

        Catches and logs exceptions at each step so all resources are
        released regardless of individual failures. Idempotent.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True

        try:
            await self.close()
        except Exception:
            logger.exception("Error during close()")

        try:
            await self.progress.shutdown()
        except Exception:
            logger.exception("Error during progress shutdown")

    def get_circuit_breaker(self, identifier: str) -> CircuitBreaker:
        """Return a cached circuit breaker for *identifier*, creating it lazily."""
        if identifier not in self.circuit_breakers:
            self.circuit_breakers[identifier] = CircuitBreaker()
        return self.circuit_breakers[identifier]
