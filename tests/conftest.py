"""Shared test fixtures for LLM Council tests."""

import logging

import pytest


@pytest.fixture(autouse=True)
def _reset_logger():
    """Clean up llm-council logger handlers between tests."""
    logger = logging.getLogger("llm-council")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)

    yield

    logger.handlers.clear()


@pytest.fixture
def mock_ctx():
    """Create a CouncilContext with mock settings for testing."""
    from llm_council.context import CouncilContext
    from llm_council.cost import CouncilCostTracker
    from llm_council.progress import ProgressManager

    return CouncilContext(
        poe_api_key="test-key",
        cost_tracker=CouncilCostTracker(),
        progress=ProgressManager(is_tty=False),
    )


@pytest.fixture
def make_ctx():
    """Factory fixture for creating a CouncilContext with optional overrides.

    Returns a callable: ``make_ctx(**overrides) -> CouncilContext``.
    Attribute overrides are applied via ``setattr`` after construction.
    """
    from llm_council.context import CouncilContext
    from llm_council.cost import CouncilCostTracker
    from llm_council.progress import ProgressManager

    def _factory(**overrides) -> CouncilContext:
        ctx = CouncilContext(
            poe_api_key="test-key",
            cost_tracker=CouncilCostTracker(),
            progress=ProgressManager(is_tty=False),
        )
        for k, v in overrides.items():
            setattr(ctx, k, v)
        return ctx

    return _factory


@pytest.fixture
def make_ctx_factory():
    """Factory fixture for creating context_factory callables.

    Returns a callable:
        ``make_ctx_factory(mock_provider, cache=None) -> Callable[[], CouncilContext]``

    The returned factory pre-injects *mock_provider* for "poe", "bedrock",
    and "openrouter" provider slots.
    """
    from llm_council.context import CouncilContext
    from llm_council.cost import CouncilCostTracker
    from llm_council.progress import ProgressManager

    def _factory(mock_provider, cache=None):
        def ctx_factory():
            ctx = CouncilContext(
                poe_api_key="test-key",
                cost_tracker=CouncilCostTracker(),
                progress=ProgressManager(is_tty=False),
                cache=cache,
            )
            ctx.providers["poe"] = mock_provider
            ctx.providers["bedrock"] = mock_provider
            ctx.providers["openrouter"] = mock_provider
            return ctx

        return ctx_factory

    return _factory


@pytest.fixture
def sample_config():
    """Valid council configuration for testing."""
    return {
        "council_models": [
            {"name": "Model-A", "provider": "poe", "bot_name": "TestBot-A"},
            {"name": "Model-B", "provider": "poe", "bot_name": "TestBot-B"},
            {"name": "Model-C", "provider": "poe", "bot_name": "TestBot-C"},
        ],
        "chairman": {"name": "Model-A", "provider": "poe", "bot_name": "TestBot-A"},
    }


@pytest.fixture
def sample_stage1_results():
    """Sample Stage 1 results for testing."""
    from llm_council.models import Stage1Result

    return [
        Stage1Result(model="Model-A", response="Response from model A about the topic."),
        Stage1Result(model="Model-B", response="Response from model B with different perspective."),
        Stage1Result(model="Model-C", response="Response from model C with yet another view."),
    ]


@pytest.fixture
def sample_label_to_model():
    """Sample label-to-model mapping."""
    return {
        "Response A": "Model-A",
        "Response B": "Model-B",
        "Response C": "Model-C",
    }
