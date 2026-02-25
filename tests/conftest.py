"""Shared test fixtures for LLM Council tests."""

import logging
import sys
from pathlib import Path

import pytest

# Try importing from installed package first, fall back to scripts directory
try:
    import llm_council  # noqa: F401 -- verify package is importable
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "skills" / "council" / "scripts"))


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Reset all module-level global state between tests to prevent leakage."""
    from llm_council.providers import reset_circuit_breakers, reset_providers, reset_semaphore

    # Reset before test
    reset_providers()
    reset_circuit_breakers()
    reset_semaphore()

    # Clean up llm-council logger handlers to prevent handler accumulation
    logger = logging.getLogger("llm-council")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)

    yield

    # Reset after test
    reset_providers()
    reset_circuit_breakers()
    reset_semaphore()
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
    return [
        {"model": "Model-A", "response": "Response from model A about the topic."},
        {"model": "Model-B", "response": "Response from model B with different perspective."},
        {"model": "Model-C", "response": "Response from model C with yet another view."},
    ]


@pytest.fixture
def sample_label_to_model():
    """Sample label-to-model mapping."""
    return {
        "Response A": "Model-A",
        "Response B": "Model-B",
        "Response C": "Model-C",
    }
