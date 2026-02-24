"""Shared test fixtures for LLM Council tests."""

import sys
from pathlib import Path

import pytest

# Try importing from installed package first, fall back to scripts directory
try:
    import llm_council  # noqa: F401 -- verify package is importable
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "skills" / "council" / "scripts"))


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
