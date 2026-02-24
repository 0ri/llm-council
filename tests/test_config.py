"""Tests for configuration validation and loading."""

import json
import os
import tempfile
from unittest.mock import patch

try:
    from llm_council.cli import load_config
    from llm_council.council import validate_config
except ImportError:
    from council import load_config, validate_config


class TestValidateConfig:
    def test_valid_config(self, sample_config):
        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            errors = validate_config(sample_config)
        assert errors == []

    def test_missing_council_models(self):
        config = {"chairman": {"name": "X", "provider": "poe", "bot_name": "X"}}
        errors = validate_config(config)
        assert any("council_models" in e for e in errors)

    def test_empty_council_models(self):
        config = {"council_models": [], "chairman": {"name": "X", "provider": "poe", "bot_name": "X"}}
        errors = validate_config(config)
        assert any("non-empty" in e.lower() for e in errors)

    def test_missing_chairman(self):
        config = {"council_models": [{"name": "X", "provider": "poe", "bot_name": "X"}]}
        errors = validate_config(config)
        assert any("chairman" in e for e in errors)

    def test_unknown_provider(self):
        config = {
            "council_models": [{"name": "X", "provider": "openai", "bot_name": "X"}],
            "chairman": {"name": "X", "provider": "poe", "bot_name": "X"},
        }
        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            errors = validate_config(config)
        assert any("unknown provider" in e.lower() or "openai" in e for e in errors)

    def test_bedrock_missing_model_id(self):
        config = {
            "council_models": [{"name": "Claude", "provider": "bedrock"}],
            "chairman": {"name": "Claude", "provider": "bedrock"},
        }
        errors = validate_config(config)
        assert any("model_id" in e for e in errors)

    def test_poe_missing_bot_name(self):
        config = {
            "council_models": [{"name": "GPT", "provider": "poe"}],
            "chairman": {"name": "GPT", "provider": "poe"},
        }
        errors = validate_config(config)
        assert any("bot_name" in e for e in errors)

    def test_missing_poe_api_key(self):
        config = {
            "council_models": [{"name": "GPT", "provider": "poe", "bot_name": "GPT-4"}],
            "chairman": {"name": "GPT", "provider": "poe", "bot_name": "GPT-4"},
        }
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("POE_API_KEY", None)
            errors = validate_config(config)
        assert any("POE_API_KEY" in e for e in errors)

    def test_invalid_budget_tokens_too_low(self):
        config = {
            "council_models": [{"name": "Claude", "provider": "bedrock", "model_id": "test", "budget_tokens": 100}],
            "chairman": {"name": "Claude", "provider": "bedrock", "model_id": "test"},
        }
        errors = validate_config(config)
        assert any("budget_tokens" in e for e in errors)

    def test_invalid_budget_tokens_too_high(self):
        config = {
            "council_models": [{"name": "Claude", "provider": "bedrock", "model_id": "test", "budget_tokens": 999999}],
            "chairman": {"name": "Claude", "provider": "bedrock", "model_id": "test"},
        }
        errors = validate_config(config)
        assert any("budget_tokens" in e for e in errors)

    def test_model_missing_name(self):
        config = {
            "council_models": [{"provider": "poe", "bot_name": "X"}],
            "chairman": {"name": "X", "provider": "poe", "bot_name": "X"},
        }
        with patch.dict(os.environ, {"POE_API_KEY": "test-key"}):
            errors = validate_config(config)
        assert any("name" in e.lower() for e in errors)


class TestLoadConfig:
    def test_load_valid_config(self, sample_config):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_config, f)
            f.flush()
            loaded = load_config(f.name)
        os.unlink(f.name)
        assert loaded["council_models"] == sample_config["council_models"]
        assert loaded["chairman"] == sample_config["chairman"]

    def test_load_missing_file_returns_defaults(self):
        config = load_config("/nonexistent/path/config.json")
        assert "council_models" in config
        assert "chairman" in config
        assert len(config["council_models"]) > 0
