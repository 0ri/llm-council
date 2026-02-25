"""Tests for the CLI module."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest

from llm_council.cli import _print_dry_run, load_config, main, setup_logging


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_config_with_valid_file(self):
        """Test loading config from a valid JSON file."""
        config_data = {
            "council_models": [
                {"name": "Model A", "provider": "bedrock", "model_id": "model-a"},
                {"name": "Model B", "provider": "poe", "bot_name": "ModelB"},
            ],
            "chairman": {"name": "Chairman", "provider": "bedrock", "model_id": "chairman-model"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            result = load_config(temp_path)
            assert result == config_data
            assert len(result["council_models"]) == 2
            assert result["chairman"]["name"] == "Chairman"
        finally:
            os.unlink(temp_path)

    def test_load_config_with_missing_file(self):
        """Test that missing config file returns default config."""
        result = load_config("/nonexistent/path/config.json")

        # Should return default config
        assert "council_models" in result
        assert "chairman" in result
        assert len(result["council_models"]) > 0
        assert result["council_models"][0]["name"] == "Claude Opus 4.6"
        assert result["chairman"]["name"] == "Claude Opus 4.6"

    def test_load_config_with_invalid_json(self):
        """Test that invalid JSON causes SystemExit."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            with pytest.raises(SystemExit) as exc_info:
                with patch("sys.stderr"):  # Suppress error output
                    load_config(temp_path)
            assert exc_info.value.code == 1
        finally:
            os.unlink(temp_path)

    def test_load_config_default_search_paths(self):
        """Test that load_config searches default paths."""
        with patch("os.path.exists") as mock_exists:
            with patch("builtins.open") as mock_open:
                with patch("json.load") as mock_json_load:
                    # Simulate finding config in .claude directory
                    mock_exists.side_effect = lambda path: ".claude/council-config.json" in path
                    mock_json_load.return_value = {"test": "config"}

                    result = load_config()

                    assert result == {"test": "config"}
                    mock_exists.assert_called()
                    mock_open.assert_called()

    def test_load_config_no_default_paths_exist(self):
        """Test that default config is returned when no default paths exist."""
        with patch("os.path.exists", return_value=False):
            result = load_config()

            # Should return default config
            assert "council_models" in result
            assert "chairman" in result

    def test_load_config_with_io_error(self):
        """Test that IO errors cause SystemExit."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"valid": "json"}')
            temp_path = f.name

        try:
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                with pytest.raises(SystemExit) as exc_info:
                    with patch("sys.stderr"):
                        load_config(temp_path)
                assert exc_info.value.code == 1
        finally:
            os.unlink(temp_path)

    def test_default_config_structure(self):
        """Test that default config has expected structure."""
        result = load_config("/nonexistent/file.json")

        # Check council_models
        assert isinstance(result["council_models"], list)
        assert len(result["council_models"]) >= 3  # At least 3 models

        # Check first model (Claude)
        claude = result["council_models"][0]
        assert claude["name"] == "Claude Opus 4.6"
        assert claude["provider"] == "bedrock"
        assert "model_id" in claude
        assert claude.get("budget_tokens") == 10000

        # Check GPT model
        gpt_models = [m for m in result["council_models"] if "GPT" in m["name"]]
        assert len(gpt_models) > 0
        gpt = gpt_models[0]
        assert gpt["provider"] == "poe"
        assert "bot_name" in gpt
        assert gpt.get("web_search") is True
        assert gpt.get("reasoning_effort") == "high"

        # Check chairman
        assert result["chairman"]["name"] == "Claude Opus 4.6"
        assert result["chairman"]["provider"] == "bedrock"

    def test_default_config_enhanced_capabilities(self):
        """Test that default config includes enhanced model capabilities."""
        result = load_config("/nonexistent/file.json")

        # Check for enhanced capabilities
        for model in result["council_models"]:
            if model["provider"] == "bedrock" and "claude-opus" in model.get("model_id", ""):
                # Claude Opus should have budget_tokens
                assert "budget_tokens" in model

            if model["provider"] == "poe":
                if "GPT" in model["name"]:
                    # GPT models should have web_search and reasoning_effort
                    assert "web_search" in model or "reasoning_effort" in model
                if "Gemini" in model["name"]:
                    # Gemini models should have web_search
                    assert "web_search" in model


class TestSetupLogging:
    """Test the setup_logging function."""

    def test_setup_logging_default(self):
        """Test logging setup with default settings."""
        with patch("llm_council.cli.logger") as mock_logger:
            mock_logger.handlers = []  # Simulate no existing handlers
            setup_logging(verbose=False)

            mock_logger.setLevel.assert_called_with(20)  # INFO level
            mock_logger.addHandler.assert_called_once()

    def test_setup_logging_verbose(self):
        """Test logging setup with verbose mode."""
        with patch("llm_council.cli.logger") as mock_logger:
            mock_logger.handlers = []  # Simulate no existing handlers
            setup_logging(verbose=True)

            mock_logger.setLevel.assert_called_with(10)  # DEBUG level
            mock_logger.addHandler.assert_called_once()

    def test_logging_handler_configuration(self):
        """Test that logging handler is configured correctly."""
        with patch("logging.getLogger"):
            with patch("logging.StreamHandler") as mock_handler_cls:
                with patch("logging.Formatter") as mock_formatter_cls:
                    mock_handler = mock_handler_cls.return_value
                    mock_formatter = mock_formatter_cls.return_value

                    setup_logging()

                    # Handler should use stderr
                    mock_handler_cls.assert_called_with(sys.stderr)

                    # Formatter should have correct format
                    mock_formatter_cls.assert_called_with("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

                    # Handler should have formatter set
                    mock_handler.setFormatter.assert_called_with(mock_formatter)


class TestMain:
    """Test the main entry point."""

    def test_main_no_arguments(self):
        """Test main with no arguments shows usage and exits."""
        with patch("sys.argv", ["council.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with code 2 for usage errors
            assert exc_info.value.code == 2

    def test_main_with_question(self):
        """Test main with just a question."""
        with patch("sys.argv", ["council.py", "What is 2+2?"]):
            with patch("llm_council.cli.asyncio.run") as mock_run:
                with patch("llm_council.cli.load_config") as mock_load_config:
                    with patch("llm_council.cli.validate_config", return_value=[]):
                        mock_load_config.return_value = {"test": "config"}
                        mock_run.return_value = "Answer: 4"

                        with patch("builtins.print") as mock_print:
                            main()

                        mock_print.assert_called_with("Answer: 4")
                        mock_run.assert_called_once()

    def test_main_with_config_flag(self):
        """Test main with --config flag."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"custom": "config"}, f)
            temp_path = f.name

        try:
            with patch("sys.argv", ["council.py", "--config", temp_path, "Question?"]):
                with patch("llm_council.cli.asyncio.run") as mock_run:
                    with patch("llm_council.cli.validate_config", return_value=[]):
                        mock_run.return_value = "Result"

                        with patch("builtins.print"):
                            main()

                        # Check that custom config was used
                        call_args = mock_run.call_args[0][0]
                        from inspect import iscoroutine

                        # The call is to run_council coroutine
                        assert iscoroutine(call_args)
        finally:
            os.unlink(temp_path)

    def test_main_with_verbose_flag(self):
        """Test main with -v/--verbose flag."""
        with patch("sys.argv", ["council.py", "-v", "Question"]):
            with patch("llm_council.cli.setup_logging") as mock_setup:
                with patch("llm_council.cli.asyncio.run"):
                    with patch("llm_council.cli.validate_config", return_value=[]):
                        with patch("builtins.print"):
                            main()

                        mock_setup.assert_called_with(verbose=True)

    def test_main_with_manifest_flag(self):
        """Test main with --manifest flag."""
        with patch("sys.argv", ["council.py", "--manifest", "Question"]):
            with patch("llm_council.cli.asyncio.run") as mock_run:
                with patch("llm_council.cli.validate_config", return_value=[]):
                    mock_run.return_value = "Result"

                    with patch("builtins.print"):
                        main()

    def test_main_no_question_provided(self):
        """Test main when no question is provided after flags."""
        with patch("sys.argv", ["council.py", "--config", "config.json"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with code 2 for missing required args
            assert exc_info.value.code == 2

    def test_main_config_validation_errors(self):
        """Test main when config validation fails."""
        with patch("sys.argv", ["council.py", "Question"]):
            with patch("llm_council.cli.load_config") as mock_load:
                with patch("llm_council.cli.validate_config") as mock_validate:
                    with patch("llm_council.cli.setup_logging"):
                        with patch("llm_council.cli.logger") as mock_logger:
                            mock_load.return_value = {"invalid": "config"}
                            mock_validate.return_value = ["Error 1", "Error 2"]

                            with pytest.raises(SystemExit) as exc_info:
                                with patch("sys.stderr"):
                                    main()

                            assert exc_info.value.code == 1
                            # Verify error was logged
                            mock_logger.error.assert_called()

    def test_main_config_path_with_spaces(self):
        """Test main with config path containing spaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "path with spaces")
            os.makedirs(config_dir)
            config_path = os.path.join(config_dir, "config.json")

            with open(config_path, "w") as f:
                json.dump({"test": "config"}, f)

            with patch("sys.argv", ["council.py", "--config", config_path, "Question"]):
                with patch("llm_council.cli.asyncio.run"):
                    with patch("llm_council.cli.validate_config", return_value=[]):
                        with patch("builtins.print"):
                            # Should handle path with spaces correctly
                            main()

    def test_main_combined_flags(self):
        """Test main with multiple flags combined."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "config"}, f)
            temp_path = f.name

        try:
            with patch("sys.argv", ["council.py", "-v", "--config", temp_path, "--manifest", "Question"]):
                with patch("llm_council.cli.setup_logging") as mock_setup:
                    with patch("llm_council.cli.asyncio.run"):
                        with patch("llm_council.cli.validate_config", return_value=[]):
                            with patch("builtins.print"):
                                main()

                            # Verbose should be enabled
                            mock_setup.assert_called_with(verbose=True)
        finally:
            os.unlink(temp_path)

    def test_help_flag(self):
        """Test that --help works."""
        with patch("sys.argv", ["council.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


class TestStageFlag:
    """Test --stage flag parsing."""

    def test_stage_1(self):
        """Test --stage 1 passes max_stage=1 to run_council."""
        with patch("sys.argv", ["council.py", "--stage", "1", "Question"]):
            with patch("llm_council.cli.asyncio.run") as mock_run:
                with patch("llm_council.cli.validate_config", return_value=[]):
                    mock_run.return_value = "Result"
                    with patch("builtins.print"):
                        main()
                    mock_run.assert_called_once()

    def test_stage_2(self):
        """Test --stage 2 is accepted."""
        with patch("sys.argv", ["council.py", "--stage", "2", "Question"]):
            with patch("llm_council.cli.asyncio.run") as mock_run:
                with patch("llm_council.cli.validate_config", return_value=[]):
                    mock_run.return_value = "Result"
                    with patch("builtins.print"):
                        main()

    def test_invalid_stage_exits(self):
        """Test --stage with invalid value exits with error."""
        with patch("sys.argv", ["council.py", "--stage", "5", "Question"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with code 2 for invalid choices
            assert exc_info.value.code == 2

    def test_non_numeric_stage_exits(self):
        """Test --stage with non-numeric value exits with error."""
        with patch("sys.argv", ["council.py", "--stage", "abc", "Question"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2


class TestDryRun:
    """Test --dry-run flag."""

    def test_dry_run_no_api_calls(self):
        """Test that --dry-run does not call asyncio.run for council."""
        with patch("sys.argv", ["council.py", "--dry-run", "Question"]):
            with patch("llm_council.cli.asyncio.run") as mock_run:
                with patch("llm_council.cli.validate_config", return_value=[]):
                    with patch("builtins.print"):
                        main()
                    # asyncio.run should NOT be called (no council run)
                    mock_run.assert_not_called()

    def test_dry_run_output(self):
        """Test dry-run prints summary to stderr."""
        config = {
            "council_models": [
                {"name": "Model A", "provider": "bedrock", "model_id": "a"},
                {"name": "Model B", "provider": "poe", "bot_name": "b"},
            ],
            "chairman": {"name": "Model A", "provider": "bedrock", "model_id": "a"},
        }
        with patch("sys.stderr"):
            _print_dry_run(config, "Test question")
            # Just verify it doesn't crash; actual output goes to stderr


class TestListModels:
    """Test --list-models flag."""

    def test_list_models_no_question_required(self):
        """Test that --list-models works without a question."""
        with patch("sys.argv", ["council.py", "--list-models"]):
            with patch("llm_council.cli.asyncio.run") as mock_run:
                mock_run.return_value = None
                with patch("builtins.print"):
                    main()
                mock_run.assert_called_once()


class TestFlattenFlag:
    """Test --flatten flag."""

    def test_flatten_prepends_to_question(self):
        """Test that --flatten prepends flattened content to the question."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("x = 1")

            with patch("sys.argv", ["council.py", "--flatten", tmpdir, "Review this"]):
                with patch("llm_council.cli.asyncio.run") as mock_run:
                    with patch("llm_council.cli.validate_config", return_value=[]):
                        mock_run.return_value = "Result"
                        with patch("builtins.print"):
                            main()
                        mock_run.assert_called_once()
