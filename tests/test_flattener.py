"""Tests for the project flattener."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from llm_council.flattener import FlattenedProject, flatten_directory


class TestFlattenBasicDirectory:
    """Test basic flattening of a directory."""

    def test_flatten_python_files(self):
        """Test flattening a directory with Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some Python files
            (Path(tmpdir) / "main.py").write_text("print('hello')")
            (Path(tmpdir) / "utils.py").write_text("def helper(): pass")

            result = flatten_directory(tmpdir)

            assert isinstance(result, FlattenedProject)
            assert result.file_count == 2
            assert result.total_chars > 0
            assert result.estimated_tokens > 0
            assert "print('hello')" in result.markdown
            assert "def helper(): pass" in result.markdown
            assert "## main.py" in result.markdown
            assert "## utils.py" in result.markdown

    def test_flatten_nested_directories(self):
        """Test flattening with nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "app.py").write_text("app = Flask(__name__)")

            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_app.py").write_text("def test_app(): pass")

            result = flatten_directory(tmpdir)

            assert result.file_count == 2
            assert "src/app.py" in result.markdown
            assert "tests/test_app.py" in result.markdown

    def test_markdown_code_blocks(self):
        """Test that files are wrapped in proper code blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "script.py").write_text("x = 1")

            result = flatten_directory(tmpdir)

            assert "```py" in result.markdown
            assert "```" in result.markdown


class TestRespectGitignore:
    """Test .gitignore pattern matching."""

    def test_respects_gitignore(self):
        """Test that .gitignore patterns exclude files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".gitignore").write_text("*.log\nsecrets/\n")
            (Path(tmpdir) / "app.py").write_text("code")
            (Path(tmpdir) / "debug.log").write_text("log data")

            secrets_dir = Path(tmpdir) / "secrets"
            secrets_dir.mkdir()
            (secrets_dir / "key.txt").write_text("secret")

            result = flatten_directory(tmpdir)

            # .gitignore itself is included (2 files), but matched files are not
            assert result.file_count == 2
            assert "app.py" in result.markdown
            assert "debug.log" not in result.markdown
            assert "key.txt" not in result.markdown

    def test_gitignore_disabled(self):
        """Test that gitignore can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".gitignore").write_text("*.log\n")
            (Path(tmpdir) / "app.py").write_text("code")
            (Path(tmpdir) / "debug.log").write_text("log data")

            result = flatten_directory(tmpdir, respect_gitignore=False)

            # .gitignore itself should be included, debug.log should be included
            assert result.file_count >= 2
            assert "debug.log" in result.markdown


class TestSkipBinaryFiles:
    """Test binary file detection."""

    def test_skips_binary_by_extension(self):
        """Test that binary files are skipped based on extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text("code")
            (Path(tmpdir) / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (Path(tmpdir) / "archive.zip").write_bytes(b"PK\x03\x04")

            result = flatten_directory(tmpdir)

            assert result.file_count == 1
            assert "app.py" in result.markdown
            assert "image.png" not in result.markdown
            assert "archive.zip" not in result.markdown


class TestMaxFileSize:
    """Test max_file_size limit."""

    def test_large_file_skipped(self):
        """Test that files exceeding max_file_size are skipped with a note."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "small.py").write_text("x = 1")
            (Path(tmpdir) / "large.py").write_text("x" * 200)

            result = flatten_directory(tmpdir, max_file_size=100)

            assert result.file_count == 1
            assert "small.py" in result.markdown
            # Large file should have a skip note
            assert "Skipped: file too large" in result.markdown


class TestTokenEstimation:
    """Test token estimation integration."""

    def test_token_estimation(self):
        """Test that tokens are estimated for the flattened content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text("print('hello world') " * 100)

            result = flatten_directory(tmpdir)

            assert result.estimated_tokens > 0
            # Token count should be reasonable (not zero, not insanely high)
            assert result.estimated_tokens < result.total_chars


class TestEmptyDirectory:
    """Test handling of empty directories."""

    def test_empty_directory(self):
        """Test flattening an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = flatten_directory(tmpdir)

            assert result.file_count == 0
            assert result.total_chars == 0
            assert result.estimated_tokens == 0
            assert result.markdown == ""


class TestErrorHandling:
    """Test error cases."""

    def test_nonexistent_path(self):
        """Test that nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            flatten_directory("/nonexistent/path/that/does/not/exist")

    def test_file_instead_of_directory(self):
        """Test that a file path raises NotADirectoryError."""
        with tempfile.NamedTemporaryFile(suffix=".py") as f:
            with pytest.raises(NotADirectoryError):
                flatten_directory(f.name)


class TestSkipDirectories:
    """Test directory skipping."""

    def test_skips_pycache(self):
        """Test that __pycache__ directories are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text("code")
            cache_dir = Path(tmpdir) / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "app.cpython-310.pyc").write_bytes(b"\x00")

            result = flatten_directory(tmpdir)

            assert result.file_count == 1
            assert "__pycache__" not in result.markdown

    def test_skips_node_modules(self):
        """Test that node_modules is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "index.js").write_text("const x = 1;")
            nm_dir = Path(tmpdir) / "node_modules"
            nm_dir.mkdir()
            pkg_dir = nm_dir / "some-package"
            pkg_dir.mkdir()
            (pkg_dir / "index.js").write_text("module.exports = {};")

            result = flatten_directory(tmpdir)

            assert result.file_count == 1
            assert "node_modules" not in result.markdown

    def test_skips_git_directory(self):
        """Test that .git directory is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text("code")
            git_dir = Path(tmpdir) / ".git"
            git_dir.mkdir()
            (git_dir / "HEAD").write_text("ref: refs/heads/main")

            result = flatten_directory(tmpdir)

            assert result.file_count == 1
            assert ".git" not in result.markdown


class TestSkipFiles:
    """Test file skipping patterns."""

    def test_skips_lock_files(self):
        """Test that lock files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text("code")
            (Path(tmpdir) / "package-lock.json").write_text("{}")
            (Path(tmpdir) / "uv.lock").write_text("lock")

            result = flatten_directory(tmpdir)

            assert result.file_count == 1
            assert "package-lock.json" not in result.markdown
            assert "uv.lock" not in result.markdown

    def test_skips_pyc_files(self):
        """Test that .pyc files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text("code")
            (Path(tmpdir) / "app.pyc").write_bytes(b"\x00")

            result = flatten_directory(tmpdir)

            assert result.file_count == 1

    def test_skips_empty_files(self):
        """Test that empty files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text("code")
            (Path(tmpdir) / "empty.py").write_text("")

            result = flatten_directory(tmpdir)

            assert result.file_count == 1
