"""Project directory flattener for feeding codebases into the council."""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

from .cost import estimate_tokens

# Extensions that are always skipped (binary / non-text)
BINARY_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".webp",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".mkv",
        ".flac",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".o",
        ".a",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".pyc",
        ".pyo",
        ".class",
        ".jar",
        ".sqlite",
        ".db",
        ".sqlite3",
        ".DS_Store",
    }
)

# Directories that are always skipped
SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
    }
)

# Files that are always skipped
SKIP_FILES = frozenset(
    {
        "*.pyc",
        "*.lock",
        "*.min.js",
        "*.min.css",
        "package-lock.json",
        "yarn.lock",
        "uv.lock",
        "settings.local.json",
        ".env",
        ".env.local",
        "credentials.json",
        "*.pem",
        "*.key",
    }
)


@dataclass
class FlattenedProject:
    """Result of flattening a project directory."""

    file_count: int
    total_chars: int
    estimated_tokens: int
    markdown: str


def _is_binary(path: Path) -> bool:
    """Check if a file is likely binary."""
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    mime, _ = mimetypes.guess_type(str(path))
    if (
        mime
        and not mime.startswith("text/")
        and mime
        not in (
            "application/json",
            "application/xml",
            "application/javascript",
            "application/x-yaml",
            "application/toml",
            "application/x-sh",
        )
    ):
        return True
    return False


def _should_skip_dir(name: str) -> bool:
    """Check if a directory should be skipped."""
    if name in SKIP_DIRS:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def _should_skip_file(path: Path) -> bool:
    """Check if a file should be skipped."""
    name = path.name
    for pattern in SKIP_FILES:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern:
            return True
    return False


def _load_gitignore(directory: Path) -> list[str]:
    """Load .gitignore patterns from a directory."""
    gitignore_path = directory / ".gitignore"
    if not gitignore_path.exists():
        return []
    patterns = []
    for line in gitignore_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def flatten_directory(
    path: str | Path,
    respect_gitignore: bool = True,
    max_file_size: int = 100_000,
) -> FlattenedProject:
    """Flatten a directory into a single markdown document.

    Args:
        path: Path to the directory to flatten.
        respect_gitignore: If True, respect .gitignore patterns.
        max_file_size: Skip files larger than this many bytes.

    Returns:
        FlattenedProject with the flattened markdown and metadata.

    Raises:
        FileNotFoundError: If the path does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    root = Path(path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    # Load .gitignore if requested
    gitignore_spec = None
    if respect_gitignore:
        patterns = _load_gitignore(root)
        if patterns:
            try:
                import pathspec

                gitignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
            except ImportError:
                # pathspec not installed — skip .gitignore matching
                pass

    sections: list[str] = []
    file_count = 0
    total_chars = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out skipped directories (in-place to prevent os.walk descending)
        dirnames[:] = [d for d in sorted(dirnames) if not _should_skip_dir(d)]

        rel_dir = Path(dirpath).relative_to(root)

        for filename in sorted(filenames):
            filepath = Path(dirpath) / filename
            rel_path = rel_dir / filename

            # Skip checks
            if _should_skip_file(filepath):
                continue
            if _is_binary(filepath):
                continue

            # gitignore
            if gitignore_spec and gitignore_spec.match_file(str(rel_path)):
                continue

            # Size check
            try:
                size = filepath.stat().st_size
            except OSError:
                continue
            if size > max_file_size:
                sections.append(f"## {rel_path}\n\n*Skipped: file too large ({size:,} bytes)*\n")
                continue
            if size == 0:
                continue

            # Read file
            try:
                content = filepath.read_text(errors="replace")
            except OSError:
                continue

            ext = filepath.suffix.lstrip(".")
            sections.append(f"## {rel_path}\n\n```{ext}\n{content}\n```\n")
            file_count += 1
            total_chars += len(content)

    markdown = "\n".join(sections)
    tokens = estimate_tokens(markdown) if markdown else 0

    return FlattenedProject(
        file_count=file_count,
        total_chars=total_chars,
        estimated_tokens=tokens,
        markdown=markdown,
    )


def main() -> None:
    """CLI entry point for flatten-project."""
    import sys

    args = sys.argv[1:]

    # Parse flags
    no_gitignore = False
    max_size = 100_000
    paths: list[str] = []

    i = 0
    while i < len(args):
        if args[i] == "--no-gitignore":
            no_gitignore = True
            i += 1
        elif args[i] == "--max-file-size" and i + 1 < len(args):
            try:
                max_size = int(args[i + 1])
            except ValueError:
                print(f"Error: --max-file-size must be an integer, got '{args[i + 1]}'", file=sys.stderr)
                sys.exit(1)
            i += 2
        elif args[i].startswith("-"):
            print(f"Unknown flag: {args[i]}", file=sys.stderr)
            print("Usage: flatten-project [--no-gitignore] [--max-file-size BYTES] PATH [PATH ...]", file=sys.stderr)
            sys.exit(1)
        else:
            paths.append(args[i])
            i += 1

    if not paths:
        print("Usage: flatten-project [--no-gitignore] [--max-file-size BYTES] PATH [PATH ...]", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        try:
            result = flatten_directory(path, respect_gitignore=not no_gitignore, max_file_size=max_size)
        except (FileNotFoundError, NotADirectoryError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        print(result.markdown)
        print(
            f"# {result.file_count} files, {result.total_chars:,} chars, ~{result.estimated_tokens:,} tokens",
            file=sys.stderr,
        )
