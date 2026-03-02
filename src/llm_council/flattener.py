"""Codebase flattener for serializing a project directory into markdown.

Exports ``flatten_directory`` which walks a directory, respects ``.gitignore``,
skips binaries, and produces a single markdown document for LLM context.
Supports ``codemap`` mode for structural skeletons (AST for Python, heuristic
for others). Also provides the ``flatten-project`` CLI via ``main()``.
"""

from __future__ import annotations

import ast
import mimetypes
import os
import re
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
        "logs",
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
        "llm-council-flat*.md",
        "council-*-report.md",
        "council-*-output.md",
        "council-self-critique*.md",
    }
)


@dataclass
class FlattenedProject:
    """Result of flattening a project directory."""

    file_count: int
    total_chars: int
    estimated_tokens: int
    markdown: str


# Extensions that mimetypes may misidentify but are definitely text/code
_TEXT_EXTENSIONS = frozenset(
    {
        ".ts",
        ".tsx",
        ".jsx",
        ".mts",
        ".cts",
        ".vue",
        ".svelte",
        ".astro",
    }
)


def _is_binary(path: Path) -> bool:
    """Check if a file is likely binary."""
    suffix = path.suffix.lower()
    if suffix in BINARY_EXTENSIONS:
        return True
    # Known text/code extensions that mimetypes misidentifies (e.g. .ts → video/mp2t)
    if suffix in _TEXT_EXTENSIONS:
        return False
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
    import fnmatch

    name = path.name
    for pattern in SKIP_FILES:
        if fnmatch.fnmatch(name, pattern):
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


def _extract_python_skeleton(content: str) -> str | None:
    """Extract structural skeleton from Python source using AST.

    Returns function/class signatures, imports, constants, and docstrings
    without implementation bodies.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    lines: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.append(ast.get_source_segment(content, node) or ast.unparse(node))

        elif isinstance(node, ast.Assign):
            # Module-level constants/assignments
            segment = ast.get_source_segment(content, node)
            if segment and len(segment) < 200:
                lines.append(segment)

        elif isinstance(node, ast.AnnAssign):
            segment = ast.get_source_segment(content, node)
            if segment and len(segment) < 200:
                lines.append(segment)

        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            lines.append(_format_funcdef(node, content))

        elif isinstance(node, ast.ClassDef):
            lines.append(_format_classdef(node, content))

    return "\n\n".join(lines) if lines else None


def _format_funcdef(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str, indent: str = "") -> str:
    """Format a function definition as signature + first-line docstring."""
    parts: list[str] = []

    # Decorators
    for dec in node.decorator_list:
        dec_text = ast.get_source_segment(source, dec) or ast.unparse(dec)
        parts.append(f"{indent}@{dec_text}")

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.get_source_segment(source, node.args) or ast.unparse(node.args)
    ret = ""
    if node.returns:
        ret_text = ast.get_source_segment(source, node.returns) or ast.unparse(node.returns)
        ret = f" -> {ret_text}"

    parts.append(f"{indent}{prefix} {node.name}({args}){ret}:")

    # First line of docstring
    docstring = ast.get_docstring(node)
    if docstring:
        first_line = docstring.split("\n")[0].strip()
        parts.append(f'{indent}    """{first_line}"""')
    else:
        parts.append(f"{indent}    ...")

    return "\n".join(parts)


def _format_classdef(node: ast.ClassDef, source: str, indent: str = "") -> str:
    """Format a class definition with method signatures."""
    parts: list[str] = []

    # Decorators
    for dec in node.decorator_list:
        dec_text = ast.get_source_segment(source, dec) or ast.unparse(dec)
        parts.append(f"{indent}@{dec_text}")

    bases = ", ".join(ast.get_source_segment(source, b) or ast.unparse(b) for b in node.bases)
    bases_str = f"({bases})" if bases else ""
    parts.append(f"{indent}class {node.name}{bases_str}:")

    # Class docstring
    docstring = ast.get_docstring(node)
    if docstring:
        first_line = docstring.split("\n")[0].strip()
        parts.append(f'{indent}    """{first_line}"""')

    # Class-level assignments and method signatures
    child_indent = indent + "    "
    has_body = False
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Assign):
            segment = ast.get_source_segment(source, child)
            if segment and len(segment) < 200:
                parts.append(f"{child_indent}{segment.strip()}")
                has_body = True
        elif isinstance(child, ast.AnnAssign):
            segment = ast.get_source_segment(source, child)
            if segment and len(segment) < 200:
                parts.append(f"{child_indent}{segment.strip()}")
                has_body = True
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append("")
            parts.append(_format_funcdef(child, source, indent=child_indent))
            has_body = True

    if not has_body:
        parts.append(f"{child_indent}...")

    return "\n".join(parts)


# Patterns for non-Python structural lines
_STRUCTURE_PATTERNS = re.compile(
    r"^\s*(?:"
    r"(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:async\s+)?(?:function|class|interface|type|enum|struct|trait|impl|mod|pub\s+fn|fn|def|const|let|var|static)\s+"
    r"|(?:export\s+)?(?:const|let|var)\s+\w+\s*[=:]"
    r"|(?:import|from|require|use|package|module)\s+"
    r"|#\[|@\w+"
    r")",
    re.MULTILINE,
)


def _extract_generic_skeleton(content: str) -> str:
    """Extract structural lines from non-Python files using heuristics."""
    lines = content.splitlines()
    result: list[str] = []

    # Always include first 3 non-empty lines for context
    non_empty = [ln for ln in lines[:10] if ln.strip()][:3]
    result.extend(non_empty)

    for line in lines:
        if _STRUCTURE_PATTERNS.match(line) and line not in result:
            result.append(line)

    return "\n".join(result) if result else content[:500]


def flatten_directory(
    path: str | Path,
    respect_gitignore: bool = True,
    max_file_size: int = 100_000,
    codemap: bool = False,
) -> FlattenedProject:
    """Flatten a directory into a single markdown document.

    Args:
        path: Path to the directory to flatten.
        respect_gitignore: If True, respect .gitignore patterns.
        max_file_size: Skip files larger than this many bytes.
        codemap: If True, extract only structural skeletons (signatures,
            imports, class/function definitions) instead of full file contents.
            Uses AST parsing for Python; heuristic pattern matching for others.

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
            import pathspec

            gitignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

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

            if codemap:
                # Extract structural skeleton
                if ext == "py":
                    skeleton = _extract_python_skeleton(content)
                    if skeleton:
                        sections.append(f"## {rel_path}\n\n```{ext}\n{skeleton}\n```\n")
                        file_count += 1
                        total_chars += len(skeleton)
                    else:
                        # Fallback for unparseable Python
                        fallback = _extract_generic_skeleton(content)
                        sections.append(f"## {rel_path}\n\n```{ext}\n{fallback}\n```\n")
                        file_count += 1
                        total_chars += len(fallback)
                else:
                    skeleton = _extract_generic_skeleton(content)
                    sections.append(f"## {rel_path}\n\n```{ext}\n{skeleton}\n```\n")
                    file_count += 1
                    total_chars += len(skeleton)
            else:
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
    use_codemap = False
    max_size = 100_000
    paths: list[str] = []

    i = 0
    while i < len(args):
        if args[i] == "--no-gitignore":
            no_gitignore = True
            i += 1
        elif args[i] == "--codemap":
            use_codemap = True
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
            _usage = "Usage: flatten-project [--no-gitignore] [--codemap] [--max-file-size BYTES] PATH ..."
            print(_usage, file=sys.stderr)
            sys.exit(1)
        else:
            paths.append(args[i])
            i += 1

    if not paths:
        _usage = "Usage: flatten-project [--no-gitignore] [--codemap] [--max-file-size BYTES] PATH ..."
        print(_usage, file=sys.stderr)
        sys.exit(1)

    for path in paths:
        try:
            result = flatten_directory(
                path, respect_gitignore=not no_gitignore, max_file_size=max_size, codemap=use_codemap
            )
        except (FileNotFoundError, NotADirectoryError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        print(result.markdown)
        print(
            f"# {result.file_count} files, {result.total_chars:,} chars, ~{result.estimated_tokens:,} tokens",
            file=sys.stderr,
        )
