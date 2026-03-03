# Contributing to LLM Council

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/0ri/llm-council.git
cd llm-council

# Install development dependencies with uv (recommended)
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_parsing.py -v

# Run with coverage
uv run pytest tests/ --cov=llm_council --cov-report=term-missing
```

### Property-Based Tests with Hypothesis

The project uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. These tests generate many random inputs to verify that properties hold across the entire input space, catching edge cases that example-based tests might miss.

```bash
# Run only property-based tests
uv run pytest tests/ -k "property" -v

# Run with Hypothesis statistics (shows shrinking, examples tried, etc.)
uv run pytest tests/ --hypothesis-show-statistics

# Run with a specific Hypothesis profile (if configured in conftest.py)
uv run pytest tests/ --hypothesis-seed=12345

# Increase the number of examples for thorough testing
HYPOTHESIS_MAX_EXAMPLES=500 uv run pytest tests/ -k "property" -v
```

When writing property-based tests, use `@given()` decorators with appropriate strategies and tag each test with a comment linking it to the requirement it validates:

```python
from hypothesis import given, strategies as st

# Feature: my-feature, Property 1: Description of the property
# Validates: Requirements X.Y
@given(st.text())
def test_property_some_invariant(input_text):
    result = my_function(input_text)
    assert some_invariant(result)
```

## Code Quality

### Linting and Formatting

```bash
# Check code style
uv run ruff check .

# Check formatting
uv run ruff format --check .

# Auto-fix issues
uv run ruff check . --fix
uv run ruff format .
```

### Code Style
- Line length: 120 characters maximum
- Formatting: Managed by ruff (see `pyproject.toml`)
- Type hints: Encouraged for public APIs
- Docstrings: Required for public functions/classes (Google-style)

## Pull Request Process

1. **Fork and clone** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** and add tests
4. **Run tests and linting**: Ensure all checks pass
5. **Commit with clear messages**: Follow conventional commits if possible
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Submit a pull request** with a clear description

## Architecture Overview

### Core Modules

The source lives in `src/llm_council/`. Here is the complete module listing:

- **`__init__.py`** — Package root and public API exports
- **`aggregation.py`** — Ranking aggregation algorithms (Borda count, etc.)
- **`budget.py`** — Budget guard: token/cost limits with reserve/commit/release
- **`cache.py`** — SQLite-backed response cache with TTL support
- **`cli.py`** — `llm-council` CLI entry point (argparse definitions)
- **`context.py`** — `CouncilContext` runtime container (lazy provider imports, API keys, config)
- **`cost.py`** — Per-stage cost tracking and token accounting
- **`council.py`** — Main orchestration: `run_council()` and `validate_config()`
- **`flattener.py`** — Codebase flattener (`flatten-project` CLI, `--flatten`, `--codemap`)
- **`formatting.py`** — Output formatting for council results
- **`manifest.py`** — Run manifest generation (run_id, timestamps, config hash)
- **`models.py`** — Pydantic data models (`CouncilConfig`, provider configs, result types)
- **`parsing.py`** — Ranking response parsing and format utilities
- **`persistence.py`** — Buffered JSONL run logger with per-stage flush
- **`progress.py`** — Progress visualization during council runs
- **`prompts.py`** — Prompt templates for all three pipeline stages
- **`run_options.py`** — `RunOptions` dataclass for `run_council()` parameters
- **`security.py`** — Input sanitization, prompt injection detection, nonce XML fencing
- **`stages/`** — 3-stage pipeline package:
  - `__init__.py` — Re-exports for backward compatibility
  - `execution.py` — `query_model`, `stream_model`, parallel dispatch, budget guards
  - `stage1.py` — Stage 1: collect individual model responses (with caching)
  - `stage2.py` — Stage 2: anonymized peer ranking with retry
  - `stage3.py` — Stage 3: chairman synthesis (query or streaming)

### Provider Modules

Provider implementations live in `src/llm_council/providers/`:

- **`__init__.py`** — `Provider` and `StreamingProvider` protocol definitions, `CircuitBreaker`, `StreamResult`, `fallback_astream` helper, and configurable timeout/retry defaults
- **`bedrock.py`** — AWS Bedrock provider (Converse API)
- **`openrouter.py`** — OpenRouter provider (multi-model gateway)
- **`poe.py`** — Poe.com provider (fastapi_poe client)

### Adding New Providers

New providers must satisfy the `Provider` protocol defined in
`src/llm_council/providers/__init__.py`. Optionally, they can also implement
`StreamingProvider` to support token-by-token streaming (used by the `--stream`
CLI flag for Stage 3 synthesis).

1. Create a new file in `src/llm_council/providers/` (e.g., `my_provider.py`)
2. Implement the `Provider` protocol from `providers/__init__.py`
3. Optionally implement `StreamingProvider` by adding an `astream()` method (see below)
4. Register the provider in the `_PROVIDER_MODULES` mapping in `context.py`
5. Update configuration documentation

Example provider structure:

```python
from llm_council.providers import Provider, StreamResult

class MyProvider:
    """Custom LLM provider implementing the Provider protocol."""

    async def query(
        self, prompt: str, model_config: dict, timeout: int
    ) -> tuple[str, dict | None]:
        """Send a prompt and return (response_text, usage_metadata)."""
        response = await self._make_api_call(prompt, timeout=timeout)
        usage = {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens}
        return response.text, usage
```

### Adding Streaming Support

To support the `--stream` CLI flag, implement the `StreamingProvider` protocol
in addition to `Provider`. This enables Stage 3 synthesis output to be streamed
to stdout token-by-token instead of waiting for the full response.

The `StreamingProvider` protocol requires an `astream()` method that returns a
`StreamResult` — an async iterator of text chunks:

```python
import typing
from llm_council.providers import Provider, StreamResult, StreamingProvider

class MyStreamingProvider:
    """Custom provider with streaming support."""

    async def query(
        self, prompt: str, model_config: dict, timeout: int
    ) -> tuple[str, dict | None]:
        """Non-streaming query (required by Provider protocol)."""
        response = await self._make_api_call(prompt, timeout=timeout)
        return response.text, {"input_tokens": response.input_tokens}

    def astream(
        self, prompt: str, model_config: dict, timeout: int
    ) -> StreamResult:
        """Stream response chunks for real-time output.

        Returns a StreamResult wrapping an async iterator of text chunks.
        The StreamResult.accumulated attribute collects the full response,
        and StreamResult.usage is populated once the stream completes.
        """
        async def _generate() -> typing.AsyncIterator[str]:
            async for chunk in self._stream_api_call(prompt, timeout=timeout):
                yield chunk

        return StreamResult(_generate())
```

Providers that do not implement `StreamingProvider` are automatically wrapped
by `fallback_astream()`, which calls `query()` and yields the full response as
a single chunk. Users can invoke streaming with:

```bash
llm-council "What is the meaning of life?" --stream
```

### Adding New Ranking Algorithms

1. Add the algorithm to `src/llm_council/aggregation.py`
2. Update `RankingAlgorithm` enum
3. Add tests in `tests/test_aggregation.py`

Example:
```python
def weighted_borda_count(rankings: list[list[str]], weights: list[float]) -> list[tuple[str, float]]:
    """Borda count with custom weights per ranker."""
    # Implementation here
```

## Testing Guidelines

- Write tests for new features
- Maintain or improve code coverage
- Use pytest fixtures for common test data
- Mock external API calls in unit tests
- Integration tests should be clearly marked
- Use Hypothesis for property-based tests where input spaces are large or edge cases matter

## Documentation

- Update README.md for user-facing changes
- Update CLAUDE.md for architectural changes
- Add docstrings to new functions/classes (Google-style with Args, Returns, Raises)
- Include type hints where appropriate
- See `docs/examples/custom_provider.py` for a complete provider implementation example

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.
