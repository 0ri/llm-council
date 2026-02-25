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
- Docstrings: Required for public functions/classes

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

- **`council.py`** - Main orchestration logic
- **`stages.py`** - 3-stage deliberation implementation
- **`providers/`** - LLM provider implementations
  - `base.py` - Provider protocol definition
  - `bedrock.py` - AWS Bedrock provider
  - `poe.py` - Poe.com provider
- **`aggregation.py`** - Ranking aggregation algorithms
- **`parsing.py`** - Response parsing utilities
- **`security.py`** - Injection hardening logic
- **`progress.py`** - Progress visualization
- **`cost.py`** - Cost tracking utilities

### Adding New Providers

1. Create a new file in `src/llm_council/providers/`
2. Implement the `Provider` protocol from `providers/base.py`
3. Add provider initialization in `stages.py`
4. Update configuration documentation

Example provider structure:
```python
from .base import Provider

class NewProvider(Provider):
    async def query(self, prompt: str, **kwargs) -> tuple[str, dict]:
        # Implement provider-specific logic
        response = await self._make_api_call(prompt)
        return response.text, {"tokens": response.usage}
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

## Documentation

- Update README.md for user-facing changes
- Update CLAUDE.md for architectural changes
- Add docstrings to new functions/classes
- Include type hints where appropriate

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.