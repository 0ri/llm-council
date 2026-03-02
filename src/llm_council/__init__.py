"""LLM Council — Multi-model deliberation with anonymized peer review.

Top-level package exposing ``run_council``, ``CouncilConfig``, and
``CouncilContext`` for programmatic usage of the 3-stage pipeline.
"""

from importlib.metadata import PackageNotFoundError, version

from llm_council.context import CouncilContext
from llm_council.council import run_council
from llm_council.models import BudgetConfig, CouncilConfig, PromptConfig

try:
    __version__ = version("llm-council-skill")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["run_council", "CouncilConfig", "CouncilContext", "BudgetConfig", "PromptConfig"]
