"""LLM Council — Multi-model deliberation with anonymized peer review.

Top-level package exposing ``run_council``, ``CouncilConfig``, and
``CouncilContext`` for programmatic usage of the 3-stage pipeline.
"""

from llm_council.context import CouncilContext
from llm_council.council import run_council
from llm_council.models import CouncilConfig

__version__ = "0.1.0"
__all__ = ["run_council", "CouncilConfig", "CouncilContext"]
