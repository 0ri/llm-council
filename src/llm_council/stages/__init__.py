"""Stage execution logic for the 3-stage council deliberation pipeline.

Re-exports the public API from sub-modules so that existing imports
like ``from llm_council.stages import query_model`` continue to work.
"""

from .execution import (  # noqa: F401
    _circuit_breaker_key,
    query_model,
    query_models_parallel,
    stream_model,
)
from .stage1 import stage1_collect_responses  # noqa: F401
from .stage2 import _get_ranking, build_ranking_prompt, stage2_collect_rankings  # noqa: F401
from .stage3 import build_synthesis_prompt, stage3_synthesize_final  # noqa: F401

__all__ = [
    "_circuit_breaker_key",
    "build_ranking_prompt",
    "build_synthesis_prompt",
    "query_model",
    "query_models_parallel",
    "stage1_collect_responses",
    "stage2_collect_rankings",
    "stage3_synthesize_final",
    "stream_model",
]
