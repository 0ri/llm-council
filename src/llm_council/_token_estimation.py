"""Shared token estimation utility.

Provides ``estimate_tokens`` using tiktoken (cl100k_base) when available,
falling back to a 4-chars-per-token heuristic. Used by both ``cost.py``
and ``flattener.py`` to avoid coupling the flattener to the cost module.
"""

from __future__ import annotations

_ENCODER = None
_LOADED = False


def estimate_tokens(text: str) -> int:
    """Estimate token count for *text*.

    Uses tiktoken cl100k_base encoding when available.
    Falls back to 4 chars/token heuristic.
    """
    global _ENCODER, _LOADED
    if not _LOADED:
        _LOADED = True
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _ENCODER = None
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return len(text) // 4
