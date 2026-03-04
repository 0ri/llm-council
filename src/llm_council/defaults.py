"""Shared default constants for LLM Council configuration.

Centralises magic numbers that were previously scattered across
``models.py``, ``cache.py``, ``run_options.py``, and ``council.py``.

Note: ``providers/__init__.py`` defines ``SOFT_TIMEOUT = 480`` as the
absolute provider-level fallback. ``DEFAULT_SOFT_TIMEOUT`` here (300 s)
is the *config-level* default used when the user omits ``soft_timeout``
from their council config.
"""

DEFAULT_CACHE_TTL: int = 86400  # 24 hours in seconds
DEFAULT_STAGE2_RETRIES: int = 1
DEFAULT_SOFT_TIMEOUT: float = 300.0  # config-level default (seconds)
