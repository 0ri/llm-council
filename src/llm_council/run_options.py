"""RunOptions dataclass for consolidating run_council parameters."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunOptions:
    """Options controlling a single ``run_council`` invocation.

    Groups the 8 optional parameters of ``run_council`` into a single
    object so callers can construct, inspect, and pass them around
    without juggling many keyword arguments.

    All fields default to the same values previously hard-coded in
    ``run_council``'s signature.
    """

    print_manifest: bool = False
    log_dir: str | None = None
    context_factory: Any = None
    max_stage: int = 3
    seed: int | None = None
    use_cache: bool = True
    cache_ttl: int = 86400
    stream: bool = False
    on_chunk: Callable[[str], Awaitable[None]] | None = field(default=None, repr=False)
