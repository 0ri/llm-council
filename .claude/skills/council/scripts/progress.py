"""
Real-time progress visualization for LLM Council.

Provides dual-mode rendering:
- TTY mode: Rich live display with spinners and colors (when stderr is a terminal)
- Non-TTY mode: Simple line-by-line log messages (when piped/subagent)
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from enum import Enum

try:
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class ModelStatus(Enum):
    """Status of a model in the council process."""
    PENDING = "pending"
    QUERYING = "querying"
    DONE = "done"
    FAILED = "failed"


@dataclass
class StageProgress:
    """Progress tracking for a single stage."""
    stage_num: int
    stage_total: int = 3
    description: str = ""
    models: dict[str, ModelStatus] = field(default_factory=dict)
    model_start_times: dict[str, float] = field(default_factory=dict)
    model_elapsed: dict[str, float] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    completed: bool = False
    summary: str | None = None


class ProgressManager:
    """Manages real-time progress display for council execution.

    In TTY mode (interactive terminal), displays a rich live-updating view
    with spinners showing which models are still being queried.

    In non-TTY mode (piped output, subagent), emits simple log lines
    to stderr as each model completes.
    """

    def __init__(self, is_tty: bool | None = None):
        if is_tty is None:
            is_tty = sys.stderr.isatty() and HAS_RICH

        self.is_tty = is_tty
        self.current_stage: StageProgress | None = None
        self.total_start_time = time.time()
        self._render_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._spinner_idx = 0
        self._spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"]

        if self.is_tty:
            self._console = Console(stderr=True)
            self._live: Live | None = None
        else:
            self._console = None
            self._live = None

    async def start_stage(self, stage: int, description: str, models: list[str]):
        """Begin a new stage, initializing model tracking."""
        async with self._lock:
            self.current_stage = StageProgress(
                stage_num=stage,
                description=description,
                models={m: ModelStatus.PENDING for m in models},
            )

        if self.is_tty:
            if self._live is None:
                self._live = Live(
                    self._render_tty(),
                    console=self._console,
                    refresh_per_second=10,
                    transient=False,
                )
                self._live.start()
            if self._render_task is None or self._render_task.done():
                self._render_task = asyncio.create_task(self._render_loop())
        else:
            n = len(models)
            self._log(f"Stage {stage}/3: {description} ({n} model{'s' if n != 1 else ''})...")

    async def update_model(
        self, model: str, status: ModelStatus, elapsed: float | None = None
    ):
        """Update a model's status."""
        async with self._lock:
            if self.current_stage is None:
                return
            self.current_stage.models[model] = status
            if status == ModelStatus.QUERYING:
                self.current_stage.model_start_times[model] = time.time()
            if elapsed is not None:
                self.current_stage.model_elapsed[model] = elapsed

        if not self.is_tty and self.current_stage:
            sn = self.current_stage.stage_num
            if status == ModelStatus.DONE:
                t = f" ({elapsed:.1f}s)" if elapsed is not None else ""
                self._log(f"Stage {sn}/3: ✓ {model} done{t}")
            elif status == ModelStatus.FAILED:
                t = f" ({elapsed:.1f}s)" if elapsed is not None else ""
                self._log(f"Stage {sn}/3: ✗ {model} failed{t}")

    async def complete_stage(self, summary: str | None = None):
        """Mark the current stage as complete."""
        if self.current_stage is None:
            return
        async with self._lock:
            self.current_stage.completed = True
            self.current_stage.summary = summary
            elapsed = time.time() - self.current_stage.start_time
            sn = self.current_stage.stage_num

        if self.is_tty:
            # Force one final render
            if self._live:
                self._live.update(self._render_tty())
        else:
            s = f" — {summary}" if summary else ""
            self._log(f"Stage {sn}/3: Complete ({elapsed:.1f}s){s}")

    async def complete_council(self, total_elapsed: float):
        """Mark the entire council as complete and clean up."""
        if not self.is_tty:
            self._log(f"Council complete ({total_elapsed:.1f}s total)")
        await self._cleanup()

    async def _cleanup(self):
        """Stop render loop and live display."""
        if self._render_task and not self._render_task.done():
            self._render_task.cancel()
            try:
                await self._render_task
            except asyncio.CancelledError:
                pass
        if self._live:
            # Final render before stopping
            self._live.update(self._render_tty())
            self._live.stop()
            self._live = None

    async def _render_loop(self):
        """Async loop that updates the TTY display at ~10 Hz."""
        try:
            while True:
                await asyncio.sleep(0.1)
                self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
                if self._live:
                    async with self._lock:
                        self._live.update(self._render_tty())
        except asyncio.CancelledError:
            pass

    def _render_tty(self) -> Text:
        """Build the rich Text display for TTY mode."""
        if self.current_stage is None:
            return Text("Initializing...")

        stage = self.current_stage
        output = Text()

        # Header
        output.append("━━━ LLM Council ", style="bold cyan")
        output.append("━" * 29, style="bold cyan")
        output.append("\n\n")

        # Stage line
        stage_elapsed = time.time() - stage.start_time
        output.append(f"Stage {stage.stage_num}/3: {stage.description}", style="bold")
        if stage.completed:
            output.append(f"  done ({stage_elapsed:.1f}s)", style="green")
        output.append("\n\n")

        # Model lines
        spinner = self._spinner_chars[self._spinner_idx]
        for model, status in stage.models.items():
            output.append("  ")
            if status == ModelStatus.DONE:
                elapsed = stage.model_elapsed.get(model, 0)
                output.append("✓", style="green")
                output.append(f" {model:<22} done ({elapsed:.1f}s)\n", style="green")
            elif status == ModelStatus.FAILED:
                elapsed = stage.model_elapsed.get(model, 0)
                output.append("✗", style="red")
                output.append(f" {model:<22} failed ({elapsed:.1f}s)\n", style="red")
            elif status == ModelStatus.QUERYING:
                start = stage.model_start_times.get(model, stage.start_time)
                running = time.time() - start
                output.append(spinner, style="cyan")
                output.append(f" {model:<22} querying... ({running:.1f}s)\n", style="cyan")
            else:  # PENDING
                output.append("○", style="dim")
                output.append(f" {model:<22} pending\n", style="dim")

        # Stage summary
        if stage.completed and stage.summary:
            output.append("\n")
            output.append("━" * 45, style="dim")
            output.append("\n")
            output.append(
                f"  {stage.summary}\n",
                style="green",
            )

        return output

    def _log(self, message: str):
        """Emit a simple log line to stderr (non-TTY mode)."""
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {message}", file=sys.stderr, flush=True)
