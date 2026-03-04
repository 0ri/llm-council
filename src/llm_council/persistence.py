"""JSONL session persistence for council run records.

Exports ``RunLogger`` which writes structured JSONL records (config,
per-stage responses/rankings/synthesis, aggregation, summary) to a per-run
file under ``--log-dir``. Sensitive data is automatically redacted via
``security.redact_sensitive`` before writing.

Writes are buffered in memory and flushed via ``flush()`` (called
automatically at the end of each ``log_stage*`` method) to avoid
blocking the event loop on every individual record.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .security import redact_sensitive


class RunLogger:
    """Writes structured JSONL records for a council run.

    Records are buffered in memory and written to disk on ``flush()``
    (called automatically after each stage logging method).
    """

    def __init__(self, log_dir: str | Path, run_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.filepath = self.log_dir / f"{run_id}.jsonl"
        self._buffer: list[str] = []

    def _append_record(self, record: dict[str, Any]) -> None:
        """Serialize a record as redacted JSON and buffer it."""
        record["run_id"] = self.run_id
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        raw = json.dumps(record, default=str)
        self._buffer.append(redact_sensitive(raw))

    def flush(self) -> None:
        """Write all buffered records to disk and clear the buffer."""
        if not self._buffer:
            return
        with open(self.filepath, "a") as f:
            for line in self._buffer:
                f.write(line + "\n")
        self._buffer.clear()

    def log_config(self, question: str, config: dict[str, Any]) -> None:
        """Write the council configuration and question as a log record."""
        self._append_record({"type": "config", "question": question, "config": config})
        self.flush()

    def log_stage1(self, results: list, token_usages: dict) -> None:
        """Write Stage 1 response records with per-model token usage."""
        for result in results:
            self._append_record(
                {
                    "type": "stage1_response",
                    "model": result.model,
                    "response": result.response,
                    "token_usage": token_usages.get(result.model),
                }
            )
        self.flush()

    def log_stage2(self, results: list, label_mappings: dict, token_usages: dict) -> None:
        """Write Stage 2 ranking records with label mappings and validity flags."""
        for result in results:
            self._append_record(
                {
                    "type": "stage2_ranking",
                    "model": result.model,
                    "ranking_text": result.ranking,
                    "parsed_ranking": result.parsed_ranking,
                    "is_valid_ballot": result.is_valid_ballot,
                    "label_mapping": label_mappings.get(result.model, {}),
                    "token_usage": token_usages.get(result.model),
                }
            )
        self.flush()

    def log_stage3(self, result: Any, token_usage: dict | None) -> None:
        """Write the Stage 3 chairman synthesis record."""
        self._append_record(
            {
                "type": "stage3_synthesis",
                "model": result.model,
                "response": result.response,
                "token_usage": token_usage,
            }
        )
        self.flush()

    def log_aggregation(self, rankings: list, valid_ballots: int, total_ballots: int) -> None:
        """Write the aggregate ranking results and ballot counts."""
        self._append_record(
            {
                "type": "aggregation",
                "rankings": [
                    {
                        "model": r.model,
                        "average_rank": r.average_rank,
                        "borda_score": r.borda_score,
                        "rankings_count": r.rankings_count,
                    }
                    for r in rankings
                ],
                "valid_ballots": valid_ballots,
                "total_ballots": total_ballots,
            }
        )
        self.flush()

    def log_summary(self, cost_summary: str, elapsed: float) -> None:
        """Write the final cost summary and elapsed time record."""
        self._append_record(
            {
                "type": "summary",
                "cost_summary": cost_summary,
                "elapsed_seconds": elapsed,
            }
        )
        self.flush()
