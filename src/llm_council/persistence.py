"""JSONL persistence for council run records."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AggregateRanking, Stage1Result, Stage2Result, Stage3Result


class RunLogger:
    """Writes structured JSONL records for a council run."""

    def __init__(self, log_dir: str | Path, run_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.filepath = self.log_dir / f"{run_id}.jsonl"

    def _write_record(self, record: dict[str, Any]) -> None:
        record["run_id"] = self.run_id
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.filepath, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_config(self, question: str, config: dict[str, Any]) -> None:
        self._write_record({"type": "config", "question": question, "config": config})

    def log_stage1(self, results: list, token_usages: dict) -> None:
        for result in results:
            model = result.model if hasattr(result, "model") else result.get("model")
            response = result.response if hasattr(result, "response") else result.get("response")
            self._write_record({
                "type": "stage1_response",
                "model": model,
                "response": response,
                "token_usage": token_usages.get(model),
            })

    def log_stage2(self, results: list, label_mappings: dict, token_usages: dict) -> None:
        for result in results:
            model = result.model if hasattr(result, "model") else result.get("model")
            self._write_record({
                "type": "stage2_ranking",
                "model": model,
                "ranking_text": result.ranking if hasattr(result, "ranking") else result.get("ranking"),
                "parsed_ranking": (
                    result.parsed_ranking if hasattr(result, "parsed_ranking") else result.get("parsed_ranking")
                ),
                "is_valid_ballot": (
                    result.is_valid_ballot if hasattr(result, "is_valid_ballot") else result.get("is_valid_ballot")
                ),
                "label_mapping": label_mappings.get(model, {}),
                "token_usage": token_usages.get(model),
            })

    def log_stage3(self, result: Any, token_usage: dict | None) -> None:
        model = result.model if hasattr(result, "model") else result.get("model")
        response = result.response if hasattr(result, "response") else result.get("response")
        self._write_record({
            "type": "stage3_synthesis",
            "model": model,
            "response": response,
            "token_usage": token_usage,
        })

    def log_aggregation(self, rankings: list, valid_ballots: int, total_ballots: int) -> None:
        self._write_record({
            "type": "aggregation",
            "rankings": [
                {
                    "model": r.model if hasattr(r, "model") else r.get("model"),
                    "average_rank": r.average_rank if hasattr(r, "average_rank") else r.get("average_rank"),
                    "borda_score": getattr(r, "borda_score", None),
                    "rankings_count": (
                        r.rankings_count if hasattr(r, "rankings_count") else r.get("rankings_count")
                    ),
                }
                for r in rankings
            ],
            "valid_ballots": valid_ballots,
            "total_ballots": total_ballots,
        })

    def log_summary(self, cost_summary: str, elapsed: float) -> None:
        self._write_record({
            "type": "summary",
            "cost_summary": cost_summary,
            "elapsed_seconds": elapsed,
        })
