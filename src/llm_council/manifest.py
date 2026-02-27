"""Run manifest for recording council execution metadata.

Exports ``RunManifest`` capturing run_id, timestamp, models, chairman,
stage counts, elapsed time, estimated tokens, and config hash. Appended
as an HTML comment block to full council output and optionally printed
as JSON to stderr via ``--manifest``.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class RunManifest:
    """Metadata about a single council run."""

    run_id: str  # UUID
    timestamp: str  # ISO 8601
    question: str  # (truncated to 200 chars)
    models: list[str]  # model names
    chairman: str
    stage1_results: int  # count of successful responses
    stage2_valid_ballots: int
    stage2_total_ballots: int
    total_elapsed_seconds: float
    estimated_tokens: int
    config_hash: str  # SHA256 of config JSON

    @classmethod
    def create(
        cls,
        question: str,
        config: dict[str, Any],
        stage1_count: int,
        valid_ballots: int,
        total_ballots: int,
        elapsed_seconds: float,
        estimated_tokens: int,
    ) -> RunManifest:
        """Create a new RunManifest from council execution data."""
        # Generate unique run ID
        run_id = str(uuid.uuid4())

        # Current timestamp in ISO 8601 format
        timestamp = datetime.now(timezone.utc).isoformat()

        # Truncate question to 200 chars
        truncated_question = question[:200] if len(question) > 200 else question

        # Extract model names
        models = [m.get("name", "unknown") for m in config.get("council_models", [])]
        chairman = config.get("chairman", {}).get("name", "unknown")

        # Generate config hash
        config_json = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()

        return cls(
            run_id=run_id,
            timestamp=timestamp,
            question=truncated_question,
            models=models,
            chairman=chairman,
            stage1_results=stage1_count,
            stage2_valid_ballots=valid_ballots,
            stage2_total_ballots=total_ballots,
            total_elapsed_seconds=elapsed_seconds,
            estimated_tokens=estimated_tokens,
            config_hash=config_hash,
        )

    def to_json(self) -> str:
        """Convert manifest to JSON string."""
        return json.dumps(asdict(self), indent=2)

    def to_comment_block(self) -> str:
        """Convert manifest to a formatted comment block for output."""
        lines = [
            "",
            "<!-- Run Manifest",
            f"Run ID: {self.run_id}",
            f"Timestamp: {self.timestamp}",
            f"Models: {', '.join(self.models)}",
            f"Chairman: {self.chairman}",
            f"Stage 1 Results: {self.stage1_results}/{len(self.models)}",
            f"Stage 2 Ballots: {self.stage2_valid_ballots}/{self.stage2_total_ballots} valid",
            f"Total Time: {self.total_elapsed_seconds:.1f}s",
            f"Est. Tokens: ~{self.estimated_tokens:,}",
            f"Config Hash: {self.config_hash[:16]}...",
            "-->",
            "",
        ]
        return "\n".join(lines)
