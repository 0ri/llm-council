"""Tests for JSONL persistence logging."""

import json

from llm_council.models import AggregateRanking, Stage1Result, Stage2Result, Stage3Result
from llm_council.persistence import RunLogger


class TestRunLogger:
    def test_creates_directory(self, tmp_path):
        log_dir = tmp_path / "logs" / "nested"
        logger = RunLogger(log_dir, "test-run-id")
        assert log_dir.exists()

    def test_writes_jsonl_records(self, tmp_path):
        logger = RunLogger(tmp_path, "test-run-id")
        logger.log_config("test question", {"council_models": []})
        lines = logger.filepath.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["type"] == "config"
        assert record["run_id"] == "test-run-id"
        assert record["question"] == "test question"
        assert "timestamp" in record

    def test_log_stage1(self, tmp_path):
        logger = RunLogger(tmp_path, "run-1")
        results = [Stage1Result(model="M1", response="R1"), Stage1Result(model="M2", response="R2")]
        logger.log_stage1(results, {"M1": {"input_tokens": 10}, "M2": None})
        lines = logger.filepath.read_text().strip().split("\n")
        assert len(lines) == 2
        r1 = json.loads(lines[0])
        assert r1["type"] == "stage1_response"
        assert r1["model"] == "M1"
        assert r1["response"] == "R1"
        assert r1["token_usage"] == {"input_tokens": 10}

    def test_log_stage2(self, tmp_path):
        logger = RunLogger(tmp_path, "run-1")
        results = [Stage2Result(model="M1", ranking="text", parsed_ranking=["A", "B"], is_valid_ballot=True)]
        label_mappings = {"M1": {"A": "Model-A", "B": "Model-B"}}
        logger.log_stage2(results, label_mappings, {"M1": None})
        lines = logger.filepath.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["type"] == "stage2_ranking"
        assert record["is_valid_ballot"] is True
        assert record["label_mapping"] == {"A": "Model-A", "B": "Model-B"}

    def test_log_stage3(self, tmp_path):
        logger = RunLogger(tmp_path, "run-1")
        result = Stage3Result(model="Chairman", response="Final answer")
        logger.log_stage3(result, {"input_tokens": 500})
        lines = logger.filepath.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["type"] == "stage3_synthesis"
        assert record["model"] == "Chairman"

    def test_log_aggregation(self, tmp_path):
        logger = RunLogger(tmp_path, "run-1")
        rankings = [AggregateRanking(model="M1", average_rank=1.5, rankings_count=3, borda_score=2.0)]
        logger.log_aggregation(rankings, valid_ballots=3, total_ballots=4)
        lines = logger.filepath.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["type"] == "aggregation"
        assert record["valid_ballots"] == 3

    def test_full_run_all_record_types(self, tmp_path):
        logger = RunLogger(tmp_path, "full-run")
        logger.log_config("q", {})
        logger.log_stage1([Stage1Result(model="M1", response="R1")], {})
        logger.log_stage2(
            [Stage2Result(model="M1", ranking="t", parsed_ranking=["A"], is_valid_ballot=True)], {"M1": {}}, {}
        )
        logger.log_aggregation([AggregateRanking(model="M1", average_rank=1.0, rankings_count=1)], 1, 1)
        logger.log_stage3(Stage3Result(model="C", response="final"), None)
        logger.log_summary("cost info", 10.5)
        lines = logger.filepath.read_text().strip().split("\n")
        types = {json.loads(line)["type"] for line in lines}
        assert types == {"config", "stage1_response", "stage2_ranking", "aggregation", "stage3_synthesis", "summary"}
