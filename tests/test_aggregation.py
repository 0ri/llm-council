"""Tests for ranking aggregation logic."""

try:
    from llm_council.aggregation import calculate_aggregate_rankings
except ImportError:
    from council import calculate_aggregate_rankings


class TestCalculateAggregateRankings:
    def test_unanimous_ranking(self):
        stage2_results = [
            {"model": "M1", "parsed_ranking": ["Response A", "Response B", "Response C"], "is_valid_ballot": True},
            {"model": "M2", "parsed_ranking": ["Response A", "Response B", "Response C"], "is_valid_ballot": True},
            {"model": "M3", "parsed_ranking": ["Response A", "Response B", "Response C"], "is_valid_ballot": True},
        ]
        label_to_model = {"Response A": "M1", "Response B": "M2", "Response C": "M3"}
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, label_to_model)
        assert valid == 3
        assert total == 3
        assert aggregate[0]["model"] == "M1"
        assert aggregate[0]["average_rank"] == 1.0
        assert aggregate[2]["model"] == "M3"
        assert aggregate[2]["average_rank"] == 3.0

    def test_mixed_rankings(self):
        stage2_results = [
            {"model": "M1", "parsed_ranking": ["Response A", "Response B"], "is_valid_ballot": True},
            {"model": "M2", "parsed_ranking": ["Response B", "Response A"], "is_valid_ballot": True},
        ]
        label_to_model = {"Response A": "M1", "Response B": "M2"}
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, label_to_model)
        assert valid == 2
        assert total == 2
        for entry in aggregate:
            assert entry["average_rank"] == 1.5

    def test_all_invalid_ballots(self):
        stage2_results = [
            {"model": "M1", "parsed_ranking": ["Response A"], "is_valid_ballot": False},
            {"model": "M2", "parsed_ranking": [], "is_valid_ballot": False},
        ]
        label_to_model = {"Response A": "M1", "Response B": "M2"}
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, label_to_model)
        assert valid == 0
        assert total == 2

    def test_empty_results(self):
        aggregate, valid, total = calculate_aggregate_rankings([], {})
        assert aggregate == []
        assert valid == 0
        assert total == 0

    def test_partial_rankings(self):
        stage2_results = [
            {"model": "M1", "parsed_ranking": ["Response A", "Response B"], "is_valid_ballot": True},
            {"model": "M2", "parsed_ranking": ["Response B"], "is_valid_ballot": False},
            {"model": "M3", "parsed_ranking": ["Response B", "Response A"], "is_valid_ballot": True},
        ]
        label_to_model = {"Response A": "M1", "Response B": "M2"}
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, label_to_model)
        assert valid == 2
        assert total == 3
