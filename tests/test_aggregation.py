"""Tests for ranking aggregation logic."""

try:
    from llm_council.aggregation import (
        bootstrap_confidence_intervals,
        calculate_aggregate_rankings,
        calculate_borda_score,
    )
except ImportError:
    from council import (
        bootstrap_confidence_intervals,
        calculate_aggregate_rankings,
        calculate_borda_score,
    )


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

    def test_per_ranker_label_mappings(self):
        """Test aggregation with per-ranker label mappings (for self-exclusion)."""
        stage2_results = [
            {"model": "M1", "parsed_ranking": ["Response A", "Response B"], "is_valid_ballot": True},
            {"model": "M2", "parsed_ranking": ["Response A", "Response B"], "is_valid_ballot": True},
            {"model": "M3", "parsed_ranking": ["Response A", "Response B"], "is_valid_ballot": True},
        ]
        # Each ranker excludes themselves and sees different labels
        per_ranker_mappings = {
            "M1": {"Response A": "M2", "Response B": "M3"},  # M1 doesn't rank itself
            "M2": {"Response A": "M1", "Response B": "M3"},  # M2 doesn't rank itself
            "M3": {"Response A": "M1", "Response B": "M2"},  # M3 doesn't rank itself
        }
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker_mappings)
        assert valid == 3
        assert total == 3
        # Each model gets ranked by 2 others
        for item in aggregate:
            assert item["rankings_count"] == 2

    def test_confidence_intervals_in_results(self):
        """Test that confidence intervals are included in results."""
        stage2_results = [
            {"model": "M1", "parsed_ranking": ["Response A", "Response B"], "is_valid_ballot": True},
            {"model": "M2", "parsed_ranking": ["Response B", "Response A"], "is_valid_ballot": True},
        ]
        label_to_model = {"Response A": "M1", "Response B": "M2"}
        aggregate, _, _ = calculate_aggregate_rankings(stage2_results, label_to_model)

        for item in aggregate:
            assert "confidence_interval" in item
            ci = item["confidence_interval"]
            assert isinstance(ci, tuple)
            assert len(ci) == 2
            assert ci[0] <= item["average_rank"] <= ci[1]

    def test_borda_scores_in_results(self):
        """Test that Borda scores are calculated correctly."""
        stage2_results = [
            {"model": "M1", "parsed_ranking": ["Response A", "Response B", "Response C"], "is_valid_ballot": True},
            {"model": "M2", "parsed_ranking": ["Response A", "Response B", "Response C"], "is_valid_ballot": True},
        ]
        label_to_model = {"Response A": "M1", "Response B": "M2", "Response C": "M3"}
        aggregate, _, _ = calculate_aggregate_rankings(stage2_results, label_to_model)

        # For 3 candidates: 1st place gets 2 points, 2nd gets 1, 3rd gets 0
        assert aggregate[0]["model"] == "M1"
        assert aggregate[0]["borda_score"] == 2.0  # Always first
        assert aggregate[1]["model"] == "M2"
        assert aggregate[1]["borda_score"] == 1.0  # Always second
        assert aggregate[2]["model"] == "M3"
        assert aggregate[2]["borda_score"] == 0.0  # Always third


class TestBootstrapConfidenceIntervals:
    def test_single_value(self):
        """Test CI for a single value (should be the value itself)."""
        positions = [2.0]
        ci_lower, ci_upper = bootstrap_confidence_intervals(positions, n_resamples=100)
        assert ci_lower == 2.0
        assert ci_upper == 2.0

    def test_multiple_identical_values(self):
        """Test CI for identical values."""
        positions = [3, 3, 3, 3]
        ci_lower, ci_upper = bootstrap_confidence_intervals(positions, n_resamples=100)
        assert ci_lower == 3.0
        assert ci_upper == 3.0

    def test_varied_values(self):
        """Test CI for varied values."""
        positions = [1, 2, 3, 4, 5]
        ci_lower, ci_upper = bootstrap_confidence_intervals(positions, n_resamples=1000)
        # Mean is 3.0, CI should be around it but wider
        assert 1.5 <= ci_lower <= 2.5
        assert 3.5 <= ci_upper <= 4.5
        assert ci_lower < ci_upper

    def test_empty_list(self):
        """Test CI for empty list."""
        ci_lower, ci_upper = bootstrap_confidence_intervals([])
        assert ci_lower == 0.0
        assert ci_upper == 0.0


class TestBordaScore:
    def test_first_place(self):
        """Test Borda score for consistent first place."""
        positions = [1, 1, 1]
        score = calculate_borda_score(positions, n_candidates=3)
        assert score == 2.0  # First place in 3-candidate race gets 2 points

    def test_last_place(self):
        """Test Borda score for consistent last place."""
        positions = [3, 3, 3]
        score = calculate_borda_score(positions, n_candidates=3)
        assert score == 0.0  # Last place gets 0 points

    def test_mixed_rankings(self):
        """Test Borda score for mixed rankings."""
        positions = [1, 2, 3]
        score = calculate_borda_score(positions, n_candidates=3)
        # (2 + 1 + 0) / 3 = 1.0
        assert score == 1.0

    def test_empty_positions(self):
        """Test Borda score for empty positions."""
        score = calculate_borda_score([], n_candidates=3)
        assert score == 0.0
