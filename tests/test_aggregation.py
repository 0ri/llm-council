"""Tests for ranking aggregation logic."""

from llm_council.aggregation import (
    bootstrap_confidence_intervals,
    calculate_aggregate_rankings,
)
from llm_council.models import Stage2Result


def _ballot(model: str, ranking: list[str], valid: bool = True) -> Stage2Result:
    """Helper to create Stage2Result objects concisely."""
    return Stage2Result(model=model, ranking="", parsed_ranking=ranking, is_valid_ballot=valid)


class TestCalculateAggregateRankings:
    def test_unanimous_ranking(self):
        stage2_results = [
            _ballot("M1", ["Response A", "Response B", "Response C"]),
            _ballot("M2", ["Response A", "Response B", "Response C"]),
            _ballot("M3", ["Response A", "Response B", "Response C"]),
        ]
        per_ranker = {
            "M1": {"Response A": "M1", "Response B": "M2", "Response C": "M3"},
            "M2": {"Response A": "M1", "Response B": "M2", "Response C": "M3"},
            "M3": {"Response A": "M1", "Response B": "M2", "Response C": "M3"},
        }
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker)
        assert valid == 3
        assert total == 3
        assert aggregate[0].model == "M1"
        assert aggregate[0].average_rank == 1.0
        assert aggregate[2].model == "M3"
        assert aggregate[2].average_rank == 3.0

    def test_mixed_rankings(self):
        stage2_results = [
            _ballot("M1", ["Response A", "Response B"]),
            _ballot("M2", ["Response B", "Response A"]),
        ]
        per_ranker = {
            "M1": {"Response A": "M1", "Response B": "M2"},
            "M2": {"Response A": "M1", "Response B": "M2"},
        }
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker)
        assert valid == 2
        assert total == 2
        for entry in aggregate:
            assert entry.average_rank == 1.5

    def test_all_invalid_ballots(self):
        stage2_results = [
            _ballot("M1", ["Response A"], valid=False),
            _ballot("M2", [], valid=False),
        ]
        per_ranker = {"M1": {"Response A": "M1"}, "M2": {}}
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker)
        assert valid == 0
        assert total == 2

    def test_empty_results(self):
        aggregate, valid, total = calculate_aggregate_rankings([], {})
        assert aggregate == []
        assert valid == 0
        assert total == 0

    def test_partial_rankings(self):
        stage2_results = [
            _ballot("M1", ["Response A", "Response B"]),
            _ballot("M2", ["Response B"], valid=False),
            _ballot("M3", ["Response B", "Response A"]),
        ]
        per_ranker = {
            "M1": {"Response A": "M1", "Response B": "M2"},
            "M2": {"Response A": "M1", "Response B": "M2"},
            "M3": {"Response A": "M1", "Response B": "M2"},
        }
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker)
        assert valid == 2
        assert total == 3

    def test_per_ranker_label_mappings(self):
        """Test aggregation with per-ranker label mappings (for self-exclusion)."""
        stage2_results = [
            _ballot("M1", ["Response A", "Response B"]),
            _ballot("M2", ["Response A", "Response B"]),
            _ballot("M3", ["Response A", "Response B"]),
        ]
        per_ranker_mappings = {
            "M1": {"Response A": "M2", "Response B": "M3"},
            "M2": {"Response A": "M1", "Response B": "M3"},
            "M3": {"Response A": "M1", "Response B": "M2"},
        }
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker_mappings)
        assert valid == 3
        assert total == 3
        for item in aggregate:
            assert item.rankings_count == 2

    def test_confidence_intervals_in_results(self):
        """Test that confidence intervals are included in results."""
        stage2_results = [
            _ballot("M1", ["Response A", "Response B"]),
            _ballot("M2", ["Response B", "Response A"]),
        ]
        per_ranker = {
            "M1": {"Response A": "M1", "Response B": "M2"},
            "M2": {"Response A": "M1", "Response B": "M2"},
        }
        aggregate, _, _ = calculate_aggregate_rankings(stage2_results, per_ranker)

        for item in aggregate:
            ci = item.confidence_interval
            assert isinstance(ci, tuple)
            assert len(ci) == 2
            assert ci[0] <= item.average_rank <= ci[1]

    def test_borda_scores_in_results(self):
        """Test that normalized Borda scores are calculated correctly (0-1 scale)."""
        stage2_results = [
            _ballot("M1", ["Response A", "Response B", "Response C"]),
            _ballot("M2", ["Response A", "Response B", "Response C"]),
        ]
        per_ranker = {
            "M1": {"Response A": "M1", "Response B": "M2", "Response C": "M3"},
            "M2": {"Response A": "M1", "Response B": "M2", "Response C": "M3"},
        }
        aggregate, _, _ = calculate_aggregate_rankings(stage2_results, per_ranker)

        # Normalized Borda: position 1 in 3-candidate ballot -> (3-1)/(3-1) = 1.0
        assert aggregate[0].model == "M1"
        assert aggregate[0].borda_score == 1.0
        # Position 2 -> (3-2)/(3-1) = 0.5
        assert aggregate[1].model == "M2"
        assert aggregate[1].borda_score == 0.5
        # Position 3 -> (3-3)/(3-1) = 0.0
        assert aggregate[2].model == "M3"
        assert aggregate[2].borda_score == 0.0


class TestBootstrapConfidenceIntervals:
    def test_single_value(self):
        positions = [2.0]
        ci_lower, ci_upper = bootstrap_confidence_intervals(positions, n_resamples=100)
        assert ci_lower == 2.0
        assert ci_upper == 2.0

    def test_multiple_identical_values(self):
        positions = [3, 3, 3, 3]
        ci_lower, ci_upper = bootstrap_confidence_intervals(positions, n_resamples=100)
        assert ci_lower == 3.0
        assert ci_upper == 3.0

    def test_varied_values(self):
        positions = [1, 2, 3, 4, 5]
        ci_lower, ci_upper = bootstrap_confidence_intervals(positions, n_resamples=1000)
        assert 1.5 <= ci_lower <= 2.5
        assert 3.5 <= ci_upper <= 4.5
        assert ci_lower < ci_upper

    def test_empty_list(self):
        ci_lower, ci_upper = bootstrap_confidence_intervals([])
        assert ci_lower == 0.0
        assert ci_upper == 0.0


class TestInvalidBallotExclusion:
    """Regression tests: invalid ballots must not influence rankings."""

    def test_invalid_ballots_excluded_from_positions(self):
        valid1 = _ballot("Ranker1", ["Response A", "Response B", "Response C"])
        valid2 = _ballot("Ranker2", ["Response A", "Response B", "Response C"])
        invalid = _ballot("Ranker3", ["Response C", "Response B", "Response A"], valid=False)

        per_ranker_mappings = {
            "Ranker1": {"Response A": "Model-A", "Response B": "Model-B", "Response C": "Model-C"},
            "Ranker2": {"Response A": "Model-A", "Response B": "Model-B", "Response C": "Model-C"},
            "Ranker3": {"Response C": "Model-A", "Response B": "Model-B", "Response A": "Model-C"},
        }

        rankings, valid_count, total_count = calculate_aggregate_rankings(
            [valid1, valid2, invalid], per_ranker_mappings
        )

        assert valid_count == 2
        assert total_count == 3

        ranking_map = {r.model: r for r in rankings}
        assert ranking_map["Model-A"].average_rank == 1.0
        assert ranking_map["Model-A"].rankings_count == 2
        assert ranking_map["Model-C"].average_rank == 3.0
        assert ranking_map["Model-C"].rankings_count == 2

    def test_all_invalid_produces_empty_rankings(self):
        invalid = _ballot("Ranker1", ["Response A", "Response B"], valid=False)
        mappings = {"Ranker1": {"Response A": "Model-A", "Response B": "Model-B"}}

        rankings, valid_count, total_count = calculate_aggregate_rankings([invalid], mappings)

        assert valid_count == 0
        assert total_count == 1
        assert rankings == []


class TestPhantomLabelDetection:
    """Item 12: Phantom (unresolvable) labels should invalidate ballots."""

    def test_phantom_label_invalidates_ballot(self):
        """A ballot with 'Response D' in a 3-model council should be invalidated."""
        stage2_results = [
            # Valid ballot: labels A, B match the mapping
            _ballot("M1", ["Response A", "Response B"]),
            # Phantom ballot: 'Response D' does not exist in mapping
            _ballot("M2", ["Response A", "Response D"]),
            _ballot("M3", ["Response B", "Response A"]),
        ]
        per_ranker = {
            "M1": {"Response A": "M2", "Response B": "M3"},
            "M2": {"Response A": "M1", "Response B": "M3"},
            "M3": {"Response A": "M1", "Response B": "M2"},
        }
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker)

        # M2's ballot should be invalidated due to phantom "Response D"
        assert valid == 2
        assert total == 3

    def test_all_phantom_labels(self):
        """A ballot where ALL labels are phantom should be invalidated."""
        stage2_results = [
            _ballot("M1", ["Response X", "Response Y"]),
        ]
        per_ranker = {
            "M1": {"Response A": "M2", "Response B": "M3"},
        }
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker)

        assert valid == 0
        assert total == 1
        assert aggregate == []

    def test_no_phantom_labels_all_valid(self):
        """When no phantom labels exist, all valid ballots should be counted."""
        stage2_results = [
            _ballot("M1", ["Response A", "Response B"]),
            _ballot("M2", ["Response A", "Response B"]),
        ]
        per_ranker = {
            "M1": {"Response A": "M2", "Response B": "M3"},
            "M2": {"Response A": "M1", "Response B": "M3"},
        }
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker)

        assert valid == 2
        assert total == 2


class TestNormalizedBordaWithUnequalBallots:
    """Item 13: Borda scores should be comparable across different ballot sizes."""

    def test_unequal_ballot_sizes_normalized(self):
        """Self-exclusion creates unequal ballot sizes; normalized Borda keeps scores fair."""
        # 3-model council with self-exclusion: each ranker sees 2 candidates
        # Cyclic ranking: M1 prefers M2, M2 prefers M3, M3 prefers M1
        stage2_results = [
            _ballot("M1", ["Response A", "Response B"]),  # M1: ranks M2 first, M3 second
            _ballot("M2", ["Response A", "Response B"]),  # M2: ranks M3 first, M1 second
            _ballot("M3", ["Response A", "Response B"]),  # M3: ranks M1 first, M2 second
        ]
        per_ranker_mappings = {
            "M1": {"Response A": "M2", "Response B": "M3"},
            "M2": {"Response A": "M3", "Response B": "M1"},
            "M3": {"Response A": "M1", "Response B": "M2"},
        }
        aggregate, valid, total = calculate_aggregate_rankings(stage2_results, per_ranker_mappings)

        assert valid == 3
        assert total == 3

        # Each model is ranked first by one ranker and second by another
        # Normalized Borda for 2 candidates: pos 1 -> (2-1)/(2-1)=1.0, pos 2 -> 0.0
        # Each model: avg normalized Borda = (1.0 + 0.0) / 2 = 0.5
        for item in aggregate:
            assert item.borda_score == 0.5

    def test_borda_normalized_to_01_scale(self):
        """Normalized Borda should produce values in [0, 1] range."""
        stage2_results = [
            _ballot("M1", ["Response A", "Response B", "Response C"]),
        ]
        per_ranker = {
            "M1": {"Response A": "X", "Response B": "Y", "Response C": "Z"},
        }
        aggregate, _, _ = calculate_aggregate_rankings(stage2_results, per_ranker)

        for item in aggregate:
            assert 0.0 <= item.borda_score <= 1.0
