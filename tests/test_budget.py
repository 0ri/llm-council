"""Tests for budget control functionality."""

import pytest

from llm_council.budget import BudgetExceededError, BudgetGuard, create_budget_guard


def _reserve_and_commit(guard, input_tokens, output_tokens, model_name):
    """Helper: reserve then commit with same estimates (replaces deprecated check_and_update)."""
    guard.reserve(input_tokens, output_tokens, model_name)
    guard.commit(input_tokens, output_tokens, model_name, reserved_input=input_tokens, reserved_output=output_tokens)


class TestBudgetGuard:
    """Test the BudgetGuard class."""

    def test_no_limits(self):
        """Test BudgetGuard with no limits set."""
        guard = BudgetGuard()
        # Should not raise even with large usage
        _reserve_and_commit(guard, 10000, 10000, "test-model")
        assert guard.total_input_tokens == 10000
        assert guard.total_output_tokens == 10000

    def test_token_limit_under(self):
        """Test staying under token limit."""
        guard = BudgetGuard(max_tokens=5000)
        _reserve_and_commit(guard, 1000, 1000, "model1")
        _reserve_and_commit(guard, 1000, 1000, "model2")
        assert guard.total_input_tokens == 2000
        assert guard.total_output_tokens == 2000
        assert len(guard.queries) == 2

    def test_token_limit_at_limit(self):
        """Test reaching exactly the token limit."""
        guard = BudgetGuard(max_tokens=5000)
        _reserve_and_commit(guard, 2500, 2500, "model1")
        assert guard.total_input_tokens == 2500
        assert guard.total_output_tokens == 2500
        # Exactly at limit should work

    def test_token_limit_exceeded(self):
        """Test exceeding token limit."""
        guard = BudgetGuard(max_tokens=5000)
        _reserve_and_commit(guard, 2000, 2000, "model1")

        # This would exceed the limit
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.reserve(2000, 2000, "model2")

        assert "Token budget exceeded" in str(exc_info.value)
        assert "8000 > 5000" in str(exc_info.value)
        # Should not have updated counts
        assert guard.total_input_tokens == 2000
        assert guard.total_output_tokens == 2000

    def test_cost_limit_under(self):
        """Test staying under cost limit."""
        guard = BudgetGuard(max_cost_usd=1.0)
        # 1000 tokens = $0.01 input + $0.03 output = $0.04
        _reserve_and_commit(guard, 1000, 1000, "model1")
        assert guard.total_cost_usd == pytest.approx(0.04, rel=1e-3)

    def test_cost_limit_exceeded(self):
        """Test exceeding cost limit."""
        guard = BudgetGuard(max_cost_usd=0.1)
        # First query: $0.04
        _reserve_and_commit(guard, 1000, 1000, "model1")
        # Second query: $0.04
        _reserve_and_commit(guard, 1000, 1000, "model2")
        # Third query would be $0.04, total would be $0.12 > $0.10
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.reserve(1000, 1000, "model3")

        assert "Cost budget exceeded" in str(exc_info.value)
        assert "$0.12 > $0.10" in str(exc_info.value)

    def test_both_limits(self):
        """Test with both token and cost limits."""
        guard = BudgetGuard(max_tokens=5000, max_cost_usd=0.5)
        _reserve_and_commit(guard, 1000, 1000, "model1")
        assert guard.total_input_tokens == 1000
        assert guard.total_output_tokens == 1000
        assert guard.total_cost_usd == pytest.approx(0.04, rel=1e-3)

    def test_custom_pricing(self):
        """Test with custom pricing."""
        guard = BudgetGuard(
            max_cost_usd=1.0,
            input_cost_per_1k=0.02,  # Double the default
            output_cost_per_1k=0.06,  # Double the default
        )
        _reserve_and_commit(guard, 1000, 1000, "model1")
        # Should be $0.02 + $0.06 = $0.08
        assert guard.total_cost_usd == pytest.approx(0.08, rel=1e-3)

    def test_summary_with_limits(self):
        """Test summary output with limits."""
        guard = BudgetGuard(max_tokens=10000, max_cost_usd=1.0)
        _reserve_and_commit(guard, 2000, 3000, "model1")

        summary = guard.summary()
        assert "5,000/10,000" in summary
        assert "50.0% used" in summary
        assert "$0.11/$1.00" in summary  # Actual cost: 2*0.01 + 3*0.03 = 0.02 + 0.09 = 0.11
        assert "11.0% used" in summary
        assert "Input: 2,000" in summary
        assert "Output: 3,000" in summary

    def test_summary_no_limits(self):
        """Test summary output without limits."""
        guard = BudgetGuard()
        _reserve_and_commit(guard, 1000, 1000, "model1")

        summary = guard.summary()
        assert "2,000 (no limit)" in summary
        assert "$0.04 (no limit)" in summary


class TestCreateBudgetGuard:
    """Test the create_budget_guard factory function."""

    def test_no_budget_config(self):
        """Test with no budget config."""
        config = {"council_models": [], "chairman": {}}
        guard = create_budget_guard(config)
        assert guard is None

    def test_empty_budget_config(self):
        """Test with empty budget config."""
        config = {"budget": {}}
        guard = create_budget_guard(config)
        assert guard is None

    def test_token_limit_only(self):
        """Test with only token limit."""
        config = {"budget": {"max_tokens": 5000}}
        guard = create_budget_guard(config)
        assert guard is not None
        assert guard.max_tokens == 5000
        assert guard.max_cost_usd is None

    def test_cost_limit_only(self):
        """Test with only cost limit."""
        config = {"budget": {"max_cost_usd": 10.0}}
        guard = create_budget_guard(config)
        assert guard is not None
        assert guard.max_tokens is None
        assert guard.max_cost_usd == 10.0

    def test_both_limits(self):
        """Test with both limits."""
        config = {"budget": {"max_tokens": 5000, "max_cost_usd": 10.0}}
        guard = create_budget_guard(config)
        assert guard is not None
        assert guard.max_tokens == 5000
        assert guard.max_cost_usd == 10.0

    def test_custom_pricing(self):
        """Test with custom pricing in config."""
        config = {
            "budget": {
                "max_tokens": 5000,
                "input_cost_per_1k": 0.005,
                "output_cost_per_1k": 0.015,
            }
        }
        guard = create_budget_guard(config)
        assert guard is not None
        assert guard.input_cost_per_1k == 0.005
        assert guard.output_cost_per_1k == 0.015

    def test_query_tracking(self):
        """Test that queries are properly tracked."""
        guard = BudgetGuard(max_tokens=10000)
        _reserve_and_commit(guard, 1000, 500, "model1")
        _reserve_and_commit(guard, 2000, 1000, "model2")

        assert len(guard.queries) == 2
        assert guard.queries[0]["model"] == "model1"
        assert guard.queries[0]["input_tokens"] == 1000
        assert guard.queries[0]["output_tokens"] == 500
        assert guard.queries[1]["model"] == "model2"
        assert guard.queries[1]["input_tokens"] == 2000
        assert guard.queries[1]["output_tokens"] == 1000


class TestReserveCommitRelease:
    """Test the reserve/commit/release protocol."""

    def test_reserve_commit_flow(self):
        """Full reserve + commit flow with actual tokens."""
        guard = BudgetGuard(max_tokens=5000)

        # Reserve with estimates
        guard.reserve(1000, 2000, "model1")
        assert guard.total_input_tokens == 1000
        assert guard.total_output_tokens == 2000

        # Query succeeds with actual (smaller) usage
        guard.commit(800, 1200, "model1", reserved_input=1000, reserved_output=2000)
        assert guard.total_input_tokens == 800
        assert guard.total_output_tokens == 1200

    def test_reserve_release_on_failure(self):
        """Reserve then release on query failure restores budget."""
        guard = BudgetGuard(max_tokens=5000)

        guard.reserve(1000, 2000, "model1")
        assert guard.total_input_tokens == 1000

        # Simulate query failure — release the reservation
        guard.release(1000, 2000, "model1")
        assert guard.total_input_tokens == 0
        assert guard.total_output_tokens == 0

    def test_commit_records_query(self):
        """commit() records actual tokens in the query log."""
        guard = BudgetGuard(max_tokens=10000)
        guard.commit(150, 80, "model1")

        assert guard.total_input_tokens == 150
        assert guard.total_output_tokens == 80
        assert len(guard.queries) == 1
        assert guard.queries[0]["input_tokens"] == 150
        assert guard.queries[0]["output_tokens"] == 80
