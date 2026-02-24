"""Tests for resilience features: circuit breaker, semaphore, cost tracking."""

from llm_council.cost import CouncilCostTracker
from llm_council.providers import CircuitBreaker, get_circuit_breaker, reset_circuit_breakers


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert not cb.is_open

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert cb.is_open

    def test_success_resets_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert not cb.is_open

    def test_cooldown_transitions_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.0)
        cb.record_failure()
        # With cooldown=0, the very first is_open check finds cooldown has elapsed
        # (time elapsed >= 0.0 is always true), transitions to half-open, returns False
        assert not cb.is_open  # Immediately half-open, allows retry

    def test_remains_open_during_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=9999.0)
        cb.record_failure()
        assert cb.is_open  # Long cooldown, circuit stays open

    def test_get_circuit_breaker_creates_per_provider(self):
        reset_circuit_breakers()
        cb1 = get_circuit_breaker("poe")
        cb2 = get_circuit_breaker("bedrock")
        assert cb1 is not cb2
        # Same provider returns same instance
        assert get_circuit_breaker("poe") is cb1
        reset_circuit_breakers()

    def test_reset_circuit_breakers(self):
        reset_circuit_breakers()
        cb1 = get_circuit_breaker("poe")
        reset_circuit_breakers()
        cb2 = get_circuit_breaker("poe")
        assert cb1 is not cb2


class TestCostTracker:
    def test_record_and_totals(self):
        tracker = CouncilCostTracker()
        tracker.record("Model-A", 1, "Hello world", "This is a response from model A.")
        assert len(tracker.usages) == 1
        assert tracker.total_tokens > 0

    def test_summary_format(self):
        tracker = CouncilCostTracker()
        tracker.record("Model-A", 1, "x" * 400, "y" * 800)
        tracker.record("Model-B", 1, "x" * 400, "y" * 600)
        summary = tracker.summary()
        assert "Stage 1" in summary
        assert "Model-A" in summary
        assert "Model-B" in summary
        assert "Total" in summary

    def test_empty_tracker(self):
        tracker = CouncilCostTracker()
        assert tracker.total_tokens == 0
        summary = tracker.summary()
        assert "Total" in summary

    def test_multi_stage_tracking(self):
        tracker = CouncilCostTracker()
        tracker.record("Model-A", 1, "question", "answer A")
        tracker.record("Model-B", 1, "question", "answer B")
        tracker.record("Model-A", 2, "ranking prompt", "ranking output")
        tracker.record("Model-A", 3, "synthesis prompt", "synthesis output")
        summary = tracker.summary()
        assert "Stage 1" in summary
        assert "Stage 2" in summary
        assert "Stage 3" in summary

    def test_token_estimation(self):
        tracker = CouncilCostTracker()
        # 400 chars input, 800 chars output -> ~100 in, ~200 out tokens
        tracker.record("Model-A", 1, "x" * 400, "y" * 800)
        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 200
        assert tracker.total_tokens == 300
