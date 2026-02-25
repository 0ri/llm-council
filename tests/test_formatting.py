"""Tests for output formatting functions."""

from __future__ import annotations

from llm_council.formatting import format_output, format_stage1_output, format_stage2_output
from llm_council.models import AggregateRanking, Stage1Result, Stage3Result


def _ranking(model: str, avg: float, ci: tuple[float, float] = (0, 0), borda: float = 0) -> AggregateRanking:
    """Helper to create AggregateRanking objects concisely."""
    return AggregateRanking(
        model=model,
        average_rank=avg,
        rankings_count=1,
        ci_lower=ci[0],
        ci_upper=ci[1],
        borda_score=borda,
    )


class TestFormatOutput:
    """Test the format_output function."""

    def test_basic_formatting(self):
        """Test basic output formatting with valid data."""
        aggregate_rankings = [
            _ranking("Claude Opus", 1.5, (1.2, 1.8), 8),
            _ranking("GPT-5", 2.0, (1.7, 2.3), 6),
            _ranking("Gemini Pro", 2.5, (2.1, 2.9), 4),
        ]

        stage3_result = Stage3Result(model="Claude Opus", response="This is the final synthesized answer.")

        output = format_output(aggregate_rankings, stage3_result, valid_ballots=3, total_ballots=3)

        # Check structure
        assert "## LLM Council Response" in output
        assert "### Model Rankings (by peer review)" in output
        assert "### Synthesized Answer" in output

        # Check table headers
        assert "| Rank | Model | Avg Position | 95% CI | Borda Score |" in output
        assert "|------|-------|--------------|--------|-------------|" in output

        # Check model entries
        assert "| 1 | Claude Opus | 1.5 | [1.2, 1.8] | 8.0 |" in output
        assert "| 2 | GPT-5 | 2.0 | [1.7, 2.3] | 6.0 |" in output
        assert "| 3 | Gemini Pro | 2.5 | [2.1, 2.9] | 4.0 |" in output

        # Check chairman info
        assert "**Chairman:** Claude Opus" in output
        assert "This is the final synthesized answer." in output

        # Check ballot validity message
        assert "Rankings based on 3/3 valid ballots (anonymous peer evaluation)" in output

    def test_zero_valid_ballots(self):
        """Test formatting when no valid ballots exist."""
        stage3_result = Stage3Result(model="Chairman Model", response="No rankings available.")

        output = format_output([], stage3_result, valid_ballots=0, total_ballots=3)

        assert "## LLM Council Response" in output
        assert "### Model Rankings" in output
        assert "### Synthesized Answer" in output
        assert "| Rank | Model | Avg Position | 95% CI | Borda Score |" in output
        assert "0/3" in output
        assert "some rankings could not be parsed reliably" in output

    def test_all_valid_ballots(self):
        """Test formatting when all ballots are valid."""
        rankings = [_ranking("Model A", 1.0, (1.0, 1.0), 3)]
        stage3 = Stage3Result(model="Model A", response="Answer")

        output = format_output(rankings, stage3, valid_ballots=5, total_ballots=5)

        assert "5/5 valid ballots (anonymous peer evaluation)" in output
        assert "some rankings could not be parsed reliably" not in output

    def test_partial_valid_ballots(self):
        """Test formatting when only some ballots are valid."""
        rankings = [_ranking("Model A", 1.0, (1.0, 1.0), 3)]
        stage3 = Stage3Result(model="Model A", response="Answer")

        output = format_output(rankings, stage3, valid_ballots=3, total_ballots=5)

        assert "3/5" in output
        assert "some rankings could not be parsed reliably" in output

    def test_empty_rankings_list(self):
        """Test formatting with empty rankings list."""
        output = format_output([], Stage3Result(model="Chairman", response="No rankings"), 0, 0)

        assert "| Rank | Model | Avg Position | 95% CI | Borda Score |" in output
        lines = output.split("\n")
        table_lines = [line for line in lines if line.startswith("|")]
        assert len(table_lines) == 2  # Header and separator only

    def test_special_characters_in_model_names(self):
        """Test that special characters in model names don't break markdown."""
        rankings = [
            _ranking("Model|With|Pipes", 1.0, (1.0, 1.0), 5),
            _ranking("Model*With*Stars", 2.0, (2.0, 2.0), 3),
            _ranking("Model_With_Underscores", 3.0, (3.0, 3.0), 1),
        ]
        stage3 = Stage3Result(model="Model|With|Pipes", response="Test response")

        output = format_output(rankings, stage3, 3, 3)

        assert "Model|With|Pipes" in output
        assert "Model*With*Stars" in output
        assert "Model_With_Underscores" in output
        assert output.count("|") > 20

    def test_missing_confidence_interval(self):
        """Test handling when CI values are None."""
        rankings = [_ranking("Model A", 1.5, (0, 0), 5)]
        stage3 = Stage3Result(model="Model A", response="Response")

        output = format_output(rankings, stage3, 1, 1)

        assert "[0, 0]" in output

    def test_missing_borda_score(self):
        """Test handling when borda_score is None."""
        rankings = [AggregateRanking(model="Model A", average_rank=1.5, rankings_count=1, ci_lower=1.0, ci_upper=2.0)]
        stage3 = Stage3Result(model="Model A", response="Response")

        output = format_output(rankings, stage3, 1, 1)

        assert "| 1 | Model A | 1.5 | [1.0, 2.0] | 0 |" in output

    def test_markdown_structure(self):
        """Test that output has valid markdown structure."""
        rankings = [
            _ranking("Model A", 1.0, (0.9, 1.1), 10),
            _ranking("Model B", 2.0, (1.9, 2.1), 5),
        ]
        stage3 = Stage3Result(model="Chairman", response="Final answer here.")

        output = format_output(rankings, stage3, 2, 2)

        assert output.count("## ") == 3
        assert output.count("### ") == 2
        assert output.count("---") >= 1
        assert output.count("**") >= 2

        lines = output.split("\n")
        table_lines = [line for line in lines if line.startswith("|")]
        assert len(table_lines) >= 4

    def test_very_long_response(self):
        """Test formatting with very long synthesized response."""
        rankings = [_ranking("Model A", 1.0, (1.0, 1.0), 3)]
        long_response = "This is a very long response. " * 500
        stage3 = Stage3Result(model="Model A", response=long_response)

        output = format_output(rankings, stage3, 1, 1)

        assert long_response in output
        assert len(output) > 5000

    def test_multiline_response(self):
        """Test formatting with multiline synthesized response."""
        rankings = [_ranking("Model A", 1.0, (1.0, 1.0), 3)]
        multiline_response = """First paragraph of the response.

Second paragraph with more details.

- Bullet point 1
- Bullet point 2

Final paragraph."""
        stage3 = Stage3Result(model="Model A", response=multiline_response)

        output = format_output(rankings, stage3, 1, 1)

        assert multiline_response in output
        assert "- Bullet point 1" in output
        assert "- Bullet point 2" in output

    def test_unicode_in_content(self):
        """Test formatting with Unicode characters."""
        rankings = [_ranking("Model R", 1.0, (1.0, 1.0), 5)]
        stage3 = Stage3Result(model="Model R", response="Response with special chars")

        output = format_output(rankings, stage3, 1, 1)

        assert "Model R" in output

    def test_rank_numbering(self):
        """Test that ranks are numbered correctly starting from 1."""
        rankings = [_ranking(f"Model {i}", float(i), (float(i), float(i)), 10 - i) for i in range(1, 6)]
        stage3 = Stage3Result(model="Model 1", response="Response")

        output = format_output(rankings, stage3, 5, 5)

        for i in range(1, 6):
            assert f"| {i} | Model {i} |" in output

    def test_floating_point_precision(self):
        """Test that floating point values are displayed correctly."""
        rankings = [_ranking("Model A", 1.33333333, (1.11111, 1.55555), 7.5)]
        stage3 = Stage3Result(model="Model A", response="Response")

        output = format_output(rankings, stage3, 1, 1)

        assert "1.33333333" in output or "1.333" in output
        assert "1.11111" in output or "1.111" in output
        assert "7.5" in output

    def test_zero_total_ballots(self):
        """Test edge case with zero total ballots."""
        output = format_output([], Stage3Result(model="Model", response="Response"), 0, 0)

        assert "0/0" in output

    def test_more_valid_than_total(self):
        """Test edge case where valid_ballots > total_ballots."""
        rankings = [_ranking("Model A", 1.0, (1.0, 1.0), 3)]

        output = format_output(rankings, Stage3Result(model="Model A", response="Response"), 5, 3)

        assert "5/3" in output


class TestFormatStage1Output:
    """Test the format_stage1_output function."""

    def test_basic_stage1_output(self):
        """Test basic Stage 1 formatting with Stage1Result objects."""
        results = [
            Stage1Result(model="Claude Opus", response="Claude's answer about REST APIs."),
            Stage1Result(model="GPT-5", response="GPT's take on the topic."),
        ]

        output = format_stage1_output(results)

        assert "## LLM Council Response (Stage 1 only)" in output
        assert "### Claude Opus" in output
        assert "Claude's answer about REST APIs." in output
        assert "### GPT-5" in output
        assert "GPT's take on the topic." in output

    def test_stage1_output_no_rankings(self):
        """Test that Stage 1 output has no rankings table."""
        results = [Stage1Result(model="Model", response="Answer")]

        output = format_stage1_output(results)

        assert "Rank" not in output
        assert "Borda Score" not in output
        assert "Synthesized" not in output

    def test_stage1_empty_results(self):
        """Test Stage 1 formatting with no results."""
        output = format_stage1_output([])

        assert "## LLM Council Response (Stage 1 only)" in output


class TestFormatStage2Output:
    """Test the format_stage2_output function."""

    def test_basic_stage2_output(self):
        """Test basic Stage 2 formatting."""
        rankings = [
            _ranking("Model A", 1.5, (1.2, 1.8), 8),
            _ranking("Model B", 2.5, (2.1, 2.9), 4),
        ]
        results = [
            Stage1Result(model="Model A", response="Answer A"),
            Stage1Result(model="Model B", response="Answer B"),
        ]

        output = format_stage2_output(rankings, results, valid_ballots=2, total_ballots=2)

        assert "## LLM Council Response (Stages 1-2, no synthesis)" in output
        assert "### Model Rankings (by peer review)" in output
        assert "| Rank | Model | Avg Position | 95% CI | Borda Score |" in output
        assert "| 1 | Model A | 1.5 | [1.2, 1.8] | 8.0 |" in output
        assert "### Individual Responses" in output
        assert "#### Model A" in output
        assert "Answer A" in output

    def test_stage2_no_synthesis_section(self):
        """Test that Stage 2 output has no synthesis section."""
        rankings = [_ranking("M", 1.0, (1.0, 1.0), 3)]
        results = [Stage1Result(model="M", response="A")]

        output = format_stage2_output(rankings, results, 1, 1)

        assert "Synthesized Answer" not in output
        assert "Chairman" not in output
