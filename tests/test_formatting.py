"""Tests for output formatting functions."""

from __future__ import annotations

from llm_council.formatting import format_output


class TestFormatOutput:
    """Test the format_output function."""

    def test_basic_formatting(self):
        """Test basic output formatting with valid data."""
        aggregate_rankings = [
            {
                "model": "Claude Opus",
                "average_rank": 1.5,
                "confidence_interval": (1.2, 1.8),
                "borda_score": 8,
            },
            {
                "model": "GPT-5",
                "average_rank": 2.0,
                "confidence_interval": (1.7, 2.3),
                "borda_score": 6,
            },
            {
                "model": "Gemini Pro",
                "average_rank": 2.5,
                "confidence_interval": (2.1, 2.9),
                "borda_score": 4,
            },
        ]

        stage3_result = {"model": "Claude Opus", "response": "This is the final synthesized answer."}

        output = format_output(aggregate_rankings, stage3_result, valid_ballots=3, total_ballots=3)

        # Check structure
        assert "## LLM Council Response" in output
        assert "### Model Rankings (by peer review)" in output
        assert "### Synthesized Answer" in output

        # Check table headers
        assert "| Rank | Model | Avg Position | 95% CI | Borda Score |" in output
        assert "|------|-------|--------------|--------|-------------|" in output

        # Check model entries
        assert "| 1 | Claude Opus | 1.5 | [1.2, 1.8] | 8 |" in output
        assert "| 2 | GPT-5 | 2.0 | [1.7, 2.3] | 6 |" in output
        assert "| 3 | Gemini Pro | 2.5 | [2.1, 2.9] | 4 |" in output

        # Check chairman info
        assert "**Chairman:** Claude Opus" in output
        assert "This is the final synthesized answer." in output

        # Check ballot validity message
        assert "Rankings based on 3/3 valid ballots (anonymous peer evaluation)" in output

    def test_zero_valid_ballots(self):
        """Test formatting when no valid ballots exist."""
        aggregate_rankings = []
        stage3_result = {"model": "Chairman Model", "response": "No rankings available."}

        output = format_output(aggregate_rankings, stage3_result, valid_ballots=0, total_ballots=3)

        # Should still have structure
        assert "## LLM Council Response" in output
        assert "### Model Rankings" in output
        assert "### Synthesized Answer" in output

        # Table should be empty but have headers
        assert "| Rank | Model | Avg Position | 95% CI | Borda Score |" in output

        # Should show 0 valid ballots
        assert "0/3" in output
        assert "some rankings could not be parsed reliably" in output

    def test_all_valid_ballots(self):
        """Test formatting when all ballots are valid."""
        rankings = [{"model": "Model A", "average_rank": 1.0, "confidence_interval": (1.0, 1.0), "borda_score": 3}]

        stage3 = {"model": "Model A", "response": "Answer"}

        output = format_output(rankings, stage3, valid_ballots=5, total_ballots=5)

        assert "5/5 valid ballots (anonymous peer evaluation)" in output
        assert "some rankings could not be parsed reliably" not in output

    def test_partial_valid_ballots(self):
        """Test formatting when only some ballots are valid."""
        rankings = [{"model": "Model A", "average_rank": 1.0, "confidence_interval": (1.0, 1.0), "borda_score": 3}]

        stage3 = {"model": "Model A", "response": "Answer"}

        output = format_output(rankings, stage3, valid_ballots=3, total_ballots=5)

        assert "3/5" in output
        assert "some rankings could not be parsed reliably" in output

    def test_empty_rankings_list(self):
        """Test formatting with empty rankings list."""
        output = format_output([], {"model": "Chairman", "response": "No rankings"}, 0, 0)

        # Should still have table structure
        assert "| Rank | Model | Avg Position | 95% CI | Borda Score |" in output
        # But no data rows (just headers and separator)
        lines = output.split("\n")
        table_lines = [line for line in lines if line.startswith("|")]
        assert len(table_lines) == 2  # Header and separator only

    def test_special_characters_in_model_names(self):
        """Test that special characters in model names don't break markdown."""
        rankings = [
            {
                "model": "Model|With|Pipes",
                "average_rank": 1.0,
                "confidence_interval": (1.0, 1.0),
                "borda_score": 5,
            },
            {
                "model": "Model*With*Stars",
                "average_rank": 2.0,
                "confidence_interval": (2.0, 2.0),
                "borda_score": 3,
            },
            {
                "model": "Model_With_Underscores",
                "average_rank": 3.0,
                "confidence_interval": (3.0, 3.0),
                "borda_score": 1,
            },
        ]

        stage3 = {"model": "Model|With|Pipes", "response": "Test response"}

        output = format_output(rankings, stage3, 3, 3)

        # Characters should appear in output (not escaped for now)
        assert "Model|With|Pipes" in output
        assert "Model*With*Stars" in output
        assert "Model_With_Underscores" in output

        # Table structure should still be valid
        assert output.count("|") > 20  # Many pipes for table structure

    def test_missing_confidence_interval(self):
        """Test handling when confidence_interval is missing."""
        rankings = [
            {
                "model": "Model A",
                "average_rank": 1.5,
                "borda_score": 5,
                # No confidence_interval key
            }
        ]

        stage3 = {"model": "Model A", "response": "Response"}

        output = format_output(rankings, stage3, 1, 1)

        # Should handle missing CI gracefully
        assert "[0, 0]" in output or "[0.0, 0.0]" in output

    def test_missing_borda_score(self):
        """Test handling when borda_score is missing."""
        rankings = [
            {
                "model": "Model A",
                "average_rank": 1.5,
                "confidence_interval": (1.0, 2.0),
                # No borda_score key
            }
        ]

        stage3 = {"model": "Model A", "response": "Response"}

        output = format_output(rankings, stage3, 1, 1)

        # Should use 0 as default
        assert "| 1 | Model A | 1.5 | [1.0, 2.0] | 0 |" in output

    def test_markdown_structure(self):
        """Test that output has valid markdown structure."""
        rankings = [
            {"model": "Model A", "average_rank": 1.0, "confidence_interval": (0.9, 1.1), "borda_score": 10},
            {"model": "Model B", "average_rank": 2.0, "confidence_interval": (1.9, 2.1), "borda_score": 5},
        ]

        stage3 = {"model": "Chairman", "response": "Final answer here."}

        output = format_output(rankings, stage3, 2, 2)

        # Check markdown elements
        # Note: "## " is also found in "### " so count is 3 (1 H2 + 2 H3 headers)
        assert output.count("## ") == 3  # One H2 header + two H3 headers contain "## "
        assert output.count("### ") == 2  # Two H3 headers
        assert output.count("---") >= 1  # At least one horizontal rule
        assert output.count("**") >= 2  # Bold text for Chairman

        # Check table structure
        lines = output.split("\n")
        table_lines = [line for line in lines if line.startswith("|")]
        assert len(table_lines) >= 4  # Header, separator, and at least 2 data rows

    def test_very_long_response(self):
        """Test formatting with very long synthesized response."""
        rankings = [{"model": "Model A", "average_rank": 1.0, "confidence_interval": (1.0, 1.0), "borda_score": 3}]

        long_response = "This is a very long response. " * 500  # ~5000 chars

        stage3 = {"model": "Model A", "response": long_response}

        output = format_output(rankings, stage3, 1, 1)

        # Should include the full response
        assert long_response in output
        assert len(output) > 5000

    def test_multiline_response(self):
        """Test formatting with multiline synthesized response."""
        rankings = [{"model": "Model A", "average_rank": 1.0, "confidence_interval": (1.0, 1.0), "borda_score": 3}]

        multiline_response = """First paragraph of the response.

Second paragraph with more details.

- Bullet point 1
- Bullet point 2

Final paragraph."""

        stage3 = {"model": "Model A", "response": multiline_response}

        output = format_output(rankings, stage3, 1, 1)

        # Should preserve the multiline structure
        assert multiline_response in output
        assert "- Bullet point 1" in output
        assert "- Bullet point 2" in output

    def test_unicode_in_content(self):
        """Test formatting with Unicode characters."""
        rankings = [{"model": "Model 🤖", "average_rank": 1.0, "confidence_interval": (1.0, 1.0), "borda_score": 5}]

        stage3 = {"model": "Model 🤖", "response": "Response with émojis 🎉 and spëcial çhars"}

        output = format_output(rankings, stage3, 1, 1)

        # Unicode should be preserved
        assert "Model 🤖" in output
        assert "émojis 🎉" in output
        assert "spëcial çhars" in output

    def test_rank_numbering(self):
        """Test that ranks are numbered correctly starting from 1."""
        rankings = [
            {"model": f"Model {i}", "average_rank": i, "confidence_interval": (i, i), "borda_score": 10 - i}
            for i in range(1, 6)
        ]

        stage3 = {"model": "Model 1", "response": "Response"}

        output = format_output(rankings, stage3, 5, 5)

        # Check rank numbers
        for i in range(1, 6):
            assert f"| {i} | Model {i} |" in output

    def test_floating_point_precision(self):
        """Test that floating point values are displayed correctly."""
        rankings = [
            {
                "model": "Model A",
                "average_rank": 1.33333333,
                "confidence_interval": (1.11111, 1.55555),
                "borda_score": 7.5,
            }
        ]

        stage3 = {"model": "Model A", "response": "Response"}

        output = format_output(rankings, stage3, 1, 1)

        # Values should be displayed as-is (Python's default float formatting)
        assert "1.33333333" in output or "1.333" in output
        assert "1.11111" in output or "1.111" in output
        assert "7.5" in output

    def test_zero_total_ballots(self):
        """Test edge case with zero total ballots."""
        output = format_output([], {"model": "Model", "response": "Response"}, 0, 0)

        assert "0/0" in output
        # Should still produce valid output without crashing

    def test_more_valid_than_total(self):
        """Test edge case where valid_ballots > total_ballots (shouldn't happen but handle gracefully)."""
        rankings = [{"model": "Model A", "average_rank": 1.0, "confidence_interval": (1.0, 1.0), "borda_score": 3}]

        output = format_output(rankings, {"model": "Model A", "response": "Response"}, 5, 3)

        # Should display as given even if illogical
        assert "5/3" in output
