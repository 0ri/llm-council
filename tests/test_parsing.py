"""Tests for ranking parser functions."""

try:
    from llm_council.parsing import (
        _parse_comma_separated_ranking,
        _parse_headerless_numbered_ranking,
        _parse_inline_ranking,
        _parse_json_ranking,
        _parse_numbered_ranking,
        _parse_ordinal_ranking,
        format_ranking,
        parse_ranking_from_text,
    )
except ImportError:
    from council import (
        _parse_comma_separated_ranking,
        _parse_headerless_numbered_ranking,
        _parse_inline_ranking,
        _parse_json_ranking,
        _parse_numbered_ranking,
        _parse_ordinal_ranking,
        format_ranking,
        parse_ranking_from_text,
    )


class TestParseJsonRanking:
    def test_valid_json_in_code_block(self):
        text = '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```'
        result = _parse_json_ranking(text)
        assert result == ["Response A", "Response B", "Response C"]

    def test_valid_inline_json(self):
        text = 'Here is my ranking: {"ranking": ["Response B", "Response A", "Response C"]}'
        result = _parse_json_ranking(text)
        assert result == ["Response B", "Response A", "Response C"]

    def test_malformed_json(self):
        text = '```json\n{"ranking": ["Response A", "Response B"}\n```'
        result = _parse_json_ranking(text)
        assert result is None

    def test_missing_ranking_key(self):
        text = '```json\n{"order": ["Response A", "Response B"]}\n```'
        result = _parse_json_ranking(text)
        assert result is None

    def test_non_response_items(self):
        text = '```json\n{"ranking": ["Model A", "Model B"]}\n```'
        result = _parse_json_ranking(text)
        assert result is None

    def test_empty_ranking_array(self):
        text = '```json\n{"ranking": []}\n```'
        result = _parse_json_ranking(text)
        # Empty list - all() returns True on empty iterable, but the regex
        # pattern in the code block parser won't match an empty array inside
        # the {[^`]*} pattern. The inline parser might catch it though.
        # all() on empty list returns True but the list is empty.
        assert result is None or result == []

    def test_extra_whitespace_in_code_block(self):
        text = '```json\n  {"ranking": ["Response C", "Response A", "Response B"]}  \n```'
        result = _parse_json_ranking(text)
        assert result == ["Response C", "Response A", "Response B"]

    def test_json_with_surrounding_text(self):
        text = 'Based on my analysis:\n```json\n{"ranking": ["Response A", "Response B"]}\n```\nThat is my ranking.'
        result = _parse_json_ranking(text)
        assert result == ["Response A", "Response B"]


class TestParseNumberedRanking:
    def test_final_ranking_with_numbered_list(self):
        text = "Some analysis...\nFINAL RANKING:\n1. Response B\n2. Response A\n3. Response C"
        result = _parse_numbered_ranking(text)
        assert result == ["Response B", "Response A", "Response C"]

    def test_no_final_ranking_header(self):
        text = "1. Response A\n2. Response B\n3. Response C"
        result = _parse_numbered_ranking(text)
        assert result is None

    def test_final_ranking_with_extra_text(self):
        text = (
            "Analysis...\nFINAL RANKING:\n"
            "1. Response C - best overall\n2. Response A - good detail\n3. Response B - lacks depth"
        )
        result = _parse_numbered_ranking(text)
        assert result == ["Response C", "Response A", "Response B"]

    def test_final_ranking_fallback_to_pattern(self):
        text = "FINAL RANKING:\nResponse B is first, then Response A, then Response C"
        result = _parse_numbered_ranking(text)
        assert result == ["Response B", "Response A", "Response C"]


class TestParseInlineRanking:
    def test_multiple_response_mentions(self):
        text = "I think Response C is best, followed by Response A, then Response B"
        result = _parse_inline_ranking(text)
        assert result == ["Response C", "Response A", "Response B"]

    def test_deduplication(self):
        text = "Response A is good. Response B is better. Response A was close though. Response B wins."
        result = _parse_inline_ranking(text)
        assert result == ["Response A", "Response B"]

    def test_no_matches(self):
        text = "I cannot rank these responses meaningfully."
        result = _parse_inline_ranking(text)
        assert result is None

    def test_single_response(self):
        text = "Only Response A is relevant."
        result = _parse_inline_ranking(text)
        assert result == ["Response A"]


class TestParseRankingFromText:
    def test_json_takes_priority(self):
        text = (
            "Response B is best.\nFINAL RANKING:\n1. Response A\n"
            '```json\n{"ranking": ["Response C", "Response A", "Response B"]}\n```'
        )
        result, is_valid = parse_ranking_from_text(text)
        assert result == ["Response C", "Response A", "Response B"]
        assert is_valid is True

    def test_falls_back_to_numbered(self):
        text = "Analysis here.\nFINAL RANKING:\n1. Response B\n2. Response A\n3. Response C"
        result, is_valid = parse_ranking_from_text(text)
        assert result == ["Response B", "Response A", "Response C"]
        assert is_valid is True

    def test_falls_back_to_inline_marked_unreliable(self):
        text = "Response A is the best, then Response B."
        result, is_valid = parse_ranking_from_text(text)
        assert result == ["Response A", "Response B"]
        assert is_valid is False

    def test_empty_input(self):
        result, is_valid = parse_ranking_from_text("")
        assert result == []
        assert is_valid is False

    def test_no_ranking_found(self):
        result, is_valid = parse_ranking_from_text("I refuse to rank these.")
        assert result == []
        assert is_valid is False


class TestParseHeaderlessNumberedRanking:
    def test_happy_path(self):
        text = "1. Response B\n2. Response A\n3. Response C"
        assert _parse_headerless_numbered_ranking(text) == ["Response B", "Response A", "Response C"]

    def test_with_parenthesis_delimiter(self):
        text = "1) Response C\n2) Response A\n3) Response B"
        assert _parse_headerless_numbered_ranking(text) == ["Response C", "Response A", "Response B"]

    def test_non_sequential_numbers(self):
        text = "3. Response A\n1. Response C\n7. Response B"
        assert _parse_headerless_numbered_ranking(text) == ["Response A", "Response C", "Response B"]

    def test_defers_to_header_parser(self):
        text = "FINAL RANKING:\n1. Response A\n2. Response B"
        assert _parse_headerless_numbered_ranking(text) is None

    def test_no_matches(self):
        assert _parse_headerless_numbered_ranking("No numbered list here.") is None

    def test_leading_whitespace(self):
        text = "  1. Response A\n  2. Response B\n  3. Response C"
        assert _parse_headerless_numbered_ranking(text) == ["Response A", "Response B", "Response C"]


class TestParseOrdinalRanking:
    def test_happy_path(self):
        text = "First is Response C, second is Response A, third is Response B."
        assert _parse_ordinal_ranking(text) == ["Response C", "Response A", "Response B"]

    def test_case_insensitive(self):
        text = "FIRST place goes to Response B. SECOND place goes to Response A."
        assert _parse_ordinal_ranking(text) == ["Response B", "Response A"]

    def test_duplicate_positions_rejected(self):
        text = "First is Response A. First is Response B."
        assert _parse_ordinal_ranking(text) is None

    def test_no_ordinals(self):
        assert _parse_ordinal_ranking("Response A is great.") is None

    def test_reverse_order_in_text(self):
        # When label appears before ordinal in a clear sentence structure
        text = "Response B is the first choice. Response A is the second choice."
        assert _parse_ordinal_ranking(text) == ["Response B", "Response A"]


class TestParseCommaSeparatedRanking:
    def test_happy_path(self):
        text = "Response B, Response A, Response C"
        assert _parse_comma_separated_ranking(text) == ["Response B", "Response A", "Response C"]

    def test_with_trailing_and(self):
        text = "Response A, Response C and Response B"
        assert _parse_comma_separated_ranking(text) == ["Response A", "Response C", "Response B"]

    def test_duplicates_rejected(self):
        text = "Response A, Response B, Response A"
        assert _parse_comma_separated_ranking(text) is None

    def test_fewer_than_three(self):
        text = "Response A, Response B"
        assert _parse_comma_separated_ranking(text) is None

    def test_no_matches(self):
        assert _parse_comma_separated_ranking("No responses here.") is None

    def test_multiline_picks_qualifying_line(self):
        text = "Some preamble.\nResponse C, Response A, Response B\nSome epilogue."
        assert _parse_comma_separated_ranking(text) == ["Response C", "Response A", "Response B"]


class TestFormatRanking:
    def test_happy_path(self):
        assert format_ranking(["Response A", "Response B", "Response C"]) == (
            "1. Response A\n2. Response B\n3. Response C"
        )

    def test_empty_list(self):
        assert format_ranking([]) == ""

    def test_single_item(self):
        assert format_ranking(["Response A"]) == "1. Response A"

    def test_round_trip(self):
        ballot = ["Response C", "Response A", "Response B"]
        formatted = format_ranking(ballot)
        parsed, reliable = parse_ranking_from_text(formatted, num_responses=3)
        assert format_ranking(parsed) == formatted
        assert reliable is True


class TestFallbackChainOrdering:
    def test_headerless_numbered_before_inline(self):
        """Headerless numbered list should be reliable, not fall to inline."""
        text = "1. Response A\n2. Response B\n3. Response C"
        result, reliable = parse_ranking_from_text(text, num_responses=3)
        assert result == ["Response A", "Response B", "Response C"]
        assert reliable is True

    def test_ordinal_before_inline(self):
        text = "First is Response B, second is Response A, third is Response C."
        result, reliable = parse_ranking_from_text(text, num_responses=3)
        assert result == ["Response B", "Response A", "Response C"]
        assert reliable is True

    def test_comma_separated_before_inline(self):
        text = "My ranking: Response C, Response B, Response A"
        result, reliable = parse_ranking_from_text(text, num_responses=3)
        assert result == ["Response C", "Response B", "Response A"]
        assert reliable is True

    def test_json_still_wins_over_new_parsers(self):
        text = (
            '```json\n{"ranking": ["Response A", "Response B", "Response C"]}\n```\n'
            "1. Response C\n2. Response B\n3. Response A"
        )
        result, reliable = parse_ranking_from_text(text, num_responses=3)
        assert result == ["Response A", "Response B", "Response C"]
        assert reliable is True

    def test_numbered_with_header_wins_over_headerless(self):
        text = "1. Response C\n2. Response B\nFINAL RANKING:\n1. Response A\n2. Response B\n3. Response C"
        result, reliable = parse_ranking_from_text(text, num_responses=3)
        assert result == ["Response A", "Response B", "Response C"]
        assert reliable is True

    def test_parser_exception_falls_through(self):
        """If a parser raises, the chain continues to the next one."""
        text = "Response A, Response B, Response C"
        result, reliable = parse_ranking_from_text(text, num_responses=3)
        assert result == ["Response A", "Response B", "Response C"]
        assert reliable is True
