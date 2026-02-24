"""Tests for ranking parser functions."""

try:
    from llm_council.parsing import (
        _parse_inline_ranking,
        _parse_json_ranking,
        _parse_numbered_ranking,
        parse_ranking_from_text,
    )
except ImportError:
    from council import (
        _parse_inline_ranking,
        _parse_json_ranking,
        _parse_numbered_ranking,
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
