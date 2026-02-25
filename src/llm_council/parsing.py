"""Ranking parser functions for extracting model rankings from text."""

from __future__ import annotations

import json
import re


def _parse_json_ranking(text: str, num_responses: int | None = None) -> list[str] | None:
    """Try to parse a JSON object/array from the text containing ranking.

    Args:
        text: Raw text that might contain JSON
        num_responses: Expected number of responses (unused, for consistency)

    Returns:
        List of ranking labels if successful, None otherwise
    """
    # Try JSON format with code blocks first
    json_match = re.search(r"```json\s*(\{[^`]*\})\s*```", text)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if "ranking" in data and isinstance(data["ranking"], list):
                # Validate that all items are "Response X" format
                ranking = data["ranking"]
                if all(isinstance(r, str) and re.match(r"^Response [A-Z]$", r) for r in ranking):
                    return ranking
        except json.JSONDecodeError:
            pass

    # Try inline JSON (without code blocks)
    inline_json_match = re.search(r'\{\s*"ranking"\s*:\s*\[([^\]]+)\]\s*\}', text)
    if inline_json_match:
        try:
            full_match = inline_json_match.group(0)
            data = json.loads(full_match)
            if "ranking" in data and isinstance(data["ranking"], list):
                ranking = data["ranking"]
                if all(isinstance(r, str) and re.match(r"^Response [A-Z]$", r) for r in ranking):
                    return ranking
        except json.JSONDecodeError:
            pass

    return None


def _parse_numbered_ranking(text: str, num_responses: int | None = None) -> list[str] | None:
    """Try to parse "1. Response A" style numbered lists.

    Args:
        text: Raw text that might contain numbered list
        num_responses: Expected number of responses (unused, for consistency)

    Returns:
        List of ranking labels if successful, None otherwise
    """
    # Look for "FINAL RANKING:" section (legacy format)
    if "FINAL RANKING:" in text:
        parts = text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format
            numbered_matches = re.findall(r"\d+\.\s*Response [A-Z]", ranking_section)
            if numbered_matches:
                return [re.search(r"Response [A-Z]", m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order from ranking section
            matches = re.findall(r"Response [A-Z]", ranking_section)
            if matches:
                return matches

    return None


def _parse_inline_ranking(text: str, num_responses: int | None = None) -> list[str] | None:
    """Try to parse inline comma-separated or any "Response X" mentions.

    Args:
        text: Raw text that might contain response labels
        num_responses: Expected number of responses (unused, for consistency)

    Returns:
        List of ranking labels if successful, None otherwise
    """
    # Last resort: try to find any "Response X" patterns in order
    matches = re.findall(r"Response [A-Z]", text)
    if matches:
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_matches: list[str] = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                unique_matches.append(m)
        return unique_matches

    return None


def parse_ranking_from_text(ranking_text: str, num_responses: int | None = None) -> tuple[list[str], bool]:
    """Parse the ranking from the model's response.

    Tries JSON format first, then falls back to text parsing.
    Validates that ranking contains exactly num_responses unique entries if specified.

    Returns:
        Tuple of (list of "Response X" strings from best to worst, success boolean)
    """
    # Try each parser in order of preference
    parsers = [(_parse_json_ranking, True), (_parse_numbered_ranking, True), (_parse_inline_ranking, False)]
    for parser, reliable in parsers:
        result = parser(ranking_text, num_responses)
        if result:
            # Validate if num_responses specified
            if num_responses is not None:
                # Check correct count
                if len(result) != num_responses:
                    continue  # Try next parser
                # Check no duplicates
                if len(set(result)) != len(result):
                    continue
                # Check all are valid labels
                valid_labels = {f"Response {chr(65 + i)}" for i in range(num_responses)}
                if not all(r in valid_labels for r in result):
                    continue
            return (result, reliable)

    return ([], False)
