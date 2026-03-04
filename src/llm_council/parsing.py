"""Ranking text parsers for extracting model rankings from Stage 2 output.

Exports ``parse_ranking_from_text`` which tries a cascade of parsers (JSON,
numbered list, ordinal words, comma-separated, inline) to extract ordered
``Response X`` labels from free-form model output. Also exports
``format_ranking`` for rendering a parsed ballot as a numbered list.
"""

from __future__ import annotations

import json
import re

from .models import generate_response_labels

ORDINALS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "twenty-first": 21,
    "twenty-second": 22,
    "twenty-third": 23,
    "twenty-fourth": 24,
    "twenty-fifth": 25,
    "twenty-sixth": 26,
}


def _parse_json_ranking(text: str, num_responses: int | None = None) -> list[str] | None:
    """Try to parse a JSON object/array from the text containing ranking.

    Args:
        text: Raw text that might contain JSON
        num_responses: Expected number of responses (unused, for consistency)

    Returns:
        List of ranking labels if successful, None otherwise
    """
    # Try JSON format with code blocks first (take LAST match to defeat injection)
    json_matches = list(re.finditer(r"```json\s*(\{[^`]*\})\s*```", text))
    json_match = json_matches[-1] if json_matches else None
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

    # Try inline JSON (without code blocks) — take LAST match to defeat injection
    inline_json_matches = list(re.finditer(r'\{\s*"ranking"\s*:\s*\[([^\]]+)\]\s*\}', text))
    inline_json_match = inline_json_matches[-1] if inline_json_matches else None
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


def _parse_headerless_numbered_ranking(text: str, num_responses: int | None = None) -> list[str] | None:
    """Parse numbered lists like '1. Response A' without a FINAL RANKING header.

    Returns labels in textual order of appearance (ignoring numeric values).
    Returns None if a FINAL RANKING header is present or no matches found.
    """
    if "FINAL RANKING:" in text:
        return None
    matches = re.findall(r"^\s*\d+[\.\)]\s*(Response [A-Z])", text, re.MULTILINE)
    return matches if matches else None


def _parse_ordinal_ranking(text: str, num_responses: int | None = None) -> list[str] | None:
    """Parse ordinal word rankings like 'first is Response A, second is Response B'.

    Maps ordinal words to positions and returns labels sorted by position.
    Returns None on duplicate positions or no matches.

    Pairing strategy: interleave ordinals and labels by text position, then
    greedily pair each ordinal with the nearest adjacent label (left or right)
    that hasn't been claimed yet, processing left-to-right.
    """
    # Collect all tokens: ('ord', offset, ordinal_position) or ('lbl', offset, label_str)
    tokens: list[tuple[str, int, int | str]] = []

    for word, pos in ORDINALS.items():
        for m in re.finditer(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
            tokens.append(("ord", m.start(), pos))

    for m in re.finditer(r"Response [A-Z]", text):
        tokens.append(("lbl", m.start(), m.group()))

    if not tokens:
        return None

    tokens.sort(key=lambda t: t[1])

    # Greedy left-to-right: for each ordinal, look at immediate left and right
    # neighbors in the sorted token list and pick the nearest unclaimed label.
    claimed: set[int] = set()  # indices of claimed label tokens
    position_to_label: dict[int, str] = {}

    for i, (kind, offset, val) in enumerate(tokens):
        if kind != "ord":
            continue
        # Search outward from position i for nearest unclaimed label
        left = i - 1
        right = i + 1
        best_idx = None
        best_dist = float("inf")
        while left >= 0 or right < len(tokens):
            if left >= 0:
                lk, lo, lv = tokens[left]
                if lk == "lbl" and left not in claimed:
                    d = offset - lo
                    if d < best_dist:
                        best_dist = d
                        best_idx = left
                    break  # found nearest left label
                left -= 1
            if right < len(tokens):
                rk, ro, rv = tokens[right]
                if rk == "lbl" and right not in claimed:
                    d = ro - offset
                    if d < best_dist:
                        best_dist = d
                        best_idx = right
                    break  # found nearest right label
                right += 1
        if best_idx is None:
            continue
        label = tokens[best_idx][2]
        pos = val
        if pos in position_to_label and position_to_label[pos] != label:
            return None
        position_to_label[pos] = label
        claimed.add(best_idx)

    if not position_to_label:
        return None
    return [label for _, label in sorted(position_to_label.items())]


def _parse_comma_separated_ranking(text: str, num_responses: int | None = None) -> list[str] | None:
    """Parse comma-separated rankings like 'Response B, Response A, Response C'.

    Handles trailing 'and' before the last item. Returns None on duplicates or no match.
    """
    for line in text.splitlines():
        # Split on commas and 'and'
        parts = re.split(r",\s*|\s+and\s+", line)
        labels = []
        for part in parts:
            m = re.search(r"Response [A-Z]", part.strip())
            if m:
                labels.append(m.group())
        if len(labels) >= 3:
            if len(labels) != len(set(labels)):
                return None  # duplicates
            return labels
    return None


def parse_ranking_from_text(ranking_text: str, num_responses: int | None = None) -> tuple[list[str], bool]:
    """Parse a model ranking from free-form text into an ordered ballot.

    Tries a cascade of parsers in decreasing order of structural reliability
    (JSON → numbered list → headerless numbered list → ordinal words →
    comma-separated → inline mentions) and returns the first valid ballot.
    Each parser call is wrapped in ``try/except`` so unexpected errors in
    any single parser fall through to the next.

    When *num_responses* is provided, the result is validated: the ballot
    must contain exactly that many unique labels, each matching
    ``Response A`` … ``Response <N>``.

    Args:
        ranking_text: Raw text output from a ranking model, potentially
            containing JSON, numbered lists, ordinal prose, or other
            free-form ranking formats.
        num_responses: Expected number of ``Response X`` labels in the
            ballot.  When ``None``, length and label validation is
            skipped.

    Returns:
        A ``(ranking, reliable)`` tuple where *ranking* is a list of
        ``"Response X"`` strings ordered from best to worst, and
        *reliable* is ``True`` when the ballot came from a structurally
        unambiguous parser (all parsers except the inline fallback).
        Returns ``([], False)`` when no parser produces a valid ballot.
    """
    parsers = [
        (_parse_json_ranking, True),
        (_parse_numbered_ranking, True),
        (_parse_headerless_numbered_ranking, True),
        (_parse_ordinal_ranking, True),
        (_parse_comma_separated_ranking, True),
        (_parse_inline_ranking, False),
    ]
    for parser, reliable in parsers:
        try:
            result = parser(ranking_text, num_responses)
        except Exception:
            continue
        if result:
            if num_responses is not None:
                if len(result) != num_responses:
                    continue
                if len(set(result)) != len(result):
                    continue
                valid_labels = set(generate_response_labels(num_responses))
                if not all(r in valid_labels for r in result):
                    continue
            return (result, reliable)

    return ([], False)


def format_ranking(ranking: list[str]) -> str:
    """Format a parsed ballot as a human-readable numbered list.

    Converts an ordered list of ``"Response X"`` labels into a
    newline-separated numbered list suitable for display or logging
    (e.g. ``"1. Response A\\n2. Response B"``).

    Args:
        ranking: Ordered list of ``"Response X"`` label strings, from
            best (index 0) to worst.

    Returns:
        A newline-separated numbered list string.  Returns an empty
        string when *ranking* is empty.
    """
    if not ranking:
        return ""
    return "\n".join(f"{i + 1}. {label}" for i, label in enumerate(ranking))
