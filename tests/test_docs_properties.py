"""Property-based tests for documentation content validation.

Feature: readme-docs-fix
"""

from pathlib import Path

import hypothesis.strategies as st
from hypothesis import given, settings

# Resolve doc file paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_README = _PROJECT_ROOT / "README.md"
_CLAUDE_MD = _PROJECT_ROOT / "CLAUDE.md"

# Read file contents once (they don't change during test runs)
_readme_content = _README.read_text().lower()
_claude_content = _CLAUDE_MD.read_text().lower()


# --- Strategies ---

# Phrases that would falsely claim the project lacks caching capability.
# These are declarative statements about the project's limitations, NOT
# instructional phrases like "run without caching" which describe how to
# disable an existing feature.
_false_caching_claims = st.sampled_from([
    "no built-in caching",
    "no built-in cache",
    "no native caching",
    "no native cache",
    "no integrated caching",
    "no integrated cache",
    "lacks caching",
    "lacks a cache",
    "lacks built-in caching",
    "lacks response caching",
    "missing caching",
    "missing cache support",
    "missing caching support",
    "does not have caching",
    "does not have a cache",
    "doesn't have caching",
    "doesn't have a cache",
    "does not include caching",
    "doesn't include caching",
    "does not support caching",
    "doesn't support caching",
    "has no caching",
    "has no cache",
    "has no built-in cache",
    "has no response cache",
    "lacking caching",
    "lacking a cache",
    "lacking cache support",
    "absent caching",
    "no caching mechanism",
    "no caching system",
    "no caching layer",
    "no cache layer",
    "no response cache",
    "no response caching",
    "caching is not supported",
    "caching is not available",
    "caching is not implemented",
    "cache is not available",
    "cache is not implemented",
    "no caching support",
    "no cache support",
])


# --- Property Test 1 ---


@settings(max_examples=150)
@given(phrase=_false_caching_claims)
def test_no_false_caching_limitation_claims(phrase: str):
    """Property 1: No false caching limitation claims.

    For any generated phrase that would falsely claim the project lacks caching,
    neither README.md nor CLAUDE.md should contain it (case-insensitive).

    **Validates: Requirements 1.1, 3.3**

    Tag: Feature: readme-docs-fix, Property 1: No false caching limitation claims
    """
    lowered = phrase.lower()
    assert lowered not in _readme_content, (
        f"README.md contains false caching claim: '{phrase}'"
    )
    assert lowered not in _claude_content, (
        f"CLAUDE.md contains false caching claim: '{phrase}'"
    )


# --- Strategies for Property Test 2 ---

# Key caching facts that must appear in both README.md and CLAUDE.md.
_key_caching_facts = st.sampled_from([
    "~/.llm-council/cache.db",
    "--no-cache",
    "enabled by default",
])


# --- Property Test 2 ---


@settings(max_examples=150)
@given(fact=_key_caching_facts)
def test_cross_document_caching_consistency(fact: str):
    """Property 2: Cross-document caching consistency.

    For each key caching fact (cache path, flag name, default behavior),
    both README.md and CLAUDE.md must contain it (case-insensitive).

    **Validates: Requirements 4.1, 4.2, 4.3**

    Tag: Feature: readme-docs-fix, Property 2: Cross-document caching consistency
    """
    lowered = fact.lower()
    assert lowered in _readme_content, (
        f"README.md is missing key caching fact: '{fact}'"
    )
    assert lowered in _claude_content, (
        f"CLAUDE.md is missing key caching fact: '{fact}'"
    )


# --- Strategies for Property Test 3 ---

# Required caching details that must appear in the README Caching section.
# Each tuple is (detail_description, search_term) where search_term is the
# text that must be present in the Caching section (case-insensitive).
_required_caching_details = st.sampled_from([
    ("SQLite-backed cache description", "sqlite"),
    ("default-enabled behavior", "enabled by default"),
    ("--no-cache flag", "--no-cache"),
    ("cache location ~/.llm-council/cache.db", "~/.llm-council/cache.db"),
    ("Stage 1 response keying", "stage 1"),
])

# Extract the Caching section from README.md (between "## Caching" and the next "##" heading)
import re as _re

_caching_section_match = _re.search(
    r"(?:^|\n)## Caching\n(.*?)(?=\n## |\Z)",
    _README.read_text(),
    _re.DOTALL,
)
_caching_section = _caching_section_match.group(0).lower() if _caching_section_match else ""


# --- Property Test 3 ---


@settings(max_examples=150)
@given(detail=_required_caching_details)
def test_caching_section_completeness(detail: tuple):
    """Property 3: Caching section contains all required information.

    For each required caching detail, verify the README Caching section
    documents it.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

    Tag: Feature: readme-docs-fix, Property 3: Caching section contains all required information
    """
    description, search_term = detail
    assert search_term.lower() in _caching_section, (
        f"README Caching section is missing required detail: {description} "
        f"(searched for '{search_term}')"
    )


# --- Strategies and helpers for Property Test 4 ---

# Extract the Known Limitations section from README.md
_limitations_section_match = _re.search(
    r"(?:^|\n)## Known Limitations\n(.*?)(?=\n## |\Z)",
    _README.read_text(),
    _re.DOTALL,
)
_limitations_section = (
    _limitations_section_match.group(1).strip() if _limitations_section_match else ""
)

# Parse individual limitation entries (lines starting with "- **...**")
_limitation_entries = _re.findall(
    r"-\s+\*\*(.+?)\*\*", _limitations_section
)

# Genuine current limitations that should remain in the section (lowercased).
_GENUINE_LIMITATIONS = {
    "no streaming output",
}

# The removed limitation that must NOT appear.
_REMOVED_LIMITATION = "no built-in caching"

# Strategy: sample from the genuine limitations set
_genuine_limitations_strategy = st.sampled_from(sorted(_GENUINE_LIMITATIONS))


# --- Property Test 4 ---


@settings(max_examples=150)
@given(limitation=_genuine_limitations_strategy)
def test_known_limitations_accuracy(limitation: str):
    """Property 4: Known Limitations accuracy.

    For all entries in the Known Limitations section, each must map to a
    genuine current limitation.  The only remaining limitation should be
    "No streaming output".  The removed "No built-in caching" entry must
    not be present.

    **Validates: Requirements 1.2, 1.3**

    Tag: Feature: readme-docs-fix, Property 4: Known Limitations accuracy
    """
    lowered_entries = [e.lower() for e in _limitation_entries]

    # Every genuine limitation should be listed
    assert limitation in lowered_entries, (
        f"Genuine limitation '{limitation}' is missing from Known Limitations section"
    )

    # No entry should be the removed false limitation
    assert _REMOVED_LIMITATION.lower() not in lowered_entries, (
        f"Removed limitation '{_REMOVED_LIMITATION}' is still present in Known Limitations"
    )

    # Every listed entry must be a genuine limitation
    for entry in lowered_entries:
        assert entry in _GENUINE_LIMITATIONS, (
            f"Unknown limitation '{entry}' found in Known Limitations section — "
            f"expected only: {_GENUINE_LIMITATIONS}"
        )
