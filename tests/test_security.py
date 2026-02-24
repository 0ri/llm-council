"""Tests for security helper functions."""

try:
    from llm_council.security import (
        build_manipulation_resistance_msg,
        format_anonymized_responses,
        wrap_untrusted_content,
    )
except ImportError:
    from council import (
        build_manipulation_resistance_msg,
        format_anonymized_responses,
        wrap_untrusted_content,
    )


class TestWrapUntrustedContent:
    def test_basic_wrapping(self):
        result = wrap_untrusted_content("test content", "Response A", "abc123")
        assert "test content" in result
        assert "Response A" in result
        assert "abc123" in result

    def test_content_with_backticks(self):
        content = "Here is code:\n```python\nprint('hello')\n```"
        result = wrap_untrusted_content(content, "Response A", "nonce42")
        assert content in result

    def test_nonce_in_xml_tags(self):
        result = wrap_untrusted_content("payload", "Response B", "deadbeef")
        assert "<response-deadbeef" in result
        assert "</response-deadbeef>" in result


class TestBuildManipulationResistanceMsg:
    def test_contains_critical_instructions(self):
        msg = build_manipulation_resistance_msg()
        assert "UNTRUSTED" in msg
        assert "NEVER" in msg

    def test_mentions_evaluation(self):
        msg = build_manipulation_resistance_msg()
        assert "evaluat" in msg.lower()


class TestFormatAnonymizedResponses:
    def test_basic_formatting(self):
        responses = [("Model-A", "Answer A"), ("Model-B", "Answer B")]
        result = format_anonymized_responses(responses)
        assert "Response A" in result
        assert "Response B" in result
        assert "Answer A" in result
        assert "Answer B" in result
        assert "Model-A" not in result
        assert "Model-B" not in result

    def test_custom_labels(self):
        responses = [("Model-A", "Answer A")]
        result = format_anonymized_responses(responses, labels=["Custom Label"])
        assert "Custom Label" in result
        assert "Answer A" in result

    def test_three_responses(self):
        responses = [("M1", "A1"), ("M2", "A2"), ("M3", "A3")]
        result = format_anonymized_responses(responses)
        assert "Response A" in result
        assert "Response B" in result
        assert "Response C" in result
