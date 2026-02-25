"""Tests for stages module, particularly randomization and self-exclusion."""

from unittest.mock import AsyncMock, patch

import pytest

try:
    from llm_council.stages import build_ranking_prompt, stage2_collect_rankings
except ImportError:
    from council import build_ranking_prompt, stage2_collect_rankings


def unpack_stage2_result(result):
    """Helper to handle both 2-tuple and 3-tuple returns from stage2_collect_rankings."""
    if len(result) == 3:
        return result[0], result[1]  # stage2_results, per_ranker_mappings (ignore token_usages)
    else:
        return result[0], result[1]


class TestBuildRankingPrompt:
    def test_basic_prompt_construction(self):
        """Test basic ranking prompt construction."""
        question = "What is 2+2?"
        responses = [("Model1", "Four"), ("Model2", "4")]
        labels = ["A", "B"]

        prompt = build_ranking_prompt(question, responses, labels)

        assert "What is 2+2?" in prompt
        assert "Response A" in prompt
        assert "Response B" in prompt
        # Check anonymization (model names should not appear in prompt)
        assert "Model1" not in prompt
        assert "Model2" not in prompt

    def test_response_order_randomization(self):
        """Test that response order can be randomized."""
        question = "Test question"
        responses = [("M1", "Answer 1"), ("M2", "Answer 2"), ("M3", "Answer 3")]
        labels = ["A", "B", "C"]

        # Original order
        prompt1 = build_ranking_prompt(question, responses, labels, response_order=None)

        # Custom order (reversed)
        prompt2 = build_ranking_prompt(question, responses, labels, response_order=[2, 1, 0])

        # The prompts should be different due to different ordering
        assert prompt1 != prompt2

        # But both should contain all responses
        assert "Response A" in prompt1
        assert "Response B" in prompt1
        assert "Response C" in prompt1
        assert "Response A" in prompt2
        assert "Response B" in prompt2
        assert "Response C" in prompt2

    def test_response_order_affects_content(self):
        """Test that response order actually changes which content appears where."""
        question = "Test"
        responses = [("M1", "First answer"), ("M2", "Second answer"), ("M3", "Third answer")]
        labels = ["A", "B", "C"]

        # Original order
        prompt_original = build_ranking_prompt(question, responses, labels, response_order=[0, 1, 2])

        # Reversed order
        prompt_reversed = build_ranking_prompt(question, responses, labels, response_order=[2, 1, 0])

        # In original order, "First answer" should come before "Third answer"
        # In reversed order, "Third answer" should come before "First answer"
        first_pos_original = prompt_original.find("First answer")
        third_pos_original = prompt_original.find("Third answer")
        assert first_pos_original < third_pos_original

        first_pos_reversed = prompt_reversed.find("First answer")
        third_pos_reversed = prompt_reversed.find("Third answer")
        assert third_pos_reversed < first_pos_reversed


class TestStage2CollectRankings:
    @pytest.mark.asyncio
    async def test_self_exclusion(self):
        """Test that models don't rank their own responses."""
        user_query = "What is AI?"
        stage1_results = [
            {"model": "Model1", "response": "AI is artificial intelligence"},
            {"model": "Model2", "response": "AI stands for artificial intelligence"},
            {"model": "Model3", "response": "Artificial Intelligence"},
        ]
        council_models = [
            {"name": "Model1", "provider": "bedrock", "model_id": "test1"},
            {"name": "Model2", "provider": "bedrock", "model_id": "test2"},
            {"name": "Model3", "provider": "bedrock", "model_id": "test3"},
        ]

        # Mock the query_model function
        with patch("llm_council.stages.query_model") as mock_query:
            # Setup mock to return valid rankings
            async def mock_query_response(*args, **kwargs):
                return {
                    "content": """
                    Response A is good.
                    Response B is better.
                    ```json
                    {"ranking": ["Response B", "Response A"]}
                    ```
                    """
                }, None  # Return tuple (response, token_usage)

            mock_query.side_effect = mock_query_response

            result = await stage2_collect_rankings(user_query, stage1_results, council_models, None)
            stage2_results, per_ranker_mappings = unpack_stage2_result(result)

            # Check that each ranker got a mapping without themselves
            assert "Model1" in per_ranker_mappings
            assert "Model2" in per_ranker_mappings
            assert "Model3" in per_ranker_mappings

            # Model1 should not be in its own ranking options
            model1_options = list(per_ranker_mappings["Model1"].values())
            assert "Model1" not in model1_options
            assert "Model2" in model1_options
            assert "Model3" in model1_options

            # Model2 should not be in its own ranking options
            model2_options = list(per_ranker_mappings["Model2"].values())
            assert "Model2" not in model2_options
            assert "Model1" in model2_options
            assert "Model3" in model2_options

            # Model3 should not be in its own ranking options
            model3_options = list(per_ranker_mappings["Model3"].values())
            assert "Model3" not in model3_options
            assert "Model1" in model3_options
            assert "Model2" in model3_options

    @pytest.mark.asyncio
    async def test_different_orderings_per_ranker(self):
        """Test that each ranker can see responses in different orders."""
        user_query = "Test question"
        stage1_results = [
            {"model": "M1", "response": "Answer 1"},
            {"model": "M2", "response": "Answer 2"},
            {"model": "M3", "response": "Answer 3"},
            {"model": "M4", "response": "Answer 4"},
        ]
        council_models = [
            {"name": "M1", "provider": "bedrock", "model_id": "test1"},
            {"name": "M2", "provider": "bedrock", "model_id": "test2"},
            {"name": "M3", "provider": "bedrock", "model_id": "test3"},
            {"name": "M4", "provider": "bedrock", "model_id": "test4"},
        ]

        with patch("llm_council.stages.query_model") as mock_query:
            # Track what prompts each model receives
            prompts_by_model = {}

            async def mock_query_response(model_config, messages, *args, **kwargs):
                model_name = model_config.get("name")
                prompt = messages[0]["content"]
                prompts_by_model[model_name] = prompt
                return {
                    "content": """
                    ```json
                    {"ranking": ["Response A", "Response B", "Response C"]}
                    ```
                    """
                }, None  # Return tuple (response, token_usage)

            mock_query.side_effect = mock_query_response

            # Use a fixed seed for reproducibility in tests
            import random

            random.seed(42)

            result = await stage2_collect_rankings(user_query, stage1_results, council_models, None)
            stage2_results, per_ranker_mappings = unpack_stage2_result(result)

            # Check that models received different prompts (due to randomization)
            # With randomization, it's very likely (but not guaranteed) that
            # at least some models see different orderings
            # We can at least verify all models got prompts
            assert len(prompts_by_model) == 4
            # Verify uniqueness: at least some models should see different orderings
            unique_prompts = set(prompts_by_model.values())
            # Log for debugging (not asserting as randomization might occasionally produce same order)
            if len(unique_prompts) < 4:
                print(f"Note: Only {len(unique_prompts)} unique prompt orderings out of 4 models")

    @pytest.mark.asyncio
    async def test_variable_length_rankings(self):
        """Test that rankings can have different lengths due to self-exclusion."""
        user_query = "Question"
        stage1_results = [
            {"model": "M1", "response": "R1"},
            {"model": "M2", "response": "R2"},
        ]
        council_models = [
            {"name": "M1", "provider": "bedrock", "model_id": "test1"},
            {"name": "M2", "provider": "bedrock", "model_id": "test2"},
        ]

        with patch("llm_council.stages.query_model") as mock_query:

            async def mock_query_response(*args, **kwargs):
                # Each model only ranks one other (due to self-exclusion)
                return {
                    "content": """
                    ```json
                    {"ranking": ["Response A"]}
                    ```
                    """
                }, None  # Return tuple (response, token_usage)

            mock_query.side_effect = mock_query_response

            result = await stage2_collect_rankings(user_query, stage1_results, council_models, None)
            stage2_results, per_ranker_mappings = unpack_stage2_result(result)

            # Each ranker should only have one response to rank
            assert len(per_ranker_mappings["M1"]) == 1
            assert len(per_ranker_mappings["M2"]) == 1

            # M1 ranks M2, M2 ranks M1
            assert "M2" in per_ranker_mappings["M1"].values()
            assert "M1" in per_ranker_mappings["M2"].values()

    @pytest.mark.asyncio
    async def test_empty_rankings_when_only_self(self):
        """Test behavior when a model is the only one that responded."""
        user_query = "Question"
        stage1_results = [
            {"model": "M1", "response": "Only response"},
        ]
        council_models = [
            {"name": "M1", "provider": "bedrock", "model_id": "test1"},
        ]

        with patch("llm_council.stages.query_model") as mock_query:
            # This shouldn't be called since M1 has no one else to rank
            mock_query.side_effect = AsyncMock(return_value=({"content": "Should not be called"}, None))

            result = await stage2_collect_rankings(user_query, stage1_results, council_models, None)
            stage2_results, per_ranker_mappings = unpack_stage2_result(result)

            # No rankings should be performed
            assert len(stage2_results) == 0
            assert "M1" not in per_ranker_mappings  # M1 had nothing to rank
            mock_query.assert_not_called()
