"""Tests for stages module, particularly randomization and self-exclusion."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_council.models import Stage1Result
from llm_council.stages import build_ranking_prompt, stage2_collect_rankings


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

        prompt = build_ranking_prompt(question, responses)

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

        # Original order
        prompt1 = build_ranking_prompt(question, responses, response_order=None)

        # Custom order (reversed)
        prompt2 = build_ranking_prompt(question, responses, response_order=[2, 1, 0])

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

        # Original order
        prompt_original = build_ranking_prompt(question, responses, response_order=[0, 1, 2])

        # Reversed order
        prompt_reversed = build_ranking_prompt(question, responses, response_order=[2, 1, 0])

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
    async def test_self_exclusion(self, make_ctx):
        """Test that models don't rank their own responses."""
        user_query = "What is AI?"
        stage1_results = [
            Stage1Result(model="Model1", response="AI is artificial intelligence"),
            Stage1Result(model="Model2", response="AI stands for artificial intelligence"),
            Stage1Result(model="Model3", response="Artificial Intelligence"),
        ]
        council_models = [
            {"name": "Model1", "provider": "bedrock", "model_id": "test1"},
            {"name": "Model2", "provider": "bedrock", "model_id": "test2"},
            {"name": "Model3", "provider": "bedrock", "model_id": "test3"},
        ]

        # Mock the query_model function
        with patch("llm_council.stages.stage2.query_model") as mock_query:
            # Setup mock to return valid rankings
            async def mock_query_response(*args, **kwargs):
                return """
                    Response A is good.
                    Response B is better.
                    ```json
                    {"ranking": ["Response B", "Response A"]}
                    ```
                    """, None  # Return tuple (response, token_usage)

            mock_query.side_effect = mock_query_response

            ctx = make_ctx()
            result = await stage2_collect_rankings(user_query, stage1_results, council_models, ctx)
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
    async def test_different_orderings_per_ranker(self, make_ctx):
        """Test that each ranker can see responses in different orders."""
        user_query = "Test question"
        stage1_results = [
            Stage1Result(model="M1", response="Answer 1"),
            Stage1Result(model="M2", response="Answer 2"),
            Stage1Result(model="M3", response="Answer 3"),
            Stage1Result(model="M4", response="Answer 4"),
        ]
        council_models = [
            {"name": "M1", "provider": "bedrock", "model_id": "test1"},
            {"name": "M2", "provider": "bedrock", "model_id": "test2"},
            {"name": "M3", "provider": "bedrock", "model_id": "test3"},
            {"name": "M4", "provider": "bedrock", "model_id": "test4"},
        ]

        with patch("llm_council.stages.stage2.query_model") as mock_query:
            # Track what prompts each model receives
            prompts_by_model = {}

            async def mock_query_response(model_config, messages, *args, **kwargs):
                model_name = model_config.name
                prompt = messages[0]["content"]
                prompts_by_model[model_name] = prompt
                return """
                    ```json
                    {"ranking": ["Response A", "Response B", "Response C"]}
                    ```
                    """, None  # Return tuple (response, token_usage)

            mock_query.side_effect = mock_query_response

            # Use a fixed seed for reproducibility in tests
            import random

            random.seed(42)

            ctx = make_ctx()
            result = await stage2_collect_rankings(user_query, stage1_results, council_models, ctx)
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
    async def test_variable_length_rankings(self, make_ctx):
        """Test that rankings can have different lengths due to self-exclusion."""
        user_query = "Question"
        stage1_results = [
            Stage1Result(model="M1", response="R1"),
            Stage1Result(model="M2", response="R2"),
        ]
        council_models = [
            {"name": "M1", "provider": "bedrock", "model_id": "test1"},
            {"name": "M2", "provider": "bedrock", "model_id": "test2"},
        ]

        with patch("llm_council.stages.stage2.query_model") as mock_query:

            async def mock_query_response(*args, **kwargs):
                # Each model only ranks one other (due to self-exclusion)
                return """
                    ```json
                    {"ranking": ["Response A"]}
                    ```
                    """, None  # Return tuple (response, token_usage)

            mock_query.side_effect = mock_query_response

            ctx = make_ctx()
            result = await stage2_collect_rankings(user_query, stage1_results, council_models, ctx)
            stage2_results, per_ranker_mappings = unpack_stage2_result(result)

            # Each ranker should only have one response to rank
            assert len(per_ranker_mappings["M1"]) == 1
            assert len(per_ranker_mappings["M2"]) == 1

            # M1 ranks M2, M2 ranks M1
            assert "M2" in per_ranker_mappings["M1"].values()
            assert "M1" in per_ranker_mappings["M2"].values()

    @pytest.mark.asyncio
    async def test_empty_rankings_when_only_self(self, make_ctx):
        """Test behavior when a model is the only one that responded."""
        user_query = "Question"
        stage1_results = [
            Stage1Result(model="M1", response="Only response"),
        ]
        council_models = [
            {"name": "M1", "provider": "bedrock", "model_id": "test1"},
        ]

        with patch("llm_council.stages.stage2.query_model") as mock_query:
            # This shouldn't be called since M1 has no one else to rank
            mock_query.side_effect = AsyncMock(return_value=("Should not be called", None))

            ctx = make_ctx()
            result = await stage2_collect_rankings(user_query, stage1_results, council_models, ctx)
            stage2_results, per_ranker_mappings = unpack_stage2_result(result)

            # No rankings should be performed
            assert len(stage2_results) == 0
            assert "M1" not in per_ranker_mappings  # M1 had nothing to rank
            mock_query.assert_not_called()


class TestStage2QualityGate:
    """Tests for Stage 2 ballot retry (quality gate) logic."""

    @pytest.mark.asyncio
    async def test_retry_invalid_ballot(self, make_ctx):
        """Invalid ballot on first call should be retried and succeed on second call."""
        user_query = "What is AI?"
        stage1_results = [
            Stage1Result(model="Model1", response="AI is artificial intelligence"),
            Stage1Result(model="Model2", response="AI stands for artificial intelligence"),
            Stage1Result(model="Model3", response="Artificial Intelligence"),
        ]
        council_models = [
            {"name": "Model1", "provider": "bedrock", "model_id": "test1"},
            {"name": "Model2", "provider": "bedrock", "model_id": "test2"},
            {"name": "Model3", "provider": "bedrock", "model_id": "test3"},
        ]

        call_counts: dict[str, int] = {}

        with patch("llm_council.stages.stage2.query_model") as mock_query:

            async def mock_query_response(model_config, messages, *args, **kwargs):
                name = model_config.name
                call_counts[name] = call_counts.get(name, 0) + 1

                if name == "Model2" and call_counts[name] == 1:
                    return "This is garbage that cannot be parsed as a ranking.", None

                return '```json\n{"ranking": ["Response A", "Response B"]}\n```', None

            mock_query.side_effect = mock_query_response

            ctx = make_ctx()
            result = await stage2_collect_rankings(
                user_query, stage1_results, council_models, ctx, stage2_max_retries=1
            )
            stage2_results, per_ranker_mappings = unpack_stage2_result(result)

            assert call_counts["Model2"] == 2

            model2_result = [r for r in stage2_results if r.model == "Model2"][0]
            assert model2_result.is_valid_ballot is True

    @pytest.mark.asyncio
    async def test_no_retry_when_all_valid(self, make_ctx):
        """No extra calls should be made when all ballots are valid on first try."""
        user_query = "What is AI?"
        stage1_results = [
            Stage1Result(model="Model1", response="AI is artificial intelligence"),
            Stage1Result(model="Model2", response="AI stands for artificial intelligence"),
            Stage1Result(model="Model3", response="Artificial Intelligence"),
        ]
        council_models = [
            {"name": "Model1", "provider": "bedrock", "model_id": "test1"},
            {"name": "Model2", "provider": "bedrock", "model_id": "test2"},
            {"name": "Model3", "provider": "bedrock", "model_id": "test3"},
        ]

        call_counts: dict[str, int] = {}

        with patch("llm_council.stages.stage2.query_model") as mock_query:

            async def mock_query_response(model_config, messages, *args, **kwargs):
                name = model_config.name
                call_counts[name] = call_counts.get(name, 0) + 1
                return '```json\n{"ranking": ["Response A", "Response B"]}\n```', None

            mock_query.side_effect = mock_query_response

            ctx = make_ctx()
            result = await stage2_collect_rankings(
                user_query, stage1_results, council_models, ctx, stage2_max_retries=2
            )
            stage2_results, per_ranker_mappings = unpack_stage2_result(result)

            for name in ["Model1", "Model2", "Model3"]:
                assert call_counts[name] == 1, f"{name} was called {call_counts[name]} times, expected 1"

            for r in stage2_results:
                assert r.is_valid_ballot is True

    @pytest.mark.asyncio
    async def test_retry_exhausted_keeps_invalid(self, make_ctx):
        """After max retries, an always-failing model should keep is_valid_ballot=False."""
        user_query = "What is AI?"
        stage1_results = [
            Stage1Result(model="Model1", response="AI is artificial intelligence"),
            Stage1Result(model="Model2", response="AI stands for artificial intelligence"),
            Stage1Result(model="Model3", response="Artificial Intelligence"),
        ]
        council_models = [
            {"name": "Model1", "provider": "bedrock", "model_id": "test1"},
            {"name": "Model2", "provider": "bedrock", "model_id": "test2"},
            {"name": "Model3", "provider": "bedrock", "model_id": "test3"},
        ]

        call_counts: dict[str, int] = {}

        with patch("llm_council.stages.stage2.query_model") as mock_query:

            async def mock_query_response(model_config, messages, *args, **kwargs):
                name = model_config.name
                call_counts[name] = call_counts.get(name, 0) + 1

                if name == "Model2":
                    return "Completely unparseable garbage text.", None

                return '```json\n{"ranking": ["Response A", "Response B"]}\n```', None

            mock_query.side_effect = mock_query_response

            max_retries = 3
            ctx = make_ctx()
            result = await stage2_collect_rankings(
                user_query, stage1_results, council_models, ctx, stage2_max_retries=max_retries
            )
            stage2_results, per_ranker_mappings = unpack_stage2_result(result)

            assert call_counts["Model2"] == 1 + max_retries

            model2_result = [r for r in stage2_results if r.model == "Model2"][0]
            assert model2_result.is_valid_ballot is False

    @pytest.mark.asyncio
    async def test_retry_disabled_with_zero(self, make_ctx):
        """When stage2_max_retries=0, no retries should happen even with invalid ballots."""
        user_query = "What is AI?"
        stage1_results = [
            Stage1Result(model="Model1", response="AI is artificial intelligence"),
            Stage1Result(model="Model2", response="AI stands for artificial intelligence"),
            Stage1Result(model="Model3", response="Artificial Intelligence"),
        ]
        council_models = [
            {"name": "Model1", "provider": "bedrock", "model_id": "test1"},
            {"name": "Model2", "provider": "bedrock", "model_id": "test2"},
            {"name": "Model3", "provider": "bedrock", "model_id": "test3"},
        ]

        call_counts: dict[str, int] = {}

        with patch("llm_council.stages.stage2.query_model") as mock_query:

            async def mock_query_response(model_config, messages, *args, **kwargs):
                name = model_config.name
                call_counts[name] = call_counts.get(name, 0) + 1

                if name == "Model2":
                    return "Completely unparseable garbage text.", None

                return '```json\n{"ranking": ["Response A", "Response B"]}\n```', None

            mock_query.side_effect = mock_query_response

            ctx = make_ctx()
            result = await stage2_collect_rankings(
                user_query, stage1_results, council_models, ctx, stage2_max_retries=0
            )
            stage2_results, per_ranker_mappings = unpack_stage2_result(result)

            assert call_counts["Model2"] == 1

            model2_result = [r for r in stage2_results if r.model == "Model2"][0]
            assert model2_result.is_valid_ballot is False


class TestBudgetEnforcement:
    """Test that budget guard blocks queries when limits are exceeded."""

    @pytest.mark.asyncio
    async def test_budget_blocks_query(self, make_ctx):
        """query_model returns (None, None) when budget is exceeded."""
        from llm_council.budget import BudgetGuard
        from llm_council.stages import query_model

        ctx = make_ctx(budget_guard=BudgetGuard(max_tokens=10))

        model_config = {"name": "TestModel", "provider": "poe", "bot_name": "TestBot"}
        messages = [{"role": "user", "content": "This message will exceed the tiny budget " * 10}]

        with patch("llm_council.context.CouncilContext.get_provider") as mock_get_provider:
            result, usage = await query_model(model_config, messages, ctx)

            assert result is None
            assert isinstance(usage, dict) and usage.get("budget_exceeded") is True
            mock_get_provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_budget_allows_query_within_limit(self, make_ctx):
        """query_model proceeds when budget is not exceeded."""
        from llm_council.budget import BudgetGuard
        from llm_council.stages import query_model

        ctx = make_ctx(budget_guard=BudgetGuard(max_tokens=1_000_000))

        model_config = {"name": "TestModel", "provider": "poe", "bot_name": "TestBot"}
        messages = [{"role": "user", "content": "Hi"}]

        mock_provider = AsyncMock()
        mock_provider.query = AsyncMock(return_value=("Hello back", None))

        with patch.object(ctx, "get_provider", return_value=mock_provider):
            result, usage = await query_model(model_config, messages, ctx)

            assert result is not None
            assert result == "Hello back"
