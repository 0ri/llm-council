"""Property-based tests for streaming output.

Feature: streaming-output
"""

from __future__ import annotations

import asyncio

import hypothesis.strategies as st
from hypothesis import given, settings

from llm_council.providers import StreamResult, fallback_astream

# --- Property Test 2: Fallback astream yields single chunk equal to query ---
# Feature: streaming-output, Property 2: Fallback astream yields single chunk equal to query


class _MockProvider:
    """A non-streaming Provider that returns a fixed response from query()."""

    def __init__(self, response: str):
        self._response = response

    async def query(self, prompt: str, model_config: dict, timeout: int) -> tuple[str, dict | None]:
        return self._response, None


@settings(max_examples=100)
@given(response=st.text(min_size=0))
def test_fallback_astream_yields_single_chunk_equal_to_query(response: str):
    """Property 2: Fallback astream yields single chunk equal to query.

    For any provider that does not implement StreamingProvider natively,
    calling fallback_astream should yield exactly one chunk whose text
    equals the full response from query, and the accumulated text on
    StreamResult should equal the query result.

    **Validates: Requirements 1.4**

    Tag: Feature: streaming-output, Property 2: Fallback astream yields single chunk equal to query
    """

    async def _run():
        provider = _MockProvider(response)
        result = fallback_astream(provider, "test prompt", {}, timeout=30)

        assert isinstance(result, StreamResult)

        chunks: list[str] = []
        async for chunk in result:
            chunks.append(chunk)

        # Exactly one chunk is yielded
        assert len(chunks) == 1, f"Expected exactly 1 chunk, got {len(chunks)}"

        # The single chunk equals the query result
        assert chunks[0] == response, f"Chunk text does not match query result: {chunks[0]!r} != {response!r}"

        # Accumulated text on StreamResult equals the query result
        assert result.accumulated == response, (
            f"Accumulated text does not match: {result.accumulated!r} != {response!r}"
        )

    asyncio.run(_run())


# --- Property Test 3: All yielded chunks are non-empty strings ---
# Feature: streaming-output, Property 3: All yielded chunks are non-empty strings


class _MockStreamingProvider:
    """A StreamingProvider that yields pre-defined chunks via astream()."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    async def query(self, prompt: str, model_config: dict, timeout: int) -> tuple[str, dict | None]:
        return "".join(self._chunks), None

    def astream(self, prompt: str, model_config: dict, timeout: int) -> StreamResult:
        async def _generate():
            for chunk in self._chunks:
                yield chunk

        return StreamResult(_generate())


@settings(max_examples=100)
@given(chunks=st.lists(st.text(min_size=1), min_size=1))
def test_all_yielded_chunks_are_non_empty_strings(chunks: list[str]):
    """Property 3: All yielded chunks are non-empty strings.

    For any streaming provider and any valid prompt, every Token_Chunk
    yielded by astream should be a str instance with length > 0. The
    async iterator should yield at least one chunk for any successful
    response.

    **Validates: Requirements 1.1**

    Tag: Feature: streaming-output, Property 3: All yielded chunks are non-empty strings
    """

    async def _run():
        provider = _MockStreamingProvider(chunks)
        result = provider.astream("test prompt", {}, timeout=30)

        assert isinstance(result, StreamResult)

        yielded: list[str] = []
        async for chunk in result:
            # Every chunk must be a str
            assert isinstance(chunk, str), f"Chunk is not a str: {type(chunk)}"
            # Every chunk must be non-empty
            assert len(chunk) > 0, f"Chunk is empty: {chunk!r}"
            yielded.append(chunk)

        # At least one chunk must be yielded for a successful response
        assert len(yielded) >= 1, "Expected at least one chunk to be yielded"

        # Accumulated text should equal the concatenation of all chunks
        assert result.accumulated == "".join(chunks), (
            f"Accumulated text mismatch: {result.accumulated!r} != {''.join(chunks)!r}"
        )

    asyncio.run(_run())


# --- Property Test 1: Streaming round-trip equivalence ---
# Feature: streaming-output, Property 1: Streaming round-trip equivalence


def _split_into_chunks(text: str, split_points: list[int]) -> list[str]:
    """Split *text* at the given sorted indices, returning non-empty parts."""
    points = sorted(set(p for p in split_points if 0 < p < len(text)))
    parts: list[str] = []
    prev = 0
    for p in points:
        parts.append(text[prev:p])
        prev = p
    parts.append(text[prev:])
    return [p for p in parts if p]  # filter any empty artefacts


class _MockRoundTripProvider:
    """A StreamingProvider whose astream yields pre-defined chunks and whose
    query returns the full concatenated text.  Both expose identical usage."""

    def __init__(self, chunks: list[str], usage: dict[str, int] | None):
        self._chunks = chunks
        self._usage = usage

    async def query(self, prompt: str, model_config: dict, timeout: int) -> tuple[str, dict[str, int] | None]:
        return "".join(self._chunks), self._usage

    def astream(self, prompt: str, model_config: dict, timeout: int) -> StreamResult:
        usage = self._usage

        async def _generate():
            for chunk in self._chunks:
                yield chunk

        result = StreamResult(_generate())
        result.usage = usage
        return result


# Strategy: random split points within the string length
_usage_strategy = st.one_of(
    st.none(),
    st.fixed_dictionaries(
        {
            "input_tokens": st.integers(min_value=0, max_value=100_000),
            "output_tokens": st.integers(min_value=0, max_value=100_000),
        }
    ),
)


@settings(max_examples=100)
@given(
    response=st.text(min_size=1),
    split_points=st.lists(st.integers(min_value=0, max_value=500), max_size=20),
    usage=_usage_strategy,
)
def test_streaming_round_trip_equivalence(
    response: str,
    split_points: list[int],
    usage: dict[str, int] | None,
):
    """Property 1: Streaming round-trip equivalence.

    For any streaming provider and any valid prompt/model_config,
    concatenating all Token_Chunk strings yielded by astream should
    produce a string identical to the text returned by query for the
    same input.  Additionally, the token usage metadata from astream
    should be equivalent to that from query.

    **Validates: Requirements 1.2, 4.1, 6.3**

    Tag: Feature: streaming-output, Property 1: Streaming round-trip equivalence
    """

    # Split the response into random non-empty chunks
    chunks = _split_into_chunks(response, split_points)
    # Guarantee at least one chunk (response has min_size=1)
    if not chunks:
        chunks = [response]

    async def _run():
        provider = _MockRoundTripProvider(chunks, usage)

        # --- astream path ---
        stream_result = provider.astream("prompt", {}, timeout=30)
        assert isinstance(stream_result, StreamResult)

        yielded: list[str] = []
        async for chunk in stream_result:
            yielded.append(chunk)

        concatenated = "".join(yielded)

        # --- query path ---
        query_text, query_usage = await provider.query("prompt", {}, timeout=30)

        # 1. Concatenation of all chunks equals the original string
        assert concatenated == response, f"Concatenated chunks != original: {concatenated!r} != {response!r}"

        # 2. query returns the same text
        assert query_text == response, f"query text != original: {query_text!r} != {response!r}"

        # 3. StreamResult.accumulated equals the original string
        assert stream_result.accumulated == response, (
            f"accumulated != original: {stream_result.accumulated!r} != {response!r}"
        )

        # 4. Usage metadata matches between astream and query
        assert stream_result.usage == query_usage, (
            f"Usage mismatch: stream={stream_result.usage!r} vs query={query_usage!r}"
        )

    asyncio.run(_run())


# --- Property Test 6: Callback receives all chunks in order ---
# Feature: streaming-output, Property 6: Callback receives all chunks in order

from llm_council.context import CouncilContext  # noqa: E402
from llm_council.stages import stream_model  # noqa: E402


class _MockStreamingProviderForCallback:
    """A StreamingProvider that yields pre-defined chunks via astream()."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    async def query(self, prompt: str, model_config: dict, timeout: int) -> tuple[str, dict | None]:
        return "".join(self._chunks), None

    def astream(self, prompt: str, model_config: dict, timeout: int) -> StreamResult:
        async def _generate():
            for chunk in self._chunks:
                yield chunk

        result = StreamResult(_generate())
        return result


@settings(max_examples=100)
@given(chunks=st.lists(st.text(min_size=1), min_size=1))
def test_callback_receives_all_chunks_in_order(chunks: list[str]):
    """Property 6: Callback receives all chunks in order.

    For any streaming Stage 3 execution with an on_chunk callback, the
    callback should be invoked once per Token_Chunk in the order they are
    yielded, and the concatenation of all callback arguments should equal
    the full chairman response text.

    **Validates: Requirement 9.3**

    Tag: Feature: streaming-output, Property 6: Callback receives all chunks in order
    """

    async def _run():
        # Record all chunks received by the callback
        received_chunks: list[str] = []

        async def recording_callback(chunk: str) -> None:
            received_chunks.append(chunk)

        # Set up a mock streaming provider
        mock_provider = _MockStreamingProviderForCallback(chunks)

        # Build a minimal CouncilContext with the mock provider pre-registered
        ctx = CouncilContext()
        ctx.providers["bedrock"] = mock_provider
        ctx.semaphore = asyncio.Semaphore(4)

        # Model config pointing to our mock provider
        model_config = {"provider": "bedrock", "name": "test-model"}
        messages = [{"role": "user", "content": "test"}]

        # Call stream_model with the recording callback
        accumulated_text, _usage = await stream_model(model_config, messages, ctx, on_chunk=recording_callback)

        # 1. Callback was invoked once per chunk, in order
        assert len(received_chunks) == len(chunks), (
            f"Expected {len(chunks)} callback invocations, got {len(received_chunks)}"
        )
        for i, (received, expected) in enumerate(zip(received_chunks, chunks, strict=False)):
            assert received == expected, f"Chunk {i} mismatch: received {received!r} != expected {expected!r}"

        # 2. Concatenation of all callback arguments equals the full response
        full_from_callback = "".join(received_chunks)
        expected_full = "".join(chunks)
        assert full_from_callback == expected_full, (
            f"Callback concatenation mismatch: {full_from_callback!r} != {expected_full!r}"
        )

        # 3. The accumulated text returned by stream_model matches too
        assert accumulated_text == expected_full, (
            f"Accumulated text mismatch: {accumulated_text!r} != {expected_full!r}"
        )

    asyncio.run(_run())


# --- Property Test 4: Stage 1 output ordering is invariant to completion order ---
# Feature: streaming-output, Property 4: Stage 1 output ordering is invariant to completion order

from unittest.mock import AsyncMock, patch  # noqa: E402

from llm_council.models import Stage1Result  # noqa: E402
from llm_council.stages import stage1_collect_responses  # noqa: E402


def _unique_model_names():
    """Strategy that generates a list of 2-8 unique model names."""
    return st.lists(
        st.from_regex(r"[a-z][a-z0-9\-]{1,15}", fullmatch=True),
        min_size=2,
        max_size=8,
        unique=True,
    )


@settings(max_examples=100)
@given(
    model_names=_unique_model_names(),
    data=st.data(),
)
def test_stage1_output_ordering_invariant_to_completion_order(
    model_names: list[str],
    data: st.DataObject,
):
    """Property 4: Stage 1 output ordering is invariant to completion order.

    For any set of council models and any permutation of their completion
    order, the final Stage 1 results list should list models in the original
    config order, not completion order.

    **Validates: Requirements 5.3**

    Tag: Feature: streaming-output, Property 4: Stage 1 output ordering is invariant to completion order
    """

    # Generate a random permutation of the model names (simulating completion order)
    completion_order = data.draw(st.permutations(model_names))

    # Generate a random response for each model
    responses = {name: f"response-from-{name}" for name in model_names}

    async def _run():
        # Build council model configs in the original order
        council_models = [{"provider": "bedrock", "name": name, "model_id": f"id-{name}"} for name in model_names]

        # Build the mock return value for query_models_parallel.
        # The dict is constructed in completion order to simulate models
        # finishing in a different order than config order.
        response_dict: dict[str, dict[str, str] | None] = {}
        usage_dict: dict[str, dict | None] = {}
        for name in completion_order:
            response_dict[name] = {"content": responses[name]}
            usage_dict[name] = None

        mock_qmp = AsyncMock(return_value=(response_dict, usage_dict))

        ctx = CouncilContext()
        ctx.providers["bedrock"] = _MockProvider("")
        ctx.semaphore = asyncio.Semaphore(4)
        # Disable cache so all models go through query_models_parallel
        ctx.cache = None

        with patch("llm_council.stages.query_models_parallel", mock_qmp):
            stage1_results, _usages = await stage1_collect_responses("test question", council_models, ctx)

        # The results must be in the original config order
        assert len(stage1_results) == len(model_names), (
            f"Expected {len(model_names)} results, got {len(stage1_results)}"
        )

        for i, (result, expected_name) in enumerate(zip(stage1_results, model_names, strict=False)):
            assert isinstance(result, Stage1Result)
            assert result.model == expected_name, (
                f"Result {i}: expected model {expected_name!r}, got {result.model!r}. "
                f"Config order: {model_names}, completion order: {list(completion_order)}"
            )
            assert result.response == responses[expected_name], f"Result {i}: response mismatch for {expected_name}"

    asyncio.run(_run())


# --- Property Test 5: Output equivalence regardless of streaming mode ---
# Feature: streaming-output, Property 5: Output equivalence regardless of streaming mode

from llm_council.models import AggregateRanking, Stage2Result, Stage3Result  # noqa: E402
from llm_council.stages import stage3_synthesize_final  # noqa: E402


class _MockDualModeProvider:
    """A provider that supports both query() and astream(), returning the same text.

    Used to verify that stage3_synthesize_final produces identical output
    regardless of whether streaming is enabled or disabled.
    """

    def __init__(self, response_text: str, chunks: list[str]):
        self._response_text = response_text
        self._chunks = chunks

    async def query(self, prompt: str, model_config: dict, timeout: int) -> tuple[str, dict | None]:
        return self._response_text, None

    def astream(self, prompt: str, model_config: dict, timeout: int) -> StreamResult:
        async def _generate():
            for chunk in self._chunks:
                yield chunk

        result = StreamResult(_generate())
        return result


@settings(max_examples=100)
@given(
    response=st.text(min_size=1, max_size=200),
    split_points=st.lists(st.integers(min_value=0, max_value=200), max_size=10),
    question=st.text(min_size=1, max_size=100),
    model_response_1=st.text(min_size=1, max_size=100),
    model_response_2=st.text(min_size=1, max_size=100),
)
def test_output_equivalence_regardless_of_streaming_mode(
    response: str,
    split_points: list[int],
    question: str,
    model_response_1: str,
    model_response_2: str,
):
    """Property 5: Output equivalence regardless of streaming mode.

    For any question and valid config, stage3_synthesize_final(stream=False)
    and stage3_synthesize_final(stream=True) should return identical final
    Stage3Result.response strings.

    **Validates: Requirements 5.4, 9.4, 9.5**

    Tag: Feature: streaming-output, Property 5: Output equivalence regardless of streaming mode
    """

    # Split the response into random non-empty chunks for the streaming path
    chunks = _split_into_chunks(response, split_points)
    if not chunks:
        chunks = [response]

    async def _run():
        # Create a mock provider that returns the same text via both query and astream
        mock_provider = _MockDualModeProvider(response, chunks)

        # Build mock Stage1Results
        stage1_results = [
            Stage1Result(model="model-a", response=model_response_1),
            Stage1Result(model="model-b", response=model_response_2),
        ]

        # Build mock Stage2Results
        stage2_results = [
            Stage2Result(
                model="model-a",
                ranking="Response A > Response B",
                parsed_ranking=["Response A", "Response B"],
                is_valid_ballot=True,
            ),
            Stage2Result(
                model="model-b",
                ranking="Response B > Response A",
                parsed_ranking=["Response B", "Response A"],
                is_valid_ballot=True,
            ),
        ]

        label_to_model = {"Response A": "model-a", "Response B": "model-b"}

        aggregate_rankings = [
            AggregateRanking(model="model-a", average_rank=1.5, rankings_count=2),
            AggregateRanking(model="model-b", average_rank=1.5, rankings_count=2),
        ]

        chairman_config = {"provider": "bedrock", "name": "chairman-model"}

        # Set up CouncilContext with the mock provider
        ctx_non_stream = CouncilContext()
        ctx_non_stream.providers["bedrock"] = mock_provider
        ctx_non_stream.semaphore = asyncio.Semaphore(4)
        ctx_non_stream.progress = None  # disable progress tracking

        ctx_stream = CouncilContext()
        ctx_stream.providers["bedrock"] = mock_provider
        ctx_stream.semaphore = asyncio.Semaphore(4)
        ctx_stream.progress = None  # disable progress tracking

        # Run with stream=False
        result_no_stream, _usage1 = await stage3_synthesize_final(
            user_query=question,
            stage1_results=stage1_results,
            stage2_results=stage2_results,
            label_to_model=label_to_model,
            aggregate_rankings=aggregate_rankings,
            chairman_config=chairman_config,
            ctx=ctx_non_stream,
            stream=False,
        )

        # Run with stream=True
        result_stream, _usage2 = await stage3_synthesize_final(
            user_query=question,
            stage1_results=stage1_results,
            stage2_results=stage2_results,
            label_to_model=label_to_model,
            aggregate_rankings=aggregate_rankings,
            chairman_config=chairman_config,
            ctx=ctx_stream,
            stream=True,
        )

        # Both should be Stage3Result instances
        assert isinstance(result_no_stream, Stage3Result)
        assert isinstance(result_stream, Stage3Result)

        # The response strings must be identical
        assert result_no_stream.response == result_stream.response, (
            f"Output mismatch: stream=False produced {result_no_stream.response!r} "
            f"but stream=True produced {result_stream.response!r}"
        )

    asyncio.run(_run())


# --- Property Test 7: Stage 1 model responses are displayed immediately on completion ---
# Feature: streaming-output, Property 7: Stage 1 model responses are displayed immediately on completion

from llm_council.progress import ModelStatus, ProgressManager  # noqa: E402


def _model_names_strategy():
    """Strategy that generates a list of 2-6 unique model names."""
    return st.lists(
        st.from_regex(r"[a-z][a-z0-9\-]{1,12}", fullmatch=True),
        min_size=2,
        max_size=6,
        unique=True,
    )


class _MockProviderForStage1:
    """A provider that returns a fixed response for each model."""

    def __init__(self, responses: dict[str, str]):
        self._responses = responses

    async def query(self, prompt: str, model_config: dict, timeout: int) -> tuple[str, dict | None]:
        name = model_config.get("name", "unknown")
        return self._responses.get(name, "default-response"), None


@settings(max_examples=100)
@given(
    model_names=_model_names_strategy(),
)
def test_stage1_model_responses_displayed_immediately_on_completion(
    model_names: list[str],
):
    """Property 7: Stage 1 model responses are displayed immediately on completion.

    For any council run with multiple models, when a model completes its
    Stage 1 response, the ProgressManager's update_model is called with
    ModelStatus.DONE for that model. Each model's completion is reported
    individually.

    **Validates: Requirements 5.1**

    Tag: Feature: streaming-output, Property 7: Stage 1 model responses are displayed immediately on completion
    """

    responses = {name: f"response-from-{name}" for name in model_names}

    async def _run():
        # Build council model configs
        council_models = [{"provider": "bedrock", "name": name, "model_id": f"id-{name}"} for name in model_names]

        # Create a mock provider that returns responses
        mock_provider = _MockProviderForStage1(responses)

        # Create a real ProgressManager and spy on update_model
        progress = ProgressManager(is_tty=False)
        original_update_model = progress.update_model

        # Track all update_model calls with their arguments
        update_calls: list[tuple[str, ModelStatus]] = []

        async def tracking_update_model(
            model: str,
            status: ModelStatus,
            elapsed: float | None = None,
            response_text: str | None = None,
        ):
            update_calls.append((model, status))
            await original_update_model(model, status, elapsed, response_text)

        progress.update_model = tracking_update_model  # type: ignore[assignment]

        # Set up CouncilContext
        ctx = CouncilContext()
        ctx.providers["bedrock"] = mock_provider
        ctx.semaphore = asyncio.Semaphore(4)
        ctx.cache = None
        ctx.progress = progress

        # Start stage so ProgressManager is initialized
        await progress.start_stage(1, "Collecting responses", model_names)

        # Run query_models_parallel (used by Stage 1)
        from llm_council.stages import query_models_parallel

        messages = [{"role": "user", "content": "test question"}]
        result_dict, _usages = await query_models_parallel(council_models, messages, ctx)

        # Verify: update_model was called with DONE for each model
        done_calls = [(name, status) for name, status in update_calls if status == ModelStatus.DONE]

        done_model_names = [name for name, _ in done_calls]

        # Every model that returned a result should have a DONE call
        for name in model_names:
            if result_dict.get(name) is not None:
                assert name in done_model_names, (
                    f"Model {name!r} completed successfully but update_model "
                    f"was never called with ModelStatus.DONE. "
                    f"DONE calls: {done_model_names}"
                )

        # Each DONE call should correspond to a model in our config
        for name in done_model_names:
            assert name in model_names, f"update_model(DONE) called for unknown model {name!r}"

    asyncio.run(_run())
