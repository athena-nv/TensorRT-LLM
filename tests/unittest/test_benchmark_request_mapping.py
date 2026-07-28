import csv
import json
from types import SimpleNamespace

from tensorrt_llm.disaggregated_params import \
    DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.serve.openai_protocol import (
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DisaggregatedParams,
)
from tensorrt_llm.serve.postprocess_handlers import \
    completion_stream_post_processor
from tensorrt_llm.serve.scripts.backend_request_func import (
    RequestFuncOutput,
    _capture_request_identifiers,
)
from tensorrt_llm.serve.scripts.benchmark_serving import save_request_mapping


def test_streaming_postprocessor_exposes_ctx_request_id_on_first_chunk():
    output = SimpleNamespace(
        index=0,
        text_diff="token",
        token_ids_diff=[],
        finish_reason=None,
        stop_reason=None,
        disaggregated_params=LlmDisaggregatedParams(
            request_type="generation_only",
            ctx_request_id=12345,
        ),
    )
    response = SimpleNamespace(
        id="server-request-id",
        outputs=[output],
        cached_tokens=0,
        _done=False,
    )
    args = SimpleNamespace(
        num_prompt_tokens=1,
        ctx_usage=None,
        stream_response_id=None,
        stream_created=None,
        stream_options=None,
        echo=False,
        first_iteration=True,
        prompt="",
        prompt_idx=0,
        num_choices=1,
        detokenize=True,
        return_logprobs=False,
        model="test-model",
    )

    chunks = completion_stream_post_processor(response, args)
    payload = json.loads(chunks[0].removeprefix("data: ").strip())

    assert payload["choices"][0]["disaggregated_params"][
        "ctx_request_id"] == 12345


def test_streaming_response_carries_request_identifiers():
    response = CompletionStreamResponse(
        id="cmpl-client-id",
        model="test-model",
        choices=[
            CompletionResponseStreamChoice(
                index=0,
                text="token",
                disaggregated_params=DisaggregatedParams(
                    request_type="generation",
                    ctx_request_id=12345,
                ),
            )
        ],
    )
    output = RequestFuncOutput()

    _capture_request_identifiers(output, response.model_dump())

    assert output.request_id == "cmpl-client-id"
    assert output.ctx_request_id == 12345


def test_save_request_mapping_preserves_submission_order_and_failures(tmp_path):
    results = {
        "request_ids": ["cmpl-2", "cmpl-1", None],
        "ctx_request_ids": [202, 101, None],
        "ttfts": [2.5, 1.25, 0.0],
        "successes": [True, True, False],
    }

    mapping_path = save_request_mapping(
        str(tmp_path),
        "/logs/benchmark-run",
        2,
        results,
    )

    with open(mapping_path, newline="", encoding="utf-8") as mapping_file:
        rows = list(csv.DictReader(mapping_file))

    assert [row["client_request_index"] for row in rows] == ["0", "1", "2"]
    assert [row["benchmark_round"] for row in rows] == ["1", "1", "2"]
    assert [row["ctx_request_id"] for row in rows] == ["202", "101", ""]
    assert [row["client_ttft_ms"] for row in rows] == [
        "2500.0",
        "1250.0",
        "",
    ]
    assert [row["status"] for row in rows] == [
        "success",
        "success",
        "failed",
    ]
