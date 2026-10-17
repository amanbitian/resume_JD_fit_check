import json

import pytest

from candidate_fit.cloud_batch import (
    OPENAI_MAX_REQUESTS,
    AnthropicBatchClient,
    BatchApiError,
    OpenAIBatchClient,
    estimate_cost,
    load_api_key,
    normalise_anthropic,
    normalise_openai,
)
from candidate_fit.batch_runner import chunk


class FakeTransport:
    """Records every call and answers from a queue of (status, body) per URL suffix."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append({"method": method, "url": url, "headers": headers, "body": body})
        for suffix, reply in self.answers:
            if url.endswith(suffix):
                status, data = reply if isinstance(reply, tuple) else (200, reply)
                return status, data if isinstance(data, bytes) else json.dumps(data).encode()
        raise AssertionError(f"unexpected call {method} {url}")


REQUESTS = [{"custom_id": "a1-0", "system": "sys", "user": "hello", "max_output_tokens": 500}]


def test_anthropic_submit_sends_the_documented_batch_shape_and_key_header():
    transport = FakeTransport([("/v1/messages/batches", {"id": "msgbatch_1"})])
    client = AnthropicBatchClient("claude-haiku-5-5", "KEY", transport=transport)
    assert client.submit(REQUESTS) == "msgbatch_1"
    call = transport.calls[0]
    body = json.loads(call["body"])
    assert body["requests"][0]["custom_id"] == "a1-0"
    assert body["requests"][0]["params"]["model"] == "claude-haiku-5-5"
    assert body["requests"][0]["params"]["max_tokens"] == 500
    assert body["requests"][0]["params"]["messages"] == [{"role": "user", "content": "hello"}]
    assert call["headers"]["x-api-key"] == "KEY" and call["headers"]["anthropic-version"] == "2023-06-01"


def test_anthropic_status_and_results_follow_results_url():
    transport = FakeTransport([
        ("/results", (200, b'{"custom_id":"a1-0","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"{}"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":4}}}}\n')),
        ("/v1/messages/batches/msgbatch_1", {"id": "msgbatch_1", "processing_status": "ended",
                                              "request_counts": {"succeeded": 1}, "results_url": "https://x/results"}),
    ])
    client = AnthropicBatchClient("m", "KEY", transport=transport)
    status = client.status("msgbatch_1")
    assert status.done and status.state == "ended"
    (reply,) = client.results("msgbatch_1")
    assert reply == {"custom_id": "a1-0", "ok": True, "text": "{}", "finish_reason": "stop", "input_tokens": 10,
                     "output_tokens": 4, "thinking_tokens": None, "error": None}


def test_anthropic_errors_and_token_limit_are_not_ok():
    errored = normalise_anthropic({"custom_id": "x", "result": {"type": "errored", "error": {"message": "bad"}}})
    assert errored["ok"] is False and "errored" in errored["error"]
    capped = normalise_anthropic({"custom_id": "y", "result": {"type": "succeeded", "message": {
        "content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens", "usage": {}}}})
    assert capped["ok"] is False and capped["finish_reason"] == "length"


def test_openai_submit_uploads_jsonl_then_creates_a_24h_batch():
    transport = FakeTransport([("/files", {"id": "file-1"}), ("/batches", {"id": "batch_1"})])
    client = OpenAIBatchClient("gpt-5.6-luna", "KEY", transport=transport)
    assert client.submit(REQUESTS) == "batch_1"
    upload, create = transport.calls
    assert b'name="purpose"' in upload["body"] and b"candidatefit" in upload["body"].split(b"\r\n")[0]
    line = json.loads(upload["body"].split(b"\r\n\r\n", 2)[2].split(b"\r\n--")[0])
    assert line["custom_id"] == "a1-0" and line["url"] == "/v1/chat/completions"
    assert line["body"]["max_completion_tokens"] == 500 and "temperature" not in line["body"]
    assert json.loads(create["body"]) == {"input_file_id": "file-1", "endpoint": "/v1/chat/completions",
                                          "completion_window": "24h"}


def test_openai_results_read_output_and_error_files_with_reasoning_tokens():
    output = {"custom_id": "a1-0", "response": {"status_code": 200, "body": {
        "choices": [{"message": {"content": "{\"input_status\": \"ok\"}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "completion_tokens_details": {"reasoning_tokens": 3}}}}}
    error = {"custom_id": "a1-1", "error": {"code": "invalid", "message": "bad request"}}
    transport = FakeTransport([
        ("/batches/batch_1", {"id": "batch_1", "status": "completed", "output_file_id": "out", "error_file_id": "err"}),
        ("/files/out/content", (200, (json.dumps(output) + "\n").encode())),
        ("/files/err/content", (200, (json.dumps(error) + "\n").encode())),
    ])
    replies = OpenAIBatchClient("m", "KEY", transport=transport).results("batch_1")
    assert replies[0]["ok"] is True and replies[0]["thinking_tokens"] == 3 and replies[0]["input_tokens"] == 12
    assert replies[1]["ok"] is False and "invalid" in replies[1]["error"]


def test_openai_status_done_only_for_terminal_states():
    transport = FakeTransport([("/batches/b", {"id": "b", "status": "in_progress"})])
    assert OpenAIBatchClient("m", "K", transport=transport).status("b").done is False


def test_http_error_raises_with_status():
    transport = FakeTransport([("/v1/messages/batches", (401, b'{"error":"unauthorized"}'))])
    with pytest.raises(BatchApiError) as caught:
        AnthropicBatchClient("m", "bad", transport=transport).submit(REQUESTS)
    assert caught.value.status == 401


def test_batches_split_on_request_count_and_bytes():
    class Small:
        max_requests = 2
        max_bytes = 10**9

    requests = [{"custom_id": str(i), "system": "s", "user": "u", "max_output_tokens": 1} for i in range(5)]
    assert [len(part) for part in chunk(requests, Small())] == [2, 2, 1]

    class Tiny:
        max_requests = OPENAI_MAX_REQUESTS
        max_bytes = 200

    assert len(chunk(requests, Tiny())) > 1


def test_cost_uses_half_the_standard_price_for_batch():
    cost = estimate_cost("claude-haiku-5-5", 1_000_000, 1_000_000)
    assert cost["known_price"] and cost["usd"] == pytest.approx((0.10 + 0.50) / 2, rel=1e-6)
    assert estimate_cost("unknown-model", 1, 1)["known_price"] is False


def test_missing_api_key_is_a_clear_error():
    with pytest.raises(BatchApiError) as caught:
        load_api_key("NO_SUCH_VARIABLE_FOR_TESTS", {})
    assert "NO_SUCH_VARIABLE_FOR_TESTS" in str(caught.value)


class SequenceTransport:
    """Answers each call from a list of (status, body) or exceptions, in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append(method)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_get_requests_are_retried_with_backoff_on_rate_limits_and_server_errors():
    sleeps = []
    transport = SequenceTransport([(429, b"slow down"), (503, b"busy"), (200, b'{"processing_status":"ended","request_counts":{}}')])
    client = AnthropicBatchClient("m", "K", transport=transport, sleep=sleeps.append)
    assert client.status("b1").done is True
    assert transport.calls == ["GET", "GET", "GET"] and sleeps == [2.0, 4.0]


def test_get_requests_are_retried_after_a_dropped_connection():
    transport = SequenceTransport([ConnectionResetError("reset"), (200, b'{"status":"in_progress"}')])
    client = OpenAIBatchClient("m", "K", transport=transport, sleep=lambda s: None)
    assert client.status("b1").done is False and len(transport.calls) == 2


def test_a_post_is_not_repeated_after_a_dropped_connection_because_it_may_have_been_created():
    transport = SequenceTransport([ConnectionResetError("reset"), (200, b'{"id":"batch_dup"}')])
    client = AnthropicBatchClient("m", "K", transport=transport, sleep=lambda s: None)
    with pytest.raises(BatchApiError, match="network error"):
        client.submit(REQUESTS)
    assert transport.calls == ["POST"]  # one attempt only


def test_a_post_is_retried_when_the_provider_says_it_is_overloaded():
    transport = SequenceTransport([(529, b"overloaded"), (200, b'{"id":"msgbatch_2"}')])
    client = AnthropicBatchClient("m", "K", transport=transport, sleep=lambda s: None)
    assert client.submit(REQUESTS) == "msgbatch_2" and transport.calls == ["POST", "POST"]


def test_retries_stop_after_five_tries_and_return_the_last_error():
    transport = SequenceTransport([(503, b"down")] * 5)
    client = OpenAIBatchClient("m", "K", transport=transport, sleep=lambda s: None)
    with pytest.raises(BatchApiError) as caught:
        client.status("b1")
    assert caught.value.status == 503 and len(transport.calls) == 5


def test_openai_finds_an_interrupted_batch_by_its_intent_metadata():
    listing = {"data": [{"id": "batch_a", "metadata": {"candidate_fit_intent": "other"}},
                        {"id": "batch_b", "metadata": {"candidate_fit_intent": "abc"}},
                        {"id": "batch_c", "metadata": {"candidate_fit_intent": "abc"}}]}
    transport = FakeTransport([("/batches?limit=100", listing)])
    client = OpenAIBatchClient("m", "K", transport=transport)
    assert client.find_submitted("abc", 0.0, 1, exclude={"batch_c"}) == ["batch_b"]


def test_openai_tags_each_batch_with_its_intent_so_it_can_be_found_later():
    transport = FakeTransport([("/files", {"id": "file-1"}), ("/batches", {"id": "batch_1"})])
    OpenAIBatchClient("m", "K", transport=transport).submit(REQUESTS, intent_id="abc")
    assert json.loads(transport.calls[1]["body"])["metadata"] == {"candidate_fit_intent": "abc"}


def test_anthropic_finds_an_interrupted_batch_by_creation_time_and_size():
    listing = {"data": [
        {"id": "old", "created_at": "2026-10-17T05:00:00Z", "request_counts": {"processing": 3}},
        {"id": "match", "created_at": "2026-10-17T10:00:30Z", "request_counts": {"processing": 2, "succeeded": 0}},
        {"id": "wrong_size", "created_at": "2026-10-17T10:00:40Z", "request_counts": {"processing": 9}},
    ]}
    transport = FakeTransport([("/v1/messages/batches?limit=100", listing)])
    client = AnthropicBatchClient("m", "K", transport=transport)
    since = 1792231200.0  # 2026-10-17T10:00:00Z
    assert client.find_submitted("abc", since, 2, exclude=set()) == ["match"]


def test_openai_reasoning_effort_is_sent_only_when_set_and_validated():
    transport = FakeTransport([("/files", {"id": "f"}), ("/batches", {"id": "b"})])
    OpenAIBatchClient("m", "K", transport=transport, reasoning_effort="low").submit(REQUESTS)
    line = json.loads(transport.calls[0]["body"].split(b"\r\n\r\n", 2)[2].split(b"\r\n--")[0])
    assert line["body"]["reasoning_effort"] == "low"
    with pytest.raises(ValueError):
        OpenAIBatchClient("m", "K", reasoning_effort="extreme")


def test_a_token_limit_mostly_used_by_hidden_reasoning_says_so():
    line = {"custom_id": "x", "response": {"status_code": 200, "body": {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4096}}}}}
    reply = normalise_openai(line)
    assert reply["ok"] is False and "hidden reasoning" in reply["error"] and reply["thinking_tokens"] == 4096


def test_an_anthropic_refusal_is_reported_as_a_refusal_not_an_empty_reply():
    reply = normalise_anthropic({"custom_id": "x", "result": {"type": "succeeded", "message": {
        "content": [], "stop_reason": "refusal", "usage": {"input_tokens": 5, "output_tokens": 0}}}})
    assert reply["ok"] is False and "refused" in reply["error"]
