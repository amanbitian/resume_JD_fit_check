import pytest

from candidate_fit.client import DualGPUClient, LocalLLMClient


def test_completion_limit_is_not_silently_accepted(monkeypatch):
    client = LocalLLMClient(retries=0)
    monkeypatch.setattr(
        client,
        "_post",
        lambda _body: {
            "choices": [{"message": {"content": '{"fit_score": 5}'}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 4096},
        },
    )
    try:
        client.complete("test")
    except RuntimeError as exc:
        assert "token limit" in str(exc)
    else:
        raise AssertionError("A token-limited response must be rejected")


def test_direct_and_gateway_urls():
    assert LocalLLMClient(base_url="http://127.0.0.1:8081").completions_url == (
        "http://127.0.0.1:8081/v1/chat/completions"
    )
    assert LocalLLMClient(base_url="http://127.0.0.1:8090/v1/9070xt").completions_url == (
        "http://127.0.0.1:8090/v1/9070xt/chat/completions"
    )
    assert LocalLLMClient(base_url="http://127.0.0.1:8090/v1/r9700/evaluator").completions_url == (
        "http://127.0.0.1:8090/v1/r9700/evaluator/chat/completions"
    )


def test_provider_options_are_sent_and_fingerprinted(monkeypatch):
    captured = {}
    client = LocalLLMClient(
        retries=0,
        api_key="explicit-key",
        client_id="local-registration-id",
        extra_body={"enable_thinking": False},
    )

    def fake_post(body):
        captured.update(body)
        return {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {},
        }

    monkeypatch.setattr(client, "_post", fake_post)
    client.complete("test", response_schema=None)

    assert client.api_key == "explicit-key"
    assert captured["enable_thinking"] is False
    assert client.generation_config["extra_body"] == {"enable_thinking": False}



def test_dual_gpu_client_accepts_only_loopback():
    for url in ("http://127.0.0.1:8090", "http://localhost:8090/"):
        assert DualGPUClient(orchestrator_url=url, lane="r9700", model="m", client_id="c").base_url.startswith("http://")
    for url in ("http://192.168.1.20:8090", "http://example.com:8090", "https://127.0.0.1.evil.test"):
        with pytest.raises(ValueError, match="non-loopback"):
            DualGPUClient(orchestrator_url=url, lane="r9700", model="m", client_id="c")
    remote = DualGPUClient(orchestrator_url="http://10.0.0.5:8090", lane="r9700", model="m", client_id="c", allow_remote=True)
    assert remote.base_url == "http://10.0.0.5:8090"


def _dual_client(**kwargs):
    return DualGPUClient(orchestrator_url="http://127.0.0.1:8090", lane="r9700", model="m", client_id="c", **kwargs)


def _capture_body(client):
    sent = {}

    def fake_post(body):
        sent.update(body)
        return {"results": {"r9700": {"status": "completed", "output_text": "{}", "finish_reason": "stop"}}}

    client._post = fake_post
    client.context_window = 1
    return sent


def test_dual_gpu_request_carries_attribution_fields():
    client = _dual_client(task_label="fit", connection_type="batch")
    sent = _capture_body(client)
    client.complete("u")
    assert sent["task_label"] == "fit"
    assert sent["connection_type"] == "batch"
    assert sent["store_history"] is True
    assert sent["client_id"] == "c" and sent["workload_kind"] == "evaluation"


def test_dual_gpu_stage_label_and_history_opt_out():
    client = _dual_client(task_label="fit", store_history=False)
    sent = _capture_body(client)
    client.complete("u", stage="jd_extraction")
    assert sent["task_label"] == "fit.jd_extraction"
    assert sent["store_history"] is False
