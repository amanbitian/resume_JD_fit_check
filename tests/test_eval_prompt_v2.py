import json
from pathlib import Path
import sys

import pytest

from candidate_fit.client import Completion, OutputTokenLimitError
from candidate_fit.dataset import PairRecord

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_eval_prompt_v2 as pv2  # noqa: E402

KIT = pv2.load_kit()
JD = "We need a Python developer with 5 years of relevant experience. Knowledge of AWS or Azure is preferred."
RESUME = "Software Engineer at Acme, 2019-2024. Built Python services."


def assessment(**changes):
    value = {
        "status": "scored",
        "checks": [
            {"id": "R1", "jd": "Python developer", "kind": "skill", "priority": "mandatory", "status": "met", "quote": "Built Python services"},
            {"id": "R2", "jd": "AWS or Azure", "kind": "skill", "priority": "preferred", "status": "not_evidenced", "quote": None},
            {"id": "R3", "jd": "5 years of relevant experience", "kind": "experience", "priority": "mandatory", "status": "met",
             "quote": "Software Engineer at Acme, 2019-2024"},
        ],
        "work": 7,
        "role": 8,
        "years": {"required": 5, "relevant": 5, "certain": True},
        "reason": "Python services built over five years; no cloud platform evidenced.",
        "review": False,
    }
    value.update(changes)
    return value


class FakeClient:
    model = "fake-model"
    lane = "r9700"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, user_prompt, *, system_prompt, response_schema, stage):
        self.prompts.append(user_prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Completion(content=reply, usage={"input_tokens": 100, "output_tokens": 40}, latency_seconds=2.0, finish_reason="stop")


def pair():
    return PairRecord(pair_id="p:1", source_file="x", row_index=1, resume=RESUME, job_description=JD, gold_label="1",
                      metadata={"selection": "smallest_1"})


def score(client):
    return pv2.score_pair(client, pair(), kit=KIT, system_prompt="system prompt")


def test_score_comes_from_the_kit_and_the_user_message_is_the_documented_json():
    client = FakeClient(["```json\n" + json.dumps(assessment()) + "\n```"])
    record = score(client)
    assert record["status"] == "ok" and record["result"]["score"] == 8 and record["result"]["fit"] == "good_fit"
    assert json.loads(client.prompts[0]) == {"job_description": JD, "resume": RESUME}
    assert record["raw_output"].startswith("```json") and record["input_tokens"] == 100 and record["output_tokens"] == 40


def test_invalid_quote_is_retried_with_the_validation_message():
    bad = assessment()
    bad["checks"][0]["quote"] = "Led a team of ten"
    client = FakeClient([json.dumps(bad), json.dumps(assessment())])
    record = score(client)
    assert record["status"] == "ok" and record["attempts"] == 2 and record["output_tokens"] == 80
    assert "R1: required resume evidence quote not found" in client.prompts[1]


def test_two_invalid_replies_give_an_error_row_with_the_raw_output_and_no_score():
    bad = json.dumps(assessment(work=11))
    record = score(FakeClient([bad, bad]))
    assert record["status"] == "error" and "work must be integer 0..10" in record["error"]
    assert record["raw_output"] == bad and "result" not in record


def test_truncated_output_is_an_error_not_a_score():
    record = score(FakeClient([OutputTokenLimitError("length")]))
    assert record["status"] == "error" and "OutputTokenLimitError" in record["error"]


def test_unscorable_assessment_has_no_score_and_says_so():
    unscorable = {"status": "unscorable", "checks": [], "work": None, "role": None,
                  "years": {"required": None, "relevant": None, "certain": False}, "reason": "Resume is unreadable.", "review": True}
    record = score(FakeClient([json.dumps(unscorable)]))
    assert record["status"] == "ok" and record["result"]["score"] is None
    assert "Unscorable" in pv2.build_breakdown(record["result"], KIT.WEIGHTS)


def test_breakdown_shows_dimensions_caps_and_review():
    flagged = assessment()
    flagged["checks"][0].update(status="not_evidenced", quote=None)
    record = score(FakeClient([json.dumps(flagged)]))
    text = pv2.build_breakdown(record["result"], KIT.WEIGHTS)
    assert "Skills 0/10 x 40% = 0.00" in text and "Education n/a (weight redistributed)" in text
    assert "Caps: <= 6 (R1: explicit must-have not evidenced)" in text and "Review: R1: mandatory qualification not evidenced" in text


def test_ensure_lane_deploys_with_the_token_budget_and_reports_that_it_owns_the_load(monkeypatch):
    calls = []
    monkeypatch.setattr(pv2, "_get", lambda path: {"servers": []})
    monkeypatch.setattr(pv2, "api_post", lambda path, body, timeout=0: (calls.append((path, body)) or (200, {})))
    assert pv2.ensure_lane("r9700", "Model-A", input_tokens=7800, max_output_tokens=4096) is True
    assert calls == [("/api/clients/deploy_lane", {"model": "Model-A", "lane": "r9700", "context_window": 24576,
                                                   "input_tokens": 7800, "max_output_tokens": 4096})]


def test_ensure_lane_never_claims_a_model_that_was_already_loaded(monkeypatch):
    monkeypatch.setattr(pv2, "_get", lambda path: {"servers": [{"target": "r9700", "model": "Model-A", "alive": True}]})
    monkeypatch.setattr(pv2, "api_post", lambda *a, **k: pytest.fail("must not redeploy a loaded model"))
    assert pv2.ensure_lane("r9700", "Model-A", input_tokens=1, max_output_tokens=1) is False


def test_ensure_lane_reports_a_refused_deploy(monkeypatch):
    monkeypatch.setattr(pv2, "_get", lambda path: {"servers": []})
    monkeypatch.setattr(pv2, "api_post", lambda *a, **k: (500, {"error": "Not enough free VRAM to keep Model-A on the GPU"}))
    with pytest.raises(RuntimeError, match="Not enough free VRAM"):
        pv2.ensure_lane("9070xt", "Model-A", input_tokens=1, max_output_tokens=1)


def test_release_unloads_only_our_model_and_retries_while_the_service_is_busy(monkeypatch):
    bodies, replies = [], [(409, {"error": "busy"}), (200, {"released": True})]
    monkeypatch.setattr(pv2.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(pv2, "api_post", lambda path, body, timeout=0: (bodies.append((path, body)) or replies.pop(0)))
    assert pv2.release_lane("r9700", "Model-A") is True
    assert bodies == [("/api/unload", {"lane": "r9700", "model": "Model-A"})] * 2  # never the bare {} or force form


def test_release_does_not_retry_when_another_model_now_serves_the_lane(monkeypatch):
    calls = []
    monkeypatch.setattr(pv2, "api_post", lambda *a, **k: (calls.append(1) or (200, {"released": False})))
    assert pv2.release_lane("r9700", "Model-A") is False and len(calls) == 1


def test_results_from_another_prompt_version_are_ignored(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [
        {"pair_id": "p:1", "status": "ok", "prompt_sha256": "old"},
        {"pair_id": "p:2", "status": "ok", "prompt_sha256": "new"},
        {"pair_id": "p:2", "status": "error", "prompt_sha256": "new"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    latest = pv2.current_records(path, "new")
    assert set(latest) == {"p:2"} and latest["p:2"]["status"] == "error"  # last record wins, old prompt excluded
    assert pv2.prompt_hash("a") != pv2.prompt_hash("b")


def test_status_check_rides_out_the_resets_of_a_just_started_service(monkeypatch):
    replies = [ConnectionResetError("reset"), ConnectionResetError("reset"), {"servers": []}]

    def flaky(path):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(pv2, "_get", flaky)
    monkeypatch.setattr(pv2.time, "sleep", lambda seconds: None)
    assert pv2.get_status() == {"servers": []}


def test_status_check_gives_up_when_the_service_never_answers(monkeypatch):
    monkeypatch.setattr(pv2, "_get", lambda path: (_ for _ in ()).throw(ConnectionResetError("reset")))
    monkeypatch.setattr(pv2.time, "sleep", lambda seconds: None)
    with pytest.raises(ConnectionResetError):
        pv2.get_status(attempts=3)


class ThinkingClient(FakeClient):
    def __init__(self, replies, usages):
        super().__init__(replies)
        self.usages = list(usages)

    def complete(self, user_prompt, *, system_prompt, response_schema, stage):
        completion = super().complete(user_prompt, system_prompt=system_prompt, response_schema=response_schema, stage=stage)
        return Completion(content=completion.content, usage=self.usages.pop(0), latency_seconds=2.0, finish_reason="stop")


def test_thinking_tokens_are_summed_across_retries_when_reported():
    bad = json.dumps(assessment(work=11))
    client = ThinkingClient([bad, json.dumps(assessment())], [
        {"input_tokens": 100, "output_tokens": 40, "thinking_tokens": 7},
        {"input_tokens": 100, "output_tokens": 40, "thinking_tokens": 5},
    ])
    record = score(client)
    assert record["status"] == "ok" and record["thinking_tokens"] == 12


def test_thinking_tokens_stay_unknown_when_the_orchestrator_does_not_report_them():
    record = score(FakeClient([json.dumps(assessment())]))
    assert record["thinking_tokens"] is None  # not written as a false 0
