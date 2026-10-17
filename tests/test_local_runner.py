import json

import pytest

from candidate_fit import local_runner
from candidate_fit.batch_runner import load_results
from candidate_fit.local_runner import LocalRunError, LocalRunner, LocalSettings, context_for
from candidate_fit.tasks import Item

VALID = json.dumps({"input_status": "ok", "work_experiences": [], "skills": [], "projects": [], "education": [], "certifications": []})


class FakeService:
    """Stands in for the orchestrator: tracks which model each lane serves and records every call."""

    def __init__(self, lanes=None):
        self.lanes = dict(lanes or {})  # lane -> (model, budget)
        self.calls = []
        self.replies = []  # queued /api/chat replies: text, ("http", status) or "unreachable"

    def status(self):
        return {"servers": [{"target": lane, "model": model, "alive": True, "reasoning_budget": budget}
                            for lane, (model, budget) in self.lanes.items()]}

    def post(self, path, body, timeout=900.0):
        self.calls.append((path, body))
        if path == "/api/clients/deploy_lane":
            self.lanes[body["lane"]] = (body["model"], 0)
            return 200, {}
        if path == "/api/unload":
            released = self.lanes.pop(body["lane"], (None,))[0] == body["model"]
            return 200, {"released": released}
        if path == "/api/chat":
            reply = self.replies.pop(0) if self.replies else VALID
            if reply == "unreachable":
                raise ConnectionResetError("service restarting")
            if isinstance(reply, tuple):
                return reply[1], {"error": "busy"}
            return 200, {"results": {body["target"]: {
                "status": "completed", "output_text": reply, "finish_reason": "stop",
                "usage": {"input_tokens": 9, "output_tokens": 4, "thinking_tokens": 0},
                "timing": {"tokens_per_second": 30.0, "prefill_tokens_per_second": 500.0}}}}
        raise AssertionError(path)

    def chats(self):
        return [c[1] for c in self.calls if c[0] == "/api/chat"]


@pytest.fixture
def service(monkeypatch):
    fake = FakeService()
    monkeypatch.setattr(local_runner, "api_post", fake.post)
    monkeypatch.setattr(local_runner, "get_status", fake.status)
    monkeypatch.setattr(local_runner, "ensure_service", lambda: None)
    monkeypatch.setattr(local_runner, "client_id", lambda: "test-client")
    monkeypatch.setattr(local_runner.time, "sleep", lambda s: None)
    return fake


def items(n):
    return [Item(i, f"RES-{i}", "system", f"job {i}", 500) for i in range(n)]


def runner(model="gemma-4-12b", **kwargs):
    return LocalRunner(LocalSettings(model, **kwargs), log=lambda m: None)


def test_the_lane_is_loaded_with_the_budget_and_released_afterwards(service, tmp_path):
    result = runner().run("resume_extraction", items(2), tmp_path)
    paths = [c[0] for c in service.calls]
    assert paths[0] == "/api/clients/deploy_lane" and paths[-1] == "/api/unload"
    assert service.calls[-1][1] == {"lane": "9070xt", "model": "gemma-4-12B-it-QAT-Q4_0"}
    assert result["ok"] == 2 and "9070xt" not in service.lanes


def test_another_applications_model_on_the_lane_is_not_replaced_unless_asked(service, tmp_path):
    service.lanes["9070xt"] = ("someone-elses-model", 0)
    with pytest.raises(LocalRunError, match="someone-elses-model"):
        runner().run("resume_extraction", items(1), tmp_path)
    assert all(c[0] != "/api/clients/deploy_lane" for c in service.calls)
    runner(replace_other_model=True).run("resume_extraction", items(1), tmp_path / "second")
    assert any(c[0] == "/api/clients/deploy_lane" for c in service.calls)


def test_a_model_already_loaded_by_someone_else_is_not_unloaded(service, tmp_path):
    service.lanes["9070xt"] = ("gemma-4-12B-it-QAT-Q4_0", 0)
    runner().run("resume_extraction", items(1), tmp_path)
    assert all(c[0] != "/api/unload" for c in service.calls)  # we did not load it, so we leave it


def test_a_run_with_a_different_thinking_budget_than_the_loaded_model_is_refused(service, tmp_path):
    service.lanes["9070xt"] = ("gemma-4-12B-it-QAT-Q4_0", 512)
    with pytest.raises(LocalRunError, match="thinking budget"):
        runner(reasoning_budget=0).run("resume_extraction", items(1), tmp_path)


def test_an_invalid_reply_is_retried_with_the_validation_message(service, tmp_path):
    service.replies = ["not json", VALID]
    runner().run("resume_extraction", items(1), tmp_path)
    chats = service.chats()
    assert len(chats) == 2 and "Your previous reply was rejected" in chats[1]["messages"][1]["content"]
    assert [r["ok"] for r in load_results(tmp_path / "results.jsonl")] == [False, True]


def test_a_rerun_only_does_the_missing_items(service, tmp_path):
    runner().run("resume_extraction", items(2), tmp_path)
    before = len(service.chats())
    summary = runner().run("resume_extraction", items(2), tmp_path)
    assert len(service.chats()) == before and summary["skipped"] == 2


def test_a_busy_service_is_retried_without_using_up_an_attempt(service, tmp_path):
    service.replies = [("http", 503), "unreachable", VALID]
    summary = runner().run("resume_extraction", items(1), tmp_path, max_attempts=1)
    assert summary["ok"] == 1 and len(service.chats()) == 3
    assert [r["attempt"] for r in load_results(tmp_path / "results.jsonl")] == [1]


def test_three_unreachable_items_in_a_row_stop_the_run_and_still_release_the_lane(service, tmp_path):
    service.replies = ["unreachable"] * 40
    with pytest.raises(LocalRunError, match="unreachable for 3 items"):
        runner().run("resume_extraction", items(10), tmp_path)
    assert service.calls[-1][0] == "/api/unload"
    assert load_results(tmp_path / "results.jsonl") == []  # nothing recorded as a model failure


def test_the_speed_figures_are_stored_with_each_result(service, tmp_path):
    runner().run("resume_extraction", items(1), tmp_path)
    (row,) = load_results(tmp_path / "results.jsonl")
    assert row["tokens_per_sec"] == 30.0 and row["prefill_tokens_per_sec"] == 500.0 and row["input_tokens"] == 9


def test_results_from_another_prompt_fingerprint_are_not_reused(service, tmp_path):
    runner().run("resume_extraction", items(1), tmp_path, fingerprint="v1")
    before = len(service.chats())
    runner().run("resume_extraction", items(1), tmp_path, fingerprint="v2")
    assert len(service.chats()) == before + 1


def test_context_is_sized_from_the_largest_request_and_rounded_to_4096():
    big = Item(0, "x", "s" * 9000, "u" * 9000, 6144)
    tokens, context = context_for([big], 6144)
    assert tokens >= 6000 and context % 4096 == 0 and context >= tokens + 6144


def test_a_jd_rubric_with_a_generic_mandatory_skill_is_retried_with_the_reason_and_accepted_once_fixed(service, tmp_path):
    """v2.2 blocks a mandatory skill taken from `otherskills` with a warning only. The reason must reach the retry."""
    import csv

    from candidate_fit import tasks

    csv.field_size_limit(10**9)
    with (tasks.JD_V22_PACKAGE / "tests" / "fixtures" / "results_full_text.csv").open(encoding="utf-8-sig", newline="") as handle:
        row = next(r for r in csv.DictReader(handle) if r["item_id"] == "JD-001951")
    rubric = json.loads(row["output"])
    for skill in rubric["skills"]:
        if skill["name"].lower() in ("problem solving", "effective communication", "technical writing"):
            skill.update(priority="preferred", priority_origin="inferred", importance="supporting")
    service.replies = [row["output"], json.dumps(rubric)]
    jd_item = Item(0, "JD-001951", "system", "job", 500, {"jd": row["jd"], "jd_prompt_version": "v2.2"})
    summary = runner().run("jd_extraction", [jd_item], tmp_path)
    retry = service.chats()[1]["messages"][1]["content"]
    assert "Your previous reply was rejected" in retry and "OTHER_SKILLS_PRIORITY_REVIEW" in retry
    rows = load_results(tmp_path / "results.jsonl")
    assert [r["ok"] for r in rows] == [False, True] and rows[1]["valid_for_cache"] is True and summary["ok"] == 1


def test_a_logistics_policy_that_is_not_an_explicit_condition_is_retried_with_the_reason_and_accepted_once_fixed(service, tmp_path):
    """v2.3 blocks on a LOGISTICS_POLICY_REVIEW warning alone, with no errors. The reason must reach the retry."""
    from test_production_tasks import salesforce_rubric

    jd, unflagged = salesforce_rubric(logistics_hard_gates=False)
    _same_jd, fixed = salesforce_rubric(logistics_hard_gates=True)
    service.replies = [json.dumps(unflagged), json.dumps(fixed)]
    jd_item = Item(0, "JD-000002", "system", "job", 500, {"jd": jd, "jd_prompt_version": "v2.3"})
    summary = runner().run("jd_extraction", [jd_item], tmp_path)
    retry = service.chats()[1]["messages"][1]["content"]
    assert "Your previous reply was rejected" in retry and "LOGISTICS_POLICY_REVIEW" in retry
    rows = load_results(tmp_path / "results.jsonl")
    assert [r["ok"] for r in rows] == [False, True] and summary["ok"] == 1
    assert "work_authorization" in rows[1]["logistics_conditions"] and rows[1]["valid_for_cache"] is True


def test_gpt_oss_is_a_selectable_local_model_on_the_small_lane_and_the_cli_offers_every_local_model():
    from candidate_fit import cli

    assert local_runner.LOCAL_MODELS["gpt-oss-20b"][:2] == ("9070xt", "gpt-oss-20b-GGUF")
    base = ["run", "--task", "jd_extraction", "--backend", "local"]
    for key in local_runner.LOCAL_MODELS:  # the flag takes every model in the one table, so the two cannot drift apart
        assert cli.build_parser().parse_args(base + ["--local-model", key]).local_model == key
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(base + ["--local-model", "not-a-model"])


def test_a_model_the_user_already_loaded_is_used_as_is_and_left_loaded(service, tmp_path):
    service.lanes["9070xt"] = ("gpt-oss-20b-GGUF", 0)
    result = runner("gpt-oss-20b").run("resume_extraction", items(1), tmp_path)
    assert result["ok"] == 1 and all(c[0] not in ("/api/clients/deploy_lane", "/api/unload") for c in service.calls)
    assert service.lanes["9070xt"] == ("gpt-oss-20b-GGUF", 0)


def jd_item_for(version, retry_temperature):
    from candidate_fit import tasks

    return Item(0, "JD-1", "system", "job", 500, {"jd": "a job", "jd_prompt_version": version}, retry_temperature=retry_temperature)


def test_a_gptoss_retry_changes_the_decoding_and_the_first_attempt_does_not(service, tmp_path):
    service.replies = ["not json", "still not json"]
    summary = runner("gpt-oss-20b").run("jd_extraction", [jd_item_for("v2.3-gptoss", 0.3)], tmp_path)
    first, second = service.chats()
    assert (first["temperature"], first["seed"]) == (0, 42)  # the first attempt is the usual greedy request
    assert (second["temperature"], second["seed"]) == (0.3, 43) and summary["failed"] == 1
    rows = load_results(tmp_path / "results.jsonl")
    assert "retry_decoding" not in rows[0] and rows[1]["retry_decoding"] == {"temperature": 0.3, "seed": 43}


def test_a_retry_for_any_other_item_is_exactly_as_before(service, tmp_path):
    """Qwen's items have no retry temperature, so the second request is the first request plus the rejection note."""
    service.replies = ["not json", "still not json"]
    runner("qwen3.8-27b").run("jd_extraction", [jd_item_for("v2.3", None)], tmp_path)
    first, second = service.chats()
    assert (first["temperature"], first["seed"]) == (0, 42) and (second["temperature"], second["seed"]) == (0, 42)
    assert {k: v for k, v in second.items() if k != "messages"} == {k: v for k, v in first.items() if k != "messages"}
    assert second["messages"][0] == first["messages"][0] and second["messages"][1]["content"].startswith(first["messages"][1]["content"])
    assert all("retry_decoding" not in r for r in load_results(tmp_path / "results.jsonl"))


def test_gptoss_is_sent_back_for_a_repeated_key_with_the_reason_and_accepted_when_fixed(service, tmp_path):
    from test_production_tasks import package_fixture

    jd, good = package_fixture()
    clean = json.dumps(good())
    repeated = clean.replace('"input_status": "ok"', '"input_status": "ok", "input_status": "ok"', 1)
    service.replies = [repeated, clean]
    item = Item(0, "JD-1", "system", "job", 500, {"jd": jd, "jd_prompt_version": "v2.3-gptoss"}, retry_temperature=0.3)
    summary = runner("gpt-oss-20b").run("jd_extraction", [item], tmp_path)
    retry = service.chats()[1]["messages"][1]["content"]
    assert "Your previous reply was rejected" in retry and "repeats the JSON key" in retry
    assert [r["ok"] for r in load_results(tmp_path / "results.jsonl")] == [False, True] and summary["ok"] == 1


def test_gemma_26b_is_a_selectable_model_on_the_small_lane_and_has_its_own_jd_prompt():
    from candidate_fit import tasks
    from candidate_fit.runner import RunPlan

    assert local_runner.LOCAL_MODELS["gemma-4-26b-a4b"][:2] == ("9070xt", "gemma-4-26B-A4B-it-QAT-Q4_0")
    plan = RunPlan(task="jd_extraction", backend="local", local_model="gemma-4-26b-a4b")
    plan.validate()
    # v2.3 plus the Gemma rules (the plain shared v2.3 is still available to it with --jd-prompt-version v2.3)
    assert plan.effective_jd_prompt_version == "v2.3-gemma26" == tasks.default_jd_prompt_version("gemma-4-26b-a4b")
