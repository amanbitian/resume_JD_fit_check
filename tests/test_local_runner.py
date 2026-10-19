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


# --- Long unattended runs: progress lines, a clean pause with Ctrl+C, resuming, and stopping on a broken service ---


def logged_runner(model="qwen3.8-27b"):
    lines = []
    return local_runner.LocalRunner(local_runner.LocalSettings(model), log=lines.append), lines


def test_a_progress_line_is_logged_after_every_n_rows_and_for_the_last_short_group(service, tmp_path):
    run, lines = logged_runner()
    run.run("resume_extraction", items(7), tmp_path, progress_every=3)
    progress = [line for line in lines if line.startswith("progress")]
    assert len(progress) == 3  # after rows 3 and 6, and for the 7th
    assert progress[0].startswith("progress 3/7 (42.86%) | usable 3, failed 0 |")
    assert progress[1].startswith("progress 6/7") and progress[2].startswith("progress 7/7 (100.00%)")
    assert all("s/row" in line and "ETA" in line and "left" in line for line in progress)
    assert any(line.startswith("resuming: 0 of 7 already settled") for line in lines)


def test_no_progress_lines_unless_asked_for(service, tmp_path):
    run, lines = logged_runner()
    run.run("resume_extraction", items(4), tmp_path)
    assert not any(line.startswith(("progress", "resuming")) for line in lines)


def test_a_resumed_run_counts_what_is_already_done_and_does_not_repeat_it(service, tmp_path):
    first, _ = logged_runner()
    first.run("resume_extraction", items(5), tmp_path, progress_every=3)
    before = len(service.chats())
    second, lines = logged_runner()
    summary = second.run("resume_extraction", items(8), tmp_path, progress_every=3)
    assert len(service.chats()) - before == 3 and summary["skipped"] == 5 and summary["ok"] == 3  # only the 3 new rows
    assert any(line.startswith("resuming: 5 of 8 already settled (5 usable, 0 failed); 3 left") for line in lines)
    assert [line for line in lines if line.startswith("progress")][-1].startswith("progress 8/8 (100.00%) | usable 8")


def test_a_row_that_runs_out_of_attempts_is_counted_as_failed_in_the_progress_line(service, tmp_path):
    service.replies = ["not json", "not json"] + [VALID] * 4
    run, lines = logged_runner()
    run.run("resume_extraction", items(3), tmp_path, progress_every=3)
    assert [line for line in lines if line.startswith("progress")][0].startswith("progress 3/3 (100.00%) | usable 2, failed 1 |")


def test_ctrl_c_finishes_the_current_request_then_stops_cleanly_and_a_rerun_continues(service, tmp_path, monkeypatch):
    import signal

    original = service.post
    chats = []

    def post(path, body, timeout=900.0):
        if path == "/api/chat":
            chats.append(1)
            if len(chats) == 2:
                signal.raise_signal(signal.SIGINT)  # the user presses Ctrl+C while the second request is running
        return original(path, body, timeout)

    monkeypatch.setattr(local_runner, "api_post", post)
    before = signal.getsignal(signal.SIGINT)
    run, lines = logged_runner()
    summary = run.run("resume_extraction", items(6), tmp_path, progress_every=3)
    assert summary["stopped"] is True and summary["ok"] == 2  # the request in flight finished and was saved
    assert len(chats) == 2 and len(load_results(tmp_path / "results.jsonl")) == 2
    assert any("stop requested" in line for line in lines) and any("stopped on request after 2 rows" in line for line in lines)
    assert "9070xt" not in service.lanes and "r9700" not in service.lanes  # the model was released
    assert signal.getsignal(signal.SIGINT) is before  # the normal Ctrl+C is restored afterwards
    monkeypatch.setattr(local_runner, "api_post", original)
    again, _ = logged_runner()
    rest = again.run("resume_extraction", items(6), tmp_path)
    assert rest["skipped"] == 2 and rest["ok"] == 4 and "stopped" not in rest  # nothing is repeated


def test_a_second_ctrl_c_stops_at_once_and_still_releases_the_model_and_keeps_the_results(service, tmp_path, monkeypatch):
    import signal

    original = service.post
    chats = []

    def post(path, body, timeout=900.0):
        if path == "/api/chat":
            chats.append(1)
            if len(chats) == 3:
                signal.raise_signal(signal.SIGINT)
                signal.raise_signal(signal.SIGINT)  # pressed twice
        return original(path, body, timeout)

    monkeypatch.setattr(local_runner, "api_post", post)
    before = signal.getsignal(signal.SIGINT)
    run, _ = logged_runner()
    with pytest.raises(KeyboardInterrupt):
        run.run("resume_extraction", items(6), tmp_path)
    assert len(load_results(tmp_path / "results.jsonl")) == 2 and "r9700" not in service.lanes
    assert signal.getsignal(signal.SIGINT) is before


def test_a_service_that_keeps_returning_errors_stops_the_run_instead_of_marking_every_item_failed(service, tmp_path):
    service.replies = [("http", 500)] * 40
    run, _ = logged_runner()
    with pytest.raises(LocalRunError, match="returned an error for 5 requests in a row.*not marked failed"):
        run.run("resume_extraction", items(30), tmp_path)
    assert len(service.chats()) == 5 and len(load_results(tmp_path / "results.jsonl")) == 5  # not 60
    assert "r9700" not in service.lanes  # the model is released even though the run stopped


def test_an_error_followed_by_a_good_reply_does_not_count_towards_the_stop(service, tmp_path):
    service.replies = [("http", 500), ("http", 500), VALID, ("http", 500), ("http", 500), VALID, VALID, VALID, VALID, VALID]
    run, _ = logged_runner()
    summary = run.run("resume_extraction", items(5), tmp_path)
    assert "stopped" not in summary and summary["ok"] + summary["failed"] == 5


def test_an_item_that_failed_before_a_stop_is_retried_with_the_reason_it_failed(service, tmp_path):
    (tmp_path / "results.jsonl").write_text(json.dumps({"item_id": "RES-0", "model": "Qwen3.8-27B-Q4_K_M", "attempt": 1, "fp": "",
                                                       "ok": False, "error": "invalid reply: the earlier reason"}) + "\n", encoding="utf-8")
    run, _ = logged_runner()
    run.run("resume_extraction", items(1), tmp_path)
    sent = service.chats()[0]["messages"][1]["content"]
    assert "Your previous reply was rejected: invalid reply: the earlier reason" in sent


# --- Several rows at once, capped by the tokens they need in the server's shared KV memory ---

import math
import random
import threading

POOL = 40_000


class ParallelFake:
    """A model service that handles requests at the same time, and records how many are running and the KV memory they need."""

    def __init__(self, delay=0.25, reply=None, server_context=None, status_code=200):
        self.lock = threading.Lock()
        self.delay, self.reply, self.server_context, self.status_code = delay, reply or VALID, server_context, status_code
        self.running, self.max_running, self.max_reserved = [], 0, 0
        self.lanes, self.calls, self.context = {}, [], None

    def status(self):
        return {"servers": [{"target": lane, "model": model, "alive": True, "reasoning_budget": 0, "context_window": self.context}
                            for lane, model in self.lanes.items()]}

    def post(self, path, body, timeout=900.0):
        with self.lock:
            self.calls.append((path, body))
        if path == "/api/clients/deploy_lane":
            with self.lock:
                self.lanes[body["lane"]] = body["model"]
                self.context = self.server_context or body["context_window"]  # a server may end up with less than was asked for
            return 200, {}
        if path == "/api/unload":
            with self.lock:
                self.lanes.pop(body["lane"], None)
            return 200, {"released": True}
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        need = math.ceil((len(system) + len(user)) / local_runner.CHARS_PER_TOKEN) + body["max_tokens"] + local_runner.SAFETY_TOKENS
        with self.lock:
            self.running.append(need)
            self.max_running = max(self.max_running, len(self.running))
            self.max_reserved = max(self.max_reserved, sum(self.running))
        threading.Event().wait(self.delay)  # a real wait that the sleep patch in other tests does not touch
        with self.lock:
            self.running.remove(need)
        if self.status_code != 200:
            return self.status_code, {"error": "broken"}
        return 200, {"results": {body["target"]: {"status": "completed", "output_text": self.reply, "finish_reason": "stop",
                                                 "usage": {"input_tokens": 9, "output_tokens": 4}, "timing": {}}}}

    def chats(self):
        return [c[1] for c in self.calls if c[0] == "/api/chat"]


@pytest.fixture
def parallel(monkeypatch):
    def install(**kwargs):
        fake = ParallelFake(**kwargs)
        monkeypatch.setattr(local_runner, "api_post", fake.post)
        monkeypatch.setattr(local_runner, "get_status", fake.status)
        monkeypatch.setattr(local_runner, "ensure_service", lambda: None)
        monkeypatch.setattr(local_runner, "client_id", lambda: "test-client")
        return fake
    return install


def rows_of(sizes_in_chars, cap=8192):
    """Rows whose request text has the given number of characters, so their token needs are known exactly."""
    return [Item(i, f"ROW-{i}", "s", "u" * chars, cap) for i, chars in enumerate(sizes_in_chars)]


TINY, SMALL, BIG = 4_000, 16_000, 40_000  # needing about 9.7k, 12.7k and 18.7k tokens with an 8,192 output limit


def run_parallel(items_, workers, tmp_path, pool=POOL, **kwargs):
    run, lines = logged_runner()
    summary = run.run("resume_extraction", items_, tmp_path, workers=workers, kv_pool=pool, **kwargs)
    return summary, lines


def test_small_rows_run_up_to_four_at_once_and_never_more_than_fit(parallel, tmp_path):
    fake = parallel()
    summary, _ = run_parallel(rows_of([TINY] * 12), 8, tmp_path)  # about 9.7k tokens each: four fit in 40k
    assert summary["ok"] == 12 and fake.max_running == 4 and fake.max_reserved <= POOL


def test_medium_rows_run_three_at_once(parallel, tmp_path):
    fake = parallel()
    summary, _ = run_parallel(rows_of([SMALL] * 9), 8, tmp_path)  # about 12.7k each: three fit, a fourth would not
    assert summary["ok"] == 9 and fake.max_running == 3 and fake.max_reserved <= POOL


def test_two_big_rows_run_together_but_never_three(parallel, tmp_path):
    fake = parallel()
    summary, _ = run_parallel(rows_of([BIG] * 6), 8, tmp_path)  # about 18.7k each: two fit (37k), three would need 56k
    assert summary["ok"] == 6 and fake.max_running == 2 and fake.max_reserved <= POOL


def test_the_workers_limit_still_applies_when_more_would_fit(parallel, tmp_path):
    fake = parallel()
    run_parallel(rows_of([TINY] * 10), 2, tmp_path)
    assert fake.max_running == 2


def test_the_memory_total_is_never_exceeded_whatever_mix_of_row_sizes_arrives(parallel, tmp_path):
    random.seed(7)
    sizes = [random.choice([TINY, SMALL, BIG, 800, 24_000]) for _ in range(40)]
    fake = parallel(delay=0.06)
    summary, _ = run_parallel(rows_of(sizes), 6, tmp_path)
    assert summary["ok"] == 40 and fake.max_reserved <= POOL and 1 < fake.max_running <= 6


def test_a_big_row_is_not_starved_by_small_ones_behind_it(parallel, tmp_path):
    fake = parallel(delay=0.1)
    order = []
    original = fake.post

    def post(path, body, timeout=900.0):
        if path == "/api/chat":
            order.append(len(body["messages"][1]["content"]))
        return original(path, body, timeout)

    local_runner.api_post = post
    run_parallel(rows_of([SMALL, BIG, SMALL, SMALL, BIG, TINY]), 4, tmp_path)
    assert order.index(BIG) < order.index(TINY)  # rows start in dataset order, so the first big row is not skipped past forever
    assert len(order) == 6


def test_a_row_that_could_never_fit_stops_the_run_before_anything_is_loaded_or_sent(parallel, tmp_path):
    fake = parallel()
    with pytest.raises(LocalRunError, match="largest row needs .* tokens.*pool is 20,000"):
        run_parallel(rows_of([TINY, 120_000]), 3, tmp_path, pool=20_000)
    assert fake.calls == []


def test_several_requests_at_once_need_the_pool_size(parallel, tmp_path):
    parallel()
    run, _ = logged_runner()
    with pytest.raises(LocalRunError, match="needs kv_pool"):
        run.run("resume_extraction", rows_of([TINY]), tmp_path, workers=2)


def test_the_model_is_loaded_with_the_whole_pool_and_the_pool_the_server_really_has_is_what_counts(parallel, tmp_path):
    fake = parallel(server_context=30_000)  # the service ends up with less than the 40,000 asked for
    summary, lines = run_parallel(rows_of([BIG] * 4), 4, tmp_path)
    deploy = next(c[1] for c in fake.calls if c[0] == "/api/clients/deploy_lane")
    assert deploy["context_window"] == POOL  # asked for the pool
    assert any("the server has a 30,000-token pool, less than the 40,000 asked for" in line for line in lines)
    assert summary["ok"] == 4 and fake.max_running == 1 and fake.max_reserved <= 30_000  # 18.7k each: only one fits in 30k


def test_the_start_of_a_parallel_run_says_how_many_rows_fit_at_once(parallel, tmp_path):
    parallel()
    _, lines = run_parallel(rows_of([TINY, SMALL, BIG, SMALL]), 4, tmp_path)
    line = next(line for line in lines if line.startswith("parallel:"))
    assert "up to 4 requests at once in a 40,000-token pool" in line and "so 4 / 3 / 2 rows at once" in line


def test_running_rows_in_parallel_gives_the_same_results_as_one_at_a_time(parallel, tmp_path):
    parallel(delay=0.02)
    sizes = [TINY, SMALL, BIG, TINY, SMALL, SMALL, TINY, BIG]
    run_parallel(rows_of(sizes), 4, tmp_path / "parallel")
    run_parallel(rows_of(sizes), 1, tmp_path / "one_by_one")

    def summary(folder):
        rows = load_results(folder / "results.jsonl")
        return sorted((r["item_id"], r["attempt"], r["ok"], r["json_valid"], r["error"], r["input_status"]) for r in rows)

    assert summary(tmp_path / "parallel") == summary(tmp_path / "one_by_one") and len(summary(tmp_path / "parallel")) == 8


def test_a_parallel_run_resumes_and_skips_what_is_done(parallel, tmp_path):
    parallel(delay=0.02)
    run_parallel(rows_of([TINY] * 5), 1, tmp_path)  # five rows one at a time, then the same folder continues in parallel
    fake = parallel(delay=0.02)
    summary, lines = run_parallel(rows_of([TINY] * 12), 4, tmp_path, progress_every=3)
    assert summary["skipped"] == 5 and summary["ok"] == 7 and len(fake.chats()) == 7
    assert any(line.startswith("resuming: 5 of 12 already settled") for line in lines)


def test_progress_lines_in_a_parallel_run_show_how_many_are_running(parallel, tmp_path):
    parallel(delay=0.05)
    _, lines = run_parallel(rows_of([TINY] * 7), 3, tmp_path, progress_every=3)
    progress = [line for line in lines if line.startswith("progress")]
    assert progress and all("s/row" in line and "running" in line and "ETA" in line for line in progress)
    assert progress[-1].startswith("progress 7/7 (100.00%)")


def test_ctrl_c_in_a_parallel_run_lets_the_running_rows_finish_starts_no_new_ones_and_a_rerun_continues(parallel, tmp_path):
    import signal

    fake = parallel(delay=0.3)
    original = fake.post
    seen = []

    def post(path, body, timeout=900.0):
        if path == "/api/chat":
            seen.append(1)
            if len(seen) == 3:
                signal.raise_signal(signal.SIGINT)
        return original(path, body, timeout)

    local_runner.api_post = post
    before = signal.getsignal(signal.SIGINT)
    summary, lines = run_parallel(rows_of([TINY] * 14), 4, tmp_path)
    saved = load_results(tmp_path / "results.jsonl")
    assert summary["stopped"] is True and 3 <= len(saved) <= 4 + 3  # those already started finished; none began after the stop
    assert len(saved) == len(seen) and not fake.lanes and signal.getsignal(signal.SIGINT) is before
    assert any("stopped on request after" in line for line in lines)
    local_runner.api_post = original
    rest, _ = run_parallel(rows_of([TINY] * 14), 4, tmp_path)
    assert rest["skipped"] == len(saved) and rest["ok"] == 14 - len(saved) and "stopped" not in rest  # nothing repeated


def test_a_broken_service_stops_a_parallel_run_after_a_few_errors_and_releases_the_model(parallel, tmp_path):
    fake = parallel(delay=0.05, status_code=500)
    with pytest.raises(LocalRunError, match="returned an error for .* requests in a row.*not marked failed"):
        run_parallel(rows_of([TINY] * 60), 4, tmp_path)
    assert len(fake.chats()) <= local_runner.MAX_SERVICE_ERRORS_IN_A_ROW + 4  # a few in flight finish; not the other 50+
    assert not fake.lanes  # released


def test_a_single_worker_run_does_not_touch_the_pool_logic(parallel, tmp_path):
    fake = parallel(delay=0.01)
    run, lines = logged_runner()
    summary = run.run("resume_extraction", rows_of([BIG] * 3), tmp_path)  # no workers, no pool: the one-at-a-time path
    assert summary["ok"] == 3 and fake.max_running == 1 and not any(line.startswith("parallel:") for line in lines)
    deploy = next(c[1] for c in fake.calls if c[0] == "/api/clients/deploy_lane")
    assert deploy["context_window"] == local_runner.context_for(rows_of([BIG] * 3), 8192)[1]  # sized as before
