import csv
import json

import pytest

from candidate_fit.batch_runner import (
    RunLock,
    RunLockedError,
    chunk,
    export_csv,
    load_results,
    run_batches,
)
from candidate_fit.cloud_batch import BatchApiError, BatchStatus
from candidate_fit.tasks import Item

VALID = json.dumps({"input_status": "ok", "work_experiences": [], "skills": [], "projects": [], "education": [], "certifications": []})


class ScriptedClient:
    """Provider stand-in: each submitted batch answers from a script of replies keyed by custom id."""

    provider = "fake"
    model = "fake-model"
    max_requests = 1000
    max_bytes = 10**9

    def __init__(self, script=None, found=None):
        self.script = list(script or [])  # one dict per batch: custom_id -> reply text (or "__expire__")
        self.submitted = []
        self.intents = []
        self.batches = {}
        self.found = found or []  # what find_submitted() reports
        self.fail_submit = None

    def submit(self, requests, intent_id=None):
        if self.fail_submit:
            raise self.fail_submit
        batch_id = f"batch-{len(self.submitted) + 1}"
        self.submitted.append([r["custom_id"] for r in requests])
        self.intents.append(intent_id)
        answers = self.script.pop(0) if self.script else {}
        if answers.get("__expire__"):
            self.batches[batch_id] = {"state": "expired", "replies": []}
        else:
            self.batches[batch_id] = {"state": "ended", "replies": [
                {"custom_id": r["custom_id"], "ok": True, "text": answers.get(r["custom_id"], VALID),
                 "finish_reason": "stop", "input_tokens": 5, "output_tokens": 3, "thinking_tokens": None, "error": None}
                for r in requests]}
        return batch_id

    def find_submitted(self, intent_id, since_epoch, expected_count, exclude):
        return list(self.found)

    def status(self, batch_id):
        state = self.batches[batch_id]["state"]
        return BatchStatus(done=state in {"ended", "expired"}, state=state, counts={})

    def results(self, batch_id):
        return list(self.batches[batch_id]["replies"])


def items(n, prefix="RES"):
    return [Item(i, f"{prefix}-{i}", "system", f"job {i}", 500) for i in range(n)]


def no_sleep(_seconds):
    return None


def run(folder, client, its, **kwargs):
    kwargs.setdefault("sleep", no_sleep)
    return run_batches("resume_extraction", its, client, folder, **kwargs)


def test_invalid_replies_are_retried_in_a_second_attempt_and_valid_ones_are_not(tmp_path):
    client = ScriptedClient([{"a1-1": "this is not json"}, {}])
    summary = run(tmp_path, client, items(3))
    rows = load_results(tmp_path / "results.jsonl")
    assert [r["attempt"] for r in rows if r["item_id"] == "RES-1"] == [1, 2]
    assert [r["ok"] for r in rows if r["item_id"] == "RES-1"] == [False, True]
    assert summary["ok"] == 3 and summary["failed"] == 0
    assert client.submitted == [["a1-0", "a1-1", "a1-2"], ["a2-1"]]


def test_a_rerun_skips_finished_items_and_submits_nothing(tmp_path):
    run(tmp_path, ScriptedClient([{}]), items(2))
    second = ScriptedClient([])
    assert run(tmp_path, second, items(2))["ok"] == 2 and second.submitted == []


def test_a_different_sample_in_the_same_folder_runs_only_the_new_items(tmp_path):
    run(tmp_path, ScriptedClient([{}]), items(2))  # RES-0 and RES-1 are done
    other = [Item(0, "RES-1", "system", "job 1", 500), Item(1, "RES-9", "system", "job 9", 500)]  # RES-1 again, RES-9 is new
    client = ScriptedClient([{}])
    summary = run(tmp_path, client, other)
    assert client.submitted == [["a1-1"]]  # only RES-9, and its custom id uses its position in this sample
    assert summary["ok"] == 2
    rows = [r for r in load_results(tmp_path / "results.jsonl") if r["item_id"] == "RES-9"]
    assert len(rows) == 1  # results are keyed by item id, never by position


def test_results_made_under_a_different_prompt_fingerprint_are_not_reused(tmp_path):
    run(tmp_path, ScriptedClient([{}]), items(2), fingerprint="prompt-v1")
    client = ScriptedClient([{}])
    run(tmp_path, client, items(2), fingerprint="prompt-v2")
    assert client.submitted == [["a1-0", "a1-1"]]  # everything runs again under the new prompt
    csv_path = export_csv("resume_extraction", items(2), tmp_path, "fake-model", fingerprint="prompt-v2")
    assert csv_path.exists()
    old = [r for r in load_results(tmp_path / "results.jsonl") if r["fp"] == "prompt-v1"]
    assert len(old) == 2  # the old results stay in the file as history


def test_a_batch_submitted_before_a_crash_is_polled_not_resubmitted(tmp_path):
    state = {"task": "resume_extraction", "model": "fake-model", "provider": "fake", "intents": [], "batches": [
        {"batch_id": "batch-old", "attempt": 1, "fp": "", "custom_ids": {"a1-0": "RES-0"}, "done": False}]}
    (tmp_path / "batch_state.json").write_text(json.dumps(state), encoding="utf-8")
    client = ScriptedClient([])
    client.batches["batch-old"] = {"state": "ended", "replies": [
        {"custom_id": "a1-0", "ok": True, "text": VALID, "finish_reason": "stop", "input_tokens": 1,
         "output_tokens": 1, "thinking_tokens": None, "error": None}]}
    summary = run(tmp_path, client, items(1))
    assert client.submitted == [] and summary["ok"] == 1


def test_an_unfinished_batch_for_items_outside_this_sample_is_refused(tmp_path):
    state = {"task": "resume_extraction", "model": "fake-model", "provider": "fake", "intents": [], "batches": [
        {"batch_id": "batch-old", "attempt": 1, "fp": "", "custom_ids": {"a1-0": "RES-OTHER"}, "done": False}]}
    (tmp_path / "batch_state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="outside this run's sample"):
        run(tmp_path, ScriptedClient([]), items(1))


def test_an_interrupted_submission_that_the_provider_did_create_is_adopted_not_paid_for_twice(tmp_path):
    state = {"task": "resume_extraction", "model": "fake-model", "provider": "fake", "batches": [], "intents": [
        {"intent_id": "abc", "attempt": 1, "fp": "", "created_epoch": 1.0, "expected": 1, "custom_ids": {"a1-0": "RES-0"}}]}
    (tmp_path / "batch_state.json").write_text(json.dumps(state), encoding="utf-8")
    client = ScriptedClient([], found=["batch-created-before-crash"])
    client.batches["batch-created-before-crash"] = {"state": "ended", "replies": [
        {"custom_id": "a1-0", "ok": True, "text": VALID, "finish_reason": "stop", "input_tokens": 1,
         "output_tokens": 1, "thinking_tokens": None, "error": None}]}
    summary = run(tmp_path, client, items(1))
    assert client.submitted == [] and summary["ok"] == 1


def test_an_interrupted_submission_with_nothing_at_the_provider_is_submitted_again(tmp_path):
    state = {"task": "resume_extraction", "model": "fake-model", "provider": "fake", "batches": [], "intents": [
        {"intent_id": "abc", "attempt": 1, "fp": "", "created_epoch": 1.0, "expected": 1, "custom_ids": {"a1-0": "RES-0"}}]}
    (tmp_path / "batch_state.json").write_text(json.dumps(state), encoding="utf-8")
    client = ScriptedClient([{}], found=[])
    assert run(tmp_path, client, items(1))["ok"] == 1 and client.submitted == [["a1-0"]]


def test_an_ambiguous_interrupted_submission_stops_instead_of_guessing(tmp_path):
    state = {"task": "resume_extraction", "model": "fake-model", "provider": "fake", "batches": [], "intents": [
        {"intent_id": "abc", "attempt": 1, "fp": "", "created_epoch": 1.0, "expected": 1, "custom_ids": {"a1-0": "RES-0"}}]}
    (tmp_path / "batch_state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="could match 2 batches"):
        run(tmp_path, ScriptedClient([], found=["b1", "b2"]), items(1))


def test_a_submission_the_provider_rejects_leaves_no_intent_but_a_server_error_keeps_it(tmp_path):
    rejected = ScriptedClient([])
    rejected.fail_submit = BatchApiError("bad request", 400)
    with pytest.raises(BatchApiError):
        run(tmp_path, rejected, items(1))
    assert json.loads((tmp_path / "batch_state.json").read_text())["intents"] == []

    unsure = ScriptedClient([])
    unsure.fail_submit = BatchApiError("gateway timeout", 504)
    with pytest.raises(BatchApiError):
        run(tmp_path, unsure, items(1))
    assert len(json.loads((tmp_path / "batch_state.json").read_text())["intents"]) == 1


def test_an_expired_batch_makes_its_items_retry_then_fail_cleanly(tmp_path):
    client = ScriptedClient([{"__expire__": True}, {"__expire__": True}])
    summary = run(tmp_path, client, items(1), max_attempts=2)
    rows = load_results(tmp_path / "results.jsonl")
    assert [r["attempt"] for r in rows] == [1, 2] and all(r["ok"] is False for r in rows)
    assert summary["failed"] == 1 and summary["failed_examples"][0][0] == "RES-0"


def test_polling_waits_until_the_batch_ends(tmp_path):
    client = ScriptedClient([{}])
    original = client.status
    calls = {"n": 0}

    def slow(batch_id):
        calls["n"] += 1
        client.batches[batch_id]["state"] = "in_progress" if calls["n"] < 3 else "ended"
        return original(batch_id)

    client.status = slow
    sleeps = []
    run(tmp_path, client, items(1), sleep=sleeps.append, poll_seconds=5)
    assert sleeps == [5, 5]


def test_a_batch_that_never_ends_raises_with_the_batch_id(tmp_path):
    client = ScriptedClient([{}])
    client.status = lambda batch_id: BatchStatus(done=False, state="in_progress", counts={})
    clock_values = iter([0.0, 10.0, 999999.0])
    with pytest.raises(RuntimeError, match="batch-1"):
        run(tmp_path, client, items(1), clock=lambda: next(clock_values), max_wait_seconds=60)


def test_a_different_model_cannot_reuse_a_run_folder(tmp_path):
    run(tmp_path, ScriptedClient([{}]), items(1))
    other = ScriptedClient([])
    other.model = "another-model"
    with pytest.raises(RuntimeError, match="different run"):
        run(tmp_path, other, items(1))


def test_two_processes_cannot_run_in_the_same_folder(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(RunLockedError):
            with RunLock(tmp_path):
                pass
    with RunLock(tmp_path):  # released again afterwards
        pass


def test_csv_export_has_one_row_per_item_the_output_last_and_keeps_speed_columns(tmp_path):
    run(tmp_path, ScriptedClient([{}]), items(2))
    rows = load_results(tmp_path / "results.jsonl")
    rows[0].update(tokens_per_sec=30.5, prefill_tokens_per_sec=500.0, latency_seconds=2.0)
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    path = export_csv("resume_extraction", items(3), tmp_path, "fake-model")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        out = list(reader)
        assert reader.fieldnames[-1] == "output"
    assert [r["resume_id"] for r in out] == ["RES-0", "RES-1", "RES-2"] and "item_id" not in out[0]
    assert out[0]["tokens_per_sec"] == "30.5" and out[0]["prefill_tokens_per_sec"] == "500.0"
    assert out[2]["status"] == "not run"


def test_batches_split_on_request_count_and_bytes():
    class Small:
        max_requests = 2
        max_bytes = 10**9

    requests = [{"custom_id": str(i), "system": "s", "user": "u", "max_output_tokens": 1} for i in range(5)]
    assert [len(part) for part in chunk(requests, Small())] == [2, 2, 1]

    class Tiny:
        max_requests = 1000
        max_bytes = 200

    assert len(chunk(requests, Tiny())) > 1


def test_the_csv_carries_the_complete_source_text_beside_every_result_including_items_not_run(tmp_path):
    long_jd = "Line one, with a comma.\nLine two with \"quotes\" and unicode \u2013 dash. " + "More text. " * 400
    # stored rows from a finished run; the third item has no result yet
    rows = [{"item_id": "JD-0", "model": "m", "attempt": 1, "fp": "", "ok": True, "output": "{}", "json_valid": True,
             "input_status": "ok", "finish_reason": "stop"}]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    its = [Item(0, "JD-0", "s", "u", 100, {"jd": long_jd}), Item(1, "JD-1", "s", "u", 100, {"jd": "second jd"})]
    path = export_csv("jd_extraction", its, tmp_path, "m")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        out = list(reader)
    assert reader.fieldnames[:2] == ["jd_id", "jd"] and reader.fieldnames[-1] == "output"
    assert out[0]["jd"] == long_jd  # complete and unaltered, not shortened or re-wrapped
    assert out[1]["jd"] == "second jd" and out[1]["status"] == "not run"


def test_resume_and_pair_csvs_get_their_own_text_columns(tmp_path):
    resume_item = Item(0, "RES-0", "s", "u", 100, {"resume": "full resume text"})
    path = export_csv("resume_extraction", [resume_item], tmp_path / "r", "m")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["resume"] == "full resume text" and "jd" not in row
    pair_item = Item(0, "PAIR-0", "s", "u", 100, {"resume": "the resume", "job_description": "the jd"})
    path = export_csv("resume_jd_pair", [pair_item], tmp_path / "p", "m")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    assert reader.fieldnames[:3] == ["pair_id", "resume", "jd"] and (row["resume"], row["jd"]) == ("the resume", "the jd")


def test_a_csv_that_is_open_elsewhere_gives_a_clear_message_not_a_traceback(tmp_path, monkeypatch):
    from pathlib import Path

    real_open = Path.open

    def locked(self, mode="r", *args, **kwargs):
        if self.name == "results.csv" and "w" in mode:
            raise PermissionError(13, "Permission denied")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", locked)
    with pytest.raises(RuntimeError, match="open in another program"):
        export_csv("resume_extraction", [Item(0, "RES-0", "s", "u", 100, {"resume": "x"})], tmp_path, "m")
