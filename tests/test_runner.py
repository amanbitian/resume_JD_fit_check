import json

import pytest

from candidate_fit import runner
from candidate_fit.cloud_batch import BatchApiError, BatchStatus
from candidate_fit.runner import RunPlan, execute
from candidate_fit.tasks import Item

JD_REPLY = json.dumps({"input_status": "ok", "work_experiences": [], "skills": [], "projects": [], "education": [], "certifications": []})


class FakeCloud:
    provider = "fake"
    model = "claude-haiku-5-5"
    max_requests = 1000
    max_bytes = 10**9

    def __init__(self):
        self.submitted = []
        self.batches = {}

    def submit(self, requests, intent_id=None):
        batch_id = f"b{len(self.submitted) + 1}"
        self.submitted.append([r["custom_id"] for r in requests])
        self.batches[batch_id] = [{"custom_id": r["custom_id"], "ok": True, "text": JD_REPLY, "finish_reason": "stop",
                                   "input_tokens": 100, "output_tokens": 50, "thinking_tokens": None, "error": None}
                                  for r in requests]
        return batch_id

    def find_submitted(self, *args):
        return []

    def status(self, batch_id):
        return BatchStatus(done=True, state="ended", counts={})

    def results(self, batch_id):
        return self.batches[batch_id]


@pytest.fixture
def two_items(monkeypatch):
    items = [Item(0, "RES-0", "system prompt", "job zero", 6144), Item(1, "RES-1", "system prompt", "job one", 6144)]
    monkeypatch.setattr(runner, "build_items", lambda *a, **k: items)
    return items


def cloud_plan(tmp_path, **kwargs):
    return RunPlan(task="resume_extraction", backend="cloud", cloud_provider="anthropic", cloud_model="claude-haiku-5-5",
                   output_root=tmp_path, **kwargs)


def test_plans_are_validated():
    bad_plans = [
        RunPlan(task="nonsense", backend="local", local_model="gemma-4-12b"),
        RunPlan(task="resume_extraction", backend="local", local_model="not-a-model"),
        RunPlan(task="resume_extraction", backend="cloud", cloud_provider="anthropic"),
        RunPlan(task="resume_extraction", backend="cloud", cloud_provider="anthropic", cloud_model="m", reasoning_effort="low"),
        RunPlan(task="resume_extraction", backend="cloud", cloud_provider="openai", cloud_model="m", reasoning_effort="extreme"),
        RunPlan(task="resume_extraction", backend="local", local_model="gemma-4-12b", limit=0),
        RunPlan(task="resume_extraction", backend="local", local_model="gemma-4-12b", limit=5, item_ids=("RES-1",)),
        RunPlan(task="resume_extraction", backend="elsewhere"),
    ]
    for plan in bad_plans:
        with pytest.raises(ValueError):
            plan.validate()


def test_the_output_folder_is_per_task_backend_and_model_and_safe_as_a_path(tmp_path):
    plan = RunPlan(task="resume_extraction", backend="cloud", cloud_provider="openai", cloud_model="gpt 5/luna:1",
                   output_root=tmp_path)
    assert plan.folder() == tmp_path / "resume_extraction" / "cloud-openai-gpt_5_luna_1"


def test_a_missing_api_key_is_reported_before_the_user_is_asked_to_confirm(tmp_path, two_items, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    asked = []
    with pytest.raises(BatchApiError, match="ANTHROPIC_API_KEY"):
        execute(cloud_plan(tmp_path), confirm=lambda summary: asked.append(summary) or True, log=lambda m: None)
    assert asked == []


def test_a_cloud_run_above_the_cost_ceiling_is_refused_before_the_confirmation(tmp_path, two_items):
    asked = []
    with pytest.raises(RuntimeError, match="above the limit"):
        execute(cloud_plan(tmp_path, max_cost_usd=0.000001), client=FakeCloud(),
                confirm=lambda summary: asked.append(summary) or True, log=lambda m: None)
    assert asked == []


def test_declining_the_confirmation_sends_nothing(tmp_path, two_items):
    cloud = FakeCloud()
    result = execute(cloud_plan(tmp_path), client=cloud, confirm=lambda summary: False, log=lambda m: None)
    assert result == {"cancelled": True} and cloud.submitted == []


def test_the_summary_says_batch_only_shows_the_cost_and_where_the_data_goes(tmp_path, two_items):
    shown = []
    execute(cloud_plan(tmp_path), client=FakeCloud(), confirm=lambda summary: shown.extend(summary) or False, log=lambda m: None)
    text = "\n".join(shown)
    assert "BATCH ONLY" in text and "Estimated batch cost" in text and "sent to anthropic servers" in text


def test_a_confirmed_cloud_run_submits_one_batch_and_writes_the_csv(tmp_path, two_items):
    cloud = FakeCloud()
    result = execute(cloud_plan(tmp_path), client=cloud, confirm=lambda summary: True, log=lambda m: None, sleep=lambda s: None)
    assert result["ok"] == 2 and result["failed"] == 0 and cloud.submitted == [["a1-0", "a1-1"]]
    assert result["csv"].endswith("results.csv")
    again = FakeCloud()
    execute(cloud_plan(tmp_path), client=again, confirm=lambda summary: True, log=lambda m: None, sleep=lambda s: None)
    assert again.submitted == []  # a rerun of the same plan pays for nothing


def test_duplicate_item_ids_are_refused_because_results_are_keyed_by_id(tmp_path, monkeypatch):
    twins = [Item(0, "RES-0", "s", "a", 100), Item(1, "RES-0", "s", "b", 100)]
    monkeypatch.setattr(runner, "build_items", lambda *a, **k: twins)
    with pytest.raises(RuntimeError, match="duplicate item ids"):
        execute(cloud_plan(tmp_path), client=FakeCloud(), confirm=lambda summary: True, log=lambda m: None)


def test_a_large_local_run_is_flagged_as_sequential_in_the_summary(tmp_path, monkeypatch):
    many = [Item(i, f"RES-{i}", "s", "u", 100) for i in range(600)]
    monkeypatch.setattr(runner, "build_items", lambda *a, **k: many)
    shown = []
    plan = RunPlan(task="resume_extraction", backend="local", local_model="gemma-4-12b", output_root=tmp_path)
    execute(plan, confirm=lambda summary: shown.extend(summary) or False, log=lambda m: None)
    assert any("one item at a time" in line for line in shown)


def test_a_run_of_listed_ids_is_labelled_that_way_and_passes_the_ids_on(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(runner, "build_items", lambda *a, **k: seen.update(k) or [Item(0, "RES-0", "s", "u", 100)])
    shown = []
    plan = RunPlan(task="resume_extraction", backend="local", local_model="gemma-4-12b", item_ids=("RES-0",), output_root=tmp_path)
    execute(plan, confirm=lambda summary: shown.extend(summary) or False, log=lambda m: None)
    assert seen["item_ids"] == ("RES-0",) and any("the listed ids" in line for line in shown)


def test_export_only_rebuilds_the_csv_without_a_key_a_model_or_a_confirmation(tmp_path, monkeypatch):
    items = [Item(0, "RES-0", "system prompt", "resume zero", 6144, {"resume": "the full resume text"})]
    monkeypatch.setattr(runner, "build_items", lambda *a, **k: items)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = execute(cloud_plan(tmp_path), confirm=lambda summary: pytest.fail("must not ask"), log=lambda m: None, export_only=True)
    assert result["exported_only"] is True
    text = open(result["csv"], encoding="utf-8-sig").read()
    assert "the full resume text" in text and "not run" in text


# --- JD prompt versions: a new prompt gets its own folder and never writes into an older prompt's results ---


def jd_and_rubric():
    """A JD and a rubric for it that every JD prompt version accepts, from the v2.1 package's own tests."""
    import importlib.util

    from candidate_fit import tasks

    spec = importlib.util.spec_from_file_location("jd_pkg_fixture_runner", tasks.JD_V21_PACKAGE / "tests" / "test_validator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.JD, module.good()


def jd_plan(tmp_path, **kwargs):
    return RunPlan(task="jd_extraction", backend="local", local_model="gemma-4-12b", output_root=tmp_path, **kwargs)


def test_each_jd_prompt_version_writes_to_its_own_folder(tmp_path):
    assert jd_plan(tmp_path).folder() == tmp_path / "jd_extraction_v2_3" / "local-gemma-4-12b"  # v2.3 is the default
    assert jd_plan(tmp_path, jd_prompt_version="v2.3").folder() == jd_plan(tmp_path).folder()
    assert jd_plan(tmp_path, jd_prompt_version="v2.2").folder() == tmp_path / "jd_extraction_v2_2" / "local-gemma-4-12b"
    assert jd_plan(tmp_path, jd_prompt_version="v2.1").folder() == tmp_path / "jd_extraction" / "local-gemma-4-12b"
    other = RunPlan(task="resume_extraction", backend="local", local_model="gemma-4-12b", output_root=tmp_path)
    assert other.folder() == tmp_path / "resume_extraction" / "local-gemma-4-12b"  # other tasks are unaffected


def test_a_jd_prompt_version_must_be_known_and_belongs_to_jd_extraction_only(tmp_path):
    with pytest.raises(ValueError, match="unknown JD prompt version"):
        jd_plan(tmp_path, jd_prompt_version="v9").validate()
    with pytest.raises(ValueError, match="jd_extraction task only"):
        RunPlan(task="resume_extraction", backend="local", local_model="gemma-4-12b", jd_prompt_version="v2.2").validate()
    jd_plan(tmp_path, jd_prompt_version="v2.1").validate()
    jd_plan(tmp_path).validate()


def test_the_chosen_prompt_version_reaches_the_items_the_fingerprint_and_the_summary(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(runner, "build_items", lambda *a, **k: seen.update(k) or [Item(0, "JD-1", "s", "u", 100, {"jd": "x"})])
    shown = {}
    for version in (None, "v2.1"):
        lines = []
        execute(jd_plan(tmp_path, jd_prompt_version=version), confirm=lambda summary: lines.extend(summary) or False, log=lambda m: None)
        shown[version] = (seen["jd_prompt_version"], lines)
    assert shown[None][0] is None and shown["v2.1"][0] == "v2.1"
    assert "JD prompt: v2.3" in shown[None][1] and "JD prompt: v2.1" in shown["v2.1"][1]
    assert any("jd_extraction_v2_3" in line for line in shown[None][1])  # the folder is shown before anything is confirmed
    assert not any("jd_extraction_v2_3" in line for line in shown["v2.1"][1])


class JdCloud(FakeCloud):
    """Answers every request with one fixed reply."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    def submit(self, requests, intent_id=None):
        batch_id = super().submit(requests, intent_id)
        for reply in self.batches[batch_id]:
            reply["text"] = self.text
        return batch_id


@pytest.mark.parametrize("older_folder", ["jd_extraction", "jd_extraction_v2_2"])  # where v2.1 and v2.2 wrote
def test_running_the_new_jd_prompt_leaves_the_older_prompts_results_untouched(tmp_path, monkeypatch, older_folder):
    jd, rubric = jd_and_rubric()
    old_folder = tmp_path / older_folder / "cloud-anthropic-claude-haiku-5-5"  # the same backend and model as the new run
    old_folder.mkdir(parents=True)
    old_files = {"results.jsonl": '{"item_id":"JD-1","fp":"old","ok":true}\n', "results.csv": "the old csv\n", "batch_state.json": "{}\n"}
    for name, text in old_files.items():
        (old_folder / name).write_text(text, encoding="utf-8")
    items = [Item(0, "JD-1", "system", "user", 100, {"jd": jd, "jd_prompt_version": "v2.3"})]
    monkeypatch.setattr(runner, "build_items", lambda *a, **k: items)
    plan = RunPlan(task="jd_extraction", backend="cloud", cloud_provider="anthropic", cloud_model="claude-haiku-5-5", output_root=tmp_path)
    result = execute(plan, client=JdCloud(json.dumps(rubric)), confirm=lambda summary: True, log=lambda m: None, sleep=lambda s: None)
    assert result["ok"] == 1 and result["failed"] == 0
    for name, text in old_files.items():  # the older prompt's folder is byte-for-byte what it was
        assert (old_folder / name).read_text(encoding="utf-8") == text
    new_folder = tmp_path / "jd_extraction_v2_3" / "cloud-anthropic-claude-haiku-5-5"
    assert result["csv"] == str(new_folder / "results.csv")
    header, row = open(new_folder / "results.csv", encoding="utf-8-sig").read().splitlines()[:2]
    assert "valid_for_cache" in header and '"True"' in row


def test_the_run_command_takes_a_jd_prompt_version_and_defaults_to_the_current_one(monkeypatch, capsys):
    from candidate_fit import cli
    from candidate_fit.tasks import DEFAULT_JD_PROMPT_VERSION

    plans = []
    monkeypatch.setattr("candidate_fit.runner.execute", lambda plan, **kwargs: plans.append(plan) or {"ok": 1})
    base = ["run", "--task", "jd_extraction", "--backend", "local", "--local-model", "gemma-4-12b"]
    for extra in ([], ["--jd-prompt-version", "v2.1"], ["--jd-prompt-version", "v2.3"]):
        with pytest.raises(SystemExit) as stopped:
            cli.main(base + extra)
        assert stopped.value.code == 0
    assert [p.jd_prompt_version for p in plans] == [None, "v2.1", "v2.3"]
    assert plans[0].folder().parts[-2] == "jd_extraction_" + DEFAULT_JD_PROMPT_VERSION.replace(".", "_")
    assert plans[1].folder().parts[-2] == "jd_extraction"
    with pytest.raises(SystemExit):
        cli.main(base + ["--jd-prompt-version", "v9"])  # argparse rejects a version that does not exist
    assert "invalid choice" in capsys.readouterr().err


# --- gpt-oss-20b gets its own JD prompt automatically; no other model can be given it ---


def test_a_gptoss_plan_picks_its_own_prompt_and_folder_and_every_other_models_plan_is_unchanged(tmp_path):
    gptoss = RunPlan(task="jd_extraction", backend="local", local_model="gpt-oss-20b", output_root=tmp_path)
    assert gptoss.effective_jd_prompt_version == "v2.3-gptoss"
    assert gptoss.folder() == tmp_path / "jd_extraction_v2_3_gptoss" / "local-gpt-oss-20b"
    gptoss.validate()
    for model in ("qwen3.8-27b", "gemma-4-12b", "qwen3.5-4b"):
        plan = RunPlan(task="jd_extraction", backend="local", local_model=model, output_root=tmp_path)
        assert plan.effective_jd_prompt_version is None  # the shared default, as before
        assert plan.folder() == tmp_path / "jd_extraction_v2_3" / f"local-{model}"
    cloud = RunPlan(task="jd_extraction", backend="cloud", cloud_provider="anthropic", cloud_model="m", output_root=tmp_path)
    assert cloud.effective_jd_prompt_version is None
    other_task = RunPlan(task="resume_extraction", backend="local", local_model="gpt-oss-20b", output_root=tmp_path)
    assert other_task.folder() == tmp_path / "resume_extraction" / "local-gpt-oss-20b"  # only the JD task has a model prompt
    other_task.validate()


def test_the_gptoss_prompt_is_refused_for_every_other_model_and_for_the_cloud(tmp_path):
    for plan in (RunPlan(task="jd_extraction", backend="local", local_model="qwen3.8-27b", jd_prompt_version="v2.3-gptoss"),
                 RunPlan(task="jd_extraction", backend="local", local_model="gemma-4-12b", jd_prompt_version="v2.3-gptoss"),
                 RunPlan(task="jd_extraction", backend="cloud", cloud_provider="anthropic", cloud_model="m", jd_prompt_version="v2.3-gptoss")):
        with pytest.raises(ValueError, match="written for gpt-oss-20b only"):
            plan.validate()
    RunPlan(task="jd_extraction", backend="local", local_model="gpt-oss-20b", jd_prompt_version="v2.3-gptoss").validate()
    # gpt-oss can still be asked for a shared version, for example to reproduce the earlier run
    older = RunPlan(task="jd_extraction", backend="local", local_model="gpt-oss-20b", jd_prompt_version="v2.3", output_root=tmp_path)
    older.validate()
    assert older.folder() == tmp_path / "jd_extraction_v2_3" / "local-gpt-oss-20b"


def test_the_prompt_version_passed_on_is_gptoss_s_own_for_gptoss_and_unset_for_the_others(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(runner, "build_items", lambda *a, **k: seen.update(k) or [Item(0, "JD-1", "s", "u", 100, {"jd": "x"})])
    shown = {}
    for model in ("gpt-oss-20b", "qwen3.8-27b"):
        lines = []
        plan = RunPlan(task="jd_extraction", backend="local", local_model=model, output_root=tmp_path)
        execute(plan, confirm=lambda summary: lines.extend(summary) or False, log=lambda m: None)
        shown[model] = (seen["jd_prompt_version"], lines)
    assert shown["gpt-oss-20b"][0] == "v2.3-gptoss" and "JD prompt: v2.3-gptoss" in shown["gpt-oss-20b"][1]
    assert shown["qwen3.8-27b"][0] is None and "JD prompt: v2.3" in shown["qwen3.8-27b"][1]


def test_the_run_command_gives_gptoss_its_prompt_without_a_flag_and_leaves_qwen_on_the_default(monkeypatch):
    from candidate_fit import cli

    plans = []
    monkeypatch.setattr("candidate_fit.runner.execute", lambda plan, **kwargs: plans.append(plan) or {"ok": 1})
    for model in ("gpt-oss-20b", "qwen3.8-27b"):
        with pytest.raises(SystemExit):
            cli.main(["run", "--task", "jd_extraction", "--backend", "local", "--local-model", model])
    assert [p.effective_jd_prompt_version for p in plans] == ["v2.3-gptoss", None]
    assert plans[0].folder().parts[-2] == "jd_extraction_v2_3_gptoss" and plans[1].folder().parts[-2] == "jd_extraction_v2_3"


def test_the_26b_gemma_plan_picks_the_gemma_rules_and_no_other_models_plan_changes(tmp_path):
    gemma = RunPlan(task="jd_extraction", backend="local", local_model="gemma-4-26b-a4b", output_root=tmp_path)
    gemma.validate()
    assert gemma.effective_jd_prompt_version == "v2.3-gemma26"
    assert gemma.folder() == tmp_path / "jd_extraction_v2_3_gemma26" / "local-gemma-4-26b-a4b"
    for model in ("qwen3.8-27b", "gemma-4-12b", "qwen3.5-4b"):  # the others keep the shared v2.3 and its folder
        plan = RunPlan(task="jd_extraction", backend="local", local_model=model, output_root=tmp_path)
        assert plan.effective_jd_prompt_version is None and plan.folder() == tmp_path / "jd_extraction_v2_3" / f"local-{model}"
    # the 26B can still be asked for the plain shared prompt, for a before-and-after comparison
    plain = RunPlan(task="jd_extraction", backend="local", local_model="gemma-4-26b-a4b", jd_prompt_version="v2.3", output_root=tmp_path)
    plain.validate()
    assert plain.folder() == tmp_path / "jd_extraction_v2_3" / "local-gemma-4-26b-a4b"


def test_the_gemma_rules_are_refused_for_every_other_model_and_for_the_cloud():
    for plan in (RunPlan(task="jd_extraction", backend="local", local_model="qwen3.8-27b", jd_prompt_version="v2.3-gemma26"),
                 RunPlan(task="jd_extraction", backend="local", local_model="gemma-4-12b", jd_prompt_version="v2.3-gemma26"),
                 RunPlan(task="jd_extraction", backend="local", local_model="gpt-oss-20b", jd_prompt_version="v2.3-gemma26"),
                 RunPlan(task="jd_extraction", backend="cloud", cloud_provider="anthropic", cloud_model="m", jd_prompt_version="v2.3-gemma26")):
        with pytest.raises(ValueError, match="written for gemma-4-26b-a4b only"):
            plan.validate()
