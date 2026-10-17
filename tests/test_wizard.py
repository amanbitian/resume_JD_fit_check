from candidate_fit import wizard
from candidate_fit.runner import RunPlan


def scripted(answers):
    """A reader that returns the given answers in order and fails if the wizard asks more than planned."""
    queue = list(answers)

    def read(prompt):
        assert queue, f"wizard asked more questions than planned: {prompt!r}"
        return queue.pop(0)

    read.left = queue
    return read


def silent(_message):
    return None


def test_local_resume_extraction_plan_from_menu_answers():
    reader = scripted(["1", "1", "1", "", ""])  # task 1, local, Qwen3.8-27B, do not replace another model, all items
    plan = wizard.build_plan(reader, silent, environ={})
    assert plan == RunPlan(task="resume_extraction", backend="local", local_model="qwen3.8-27b", limit=None,
                           seed=42, replace_other_model=False)
    assert reader.left == []


def test_local_plan_can_agree_to_replace_another_model():
    plan = wizard.build_plan(scripted(["2", "1", "2", "y", ""]), silent, environ={})
    assert plan.local_model == "gemma-4-12b" and plan.replace_other_model is True


def test_cloud_pair_plan_uses_the_entered_model_id_and_sample_settings():
    reader = scripted(["3", "2", "1", "claude-haiku-5-5", "500", "7"])  # pair, cloud, Anthropic, id, 500 items, seed 7
    plan = wizard.build_plan(reader, silent, environ={"ANTHROPIC_API_KEY": "k"})
    assert plan.task == "resume_jd_pair" and plan.backend == "cloud" and plan.cloud_provider == "anthropic"
    assert plan.cloud_model == "claude-haiku-5-5" and plan.limit == 500 and plan.seed == 7
    assert plan.reasoning_effort is None  # Anthropic has no reasoning-effort question


def test_openai_plan_asks_for_a_reasoning_effort():
    reader = scripted(["2", "2", "2", "gpt-5.6-luna", "3", ""])  # JD, cloud, OpenAI, id, effort 3rd = low, all items
    plan = wizard.build_plan(reader, silent, environ={"OPENAI_API_KEY": "k"})
    assert plan.cloud_provider == "openai" and plan.reasoning_effort == "low"
    skipped = wizard.build_plan(scripted(["2", "2", "2", "gpt-5.6-luna", "1", ""]), silent, environ={"OPENAI_API_KEY": "k"})
    assert skipped.reasoning_effort is None


def test_cloud_plan_stops_with_a_message_when_the_key_is_missing():
    messages = []
    reader = scripted(["2", "2", "2", "gpt-5.6-luna"])
    assert wizard.build_plan(reader, messages.append, environ={}) is None
    assert any("OPENAI_API_KEY" in m for m in messages)


def test_invalid_menu_answers_are_asked_again():
    messages = []
    reader = scripted(["9", "x", "2", "2", "1", "", "3", ""])  # two bad task answers, then JD, cloud, Anthropic
    plan = wizard.build_plan(reader, messages.append, environ={"ANTHROPIC_API_KEY": "k"})
    assert plan.task == "jd_extraction" and plan.backend == "cloud" and plan.limit == 3
    assert sum("Please enter one of the numbers" in m for m in messages) == 2


def test_cloud_is_only_offered_as_batch():
    menus = []
    reader = scripted(["1", "2", "1", "claude-haiku-5-5", ""])
    wizard.build_plan(reader, menus.append, environ={"ANTHROPIC_API_KEY": "k"})
    text = "\n".join(menus)
    assert "batch API only" in text and "sync" not in text.lower()


def fake_execute(captured):
    def execute(plan, *, confirm, log, sleep=None, clock=None):
        captured["asked"] = confirm(["Task: x", "Data: sent to anthropic servers."])
        return {"ok": 1} if captured["asked"] else {"cancelled": True}

    return execute


def test_a_cloud_run_needs_both_the_data_acknowledgement_and_the_submit_confirmation(monkeypatch):
    plan_answers = ["1", "2", "1", "claude-haiku-5-5", "", ]
    env = {"ANTHROPIC_API_KEY": "k"}
    for answers, expected in ((["n"], {"cancelled": True}), (["y", "n"], {"cancelled": True}), (["y", "y"], {"ok": 1})):
        captured = {}
        monkeypatch.setattr(wizard, "execute", fake_execute(captured))
        result = wizard.run_wizard(scripted(plan_answers + answers), silent, env)
        assert result == expected


def test_a_local_run_asks_one_question_and_defaults_to_yes(monkeypatch):
    captured = {}
    monkeypatch.setattr(wizard, "execute", fake_execute(captured))
    result = wizard.run_wizard(scripted(["1", "1", "1", "", "", ""]), silent, {})  # last "" accepts the default
    assert result == {"ok": 1} and captured["asked"] is True
