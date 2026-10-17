import json
from pathlib import Path
import sys

import pytest

from candidate_fit.client import Completion
from candidate_fit.dataset import PairRecord

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_eval_v3 as v3  # noqa: E402

REFERENCE_DATE = "2026-10-17"
PROMPTS = {key: f"{key} {{REFERENCE_DATE}} {{RESUME_TEXT}} {{JD_TEXT}} {{JD_EXTRACTION_JSON}} {{RESUME_EXTRACTION_JSON}}" for key in (
    "resume_system", "resume_user", "jd_system", "jd_user", "match_system", "match_user")}


def fixture():
    jd = {
        "input_status": "ok", "seniority": "mid",
        "skills": [{"id": "S1", "name": "Python", "priority": "mandatory", "importance": "critical", "hard_gate_explicit": True, "any_of_group": None}],
        "any_of_groups": [], "key_responsibilities": [{"id": "D1", "importance": "core"}], "requirements": [],
        "experience_requirement": {"min_years": 5, "basis": "relevant", "priority": "mandatory", "hard_gate_explicit": True},
        "education_requirement": {"min_level": None, "priority": None, "hard_gate_explicit": False},
    }
    resume = {
        "input_status": "ok", "suspicious_instructions_detected": False,
        "experience_summary": {"date_quality": "complete"}, "skills": [{"name": "Python"}],
        "work_experiences": [{"id": "E1", "employment_type": "full_time", "start": "2020-01", "end": "2024-12"}],
        "projects": [], "education": [], "certifications": [],
    }
    match = {
        "input_check": {"status": "ok"},
        "skill_matches": [{"jd_skill_id": "S1", "status": "met", "match_type": "exact", "resume_skill": "Python",
                           "resume_refs": ["E1"], "modifiers": [], "match_score": 10}],
        "group_matches": [], "responsibility_matches": [{"jd_responsibility_id": "D1", "status": "demonstrated"}],
        "requirement_matches": [], "education_match": {"status": "not_applicable"},
        "experience_relevance": [{"resume_experience_id": "E1", "relevance": "high"}],
        "alignment": {"role": "exact", "functional": "exact", "seniority_fit": "match", "industry": "not_applicable"},
        "needs_human_review": False, "confidence": 0.9, "rationale": "Five relevant years of Python.",
    }
    return jd, resume, match


class FakeClient:
    model = "fake-model"
    lane = "r9700"

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def complete(self, user_prompt, *, system_prompt, response_schema, stage):
        self.calls.append(stage)
        reply = self.replies[stage]
        reply = reply.pop(0) if isinstance(reply, list) else reply
        return Completion(content=reply, usage={"input_tokens": 100, "output_tokens": 50}, latency_seconds=2.0, finish_reason="stop")


def pair():
    return PairRecord(pair_id="p:1", source_file="x", row_index=1, resume="Python developer", job_description="Python role",
                      gold_label="1", metadata={"selection": "smallest_1"})


def run(client, tmp_path):
    return v3.process_pair(client, pair(), prompts=PROMPTS, cache={}, cache_path=tmp_path / "cache.jsonl", reference_date=REFERENCE_DATE)


def test_final_score_comes_from_the_calculator_not_the_model(tmp_path):
    jd, resume, match = fixture()
    match["final_score"] = 1  # a model-supplied score must be discarded
    client = FakeClient({"resume_extraction": json.dumps(resume), "jd_extraction": "```json\n" + json.dumps(jd) + "\n```",
                         "relevancy_matching": json.dumps(match)})
    record = run(client, tmp_path)
    assert record["status"] == "ok"
    assert record["result"]["final_score"] == 10
    assert record["result"]["fit_band"] == "strong_fit"
    assert [s["stage"] for s in record["stages"]] == ["resume_extraction", "jd_extraction", "relevancy_matching"]
    assert (tmp_path / "cache.jsonl").read_text(encoding="utf-8").count("\n") == 2
    reasoning = v3.build_reasoning(record)
    assert "Skills 10/10 x 35% = 3.50" in reasoning and "Final = 10" in reasoning and "Five relevant years" in reasoning


def test_invalid_match_is_retried_then_never_scored(tmp_path):
    jd, resume, match = fixture()
    bad = dict(match, skill_matches=[])
    client = FakeClient({"resume_extraction": json.dumps(resume), "jd_extraction": json.dumps(jd),
                         "relevancy_matching": [json.dumps(bad), json.dumps(bad)]})
    record = run(client, tmp_path)
    assert record["status"] == "error" and record["failed_stage"] == "relevancy_matching"
    assert "final_score" not in record and client.calls.count("relevancy_matching") == 2
    assert "No score assigned" in v3.build_reasoning(record)


def test_coverage_error_names_the_missing_ids():
    jd, resume, match = fixture()
    jd["key_responsibilities"].append({"id": "D2", "importance": "secondary"})
    jd["requirements"] = [{"id": "R1", "category": "other", "priority": "preferred"}]
    with pytest.raises(ValueError) as caught:
        v3.check_coverage(jd, resume, match)
    assert "responsibility entries missing: D2" in str(caught.value)
    assert "requirement entries missing: R1" in str(caught.value)


def test_score_band_error_names_the_skill():
    jd, resume, match = fixture()
    match["skill_matches"][0].update(status="partial", match_score=0)
    with pytest.raises(ValueError, match=r"S1 status partial needs match_score 3-6 \(got 0\)"):
        v3.check_coverage(jd, resume, match)


def test_insufficient_input_check_on_usable_documents_is_rejected():
    jd, resume, match = fixture()
    resume["input_status"] = "partial"
    match["input_check"] = {"status": "insufficient_resume", "notes": "dates missing"}
    with pytest.raises(ValueError, match="input_check.status must be 'ok'"):
        v3.check_coverage(jd, resume, match)


def test_unknown_resume_ref_lists_the_allowed_ids():
    jd, resume, match = fixture()
    match["skill_matches"][0]["resume_refs"] = ["E9"]
    with pytest.raises(ValueError) as caught:
        v3.check_coverage(jd, resume, match)
    assert "S1 cites unknown resume_refs E9" in str(caught.value) and "E1" in str(caught.value)


def test_retry_recovers_from_a_bad_first_reply(tmp_path):
    jd, resume, match = fixture()
    client = FakeClient({"resume_extraction": json.dumps(resume), "jd_extraction": json.dumps(jd),
                         "relevancy_matching": ["not json", json.dumps(match)]})
    record = run(client, tmp_path)
    assert record["status"] == "ok" and record["stages"][-1]["attempts"] == 2


def test_unreadable_resume_is_unscorable_and_skips_matching(tmp_path):
    jd, resume, _ = fixture()
    resume["input_status"] = "empty_or_unreadable"
    client = FakeClient({"resume_extraction": json.dumps(resume), "jd_extraction": json.dumps(jd)})
    record = run(client, tmp_path)
    assert record["status"] == "ok"
    assert record["result"]["final_score"] is None and record["result"]["fit_band"] == "unscorable"
    assert "relevancy_matching" not in client.calls
    assert "Unscorable" in v3.build_reasoning(record)


def test_parse_first_json_object_ignores_trailing_text():
    assert v3.parse_first_json_object('```json\n{"a": 1}\n```\nextra {"b": 2}') == {"a": 1}
    with pytest.raises(ValueError):
        v3.parse_first_json_object("no json here")


def test_stage_thinking_tokens_are_summed_across_retries():
    from candidate_fit.client import Completion as _Completion

    class Reporting:
        model = "fake"
        lane = "r9700"

        def __init__(self):
            self.usages = [
                {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 3},
                {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 4},
            ]
            self.replies = ["not json", json.dumps({"ok": True})]

        def complete(self, user_prompt, *, system_prompt, response_schema, stage):
            return _Completion(content=self.replies.pop(0), usage=self.usages.pop(0), latency_seconds=1.0, finish_reason="stop")

    _value, _raw, metrics = v3.run_stage(Reporting(), stage="s", system_prompt="p", user_prompt="u", validate=lambda v: None)
    assert metrics["thinking_tokens"] == 7 and metrics["attempts"] == 2
