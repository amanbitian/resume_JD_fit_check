import json
import threading
import time

import pytest

from candidate_fit.client import Completion
from candidate_fit.dataset import PairRecord
from candidate_fit.optimized import (
    JD_ANALYSIS_PROMPT,
    PAIR_SCORING_PROMPT,
    RESUME_ANALYSIS_PROMPT,
    _assemble,
    _validate_extraction,
    score_dataset_optimized,
)


def test_optimized_assembly_expands_indexes_and_derives_mandatory_gaps():
    jd = {
        "job_title": "Engineer",
        "all_jd_skills": [{"skill": "Python", "priority": "mandatory", "importance": "critical"}],
        "requirements": [
            {"category": "skill", "requirement": "Python", "priority": "mandatory", "importance": "critical"}
        ],
        "top_critical_skills": [{"skill": "Python"}],
    }
    resume = {
        "primary_functional_domain": "software_engineering",
        "total_years_experience": 3,
        "work_experiences": [{"role_title": "Developer", "employer": "Acme", "evidence": "Python work"}],
    }
    pair = {
        "c": 0.8,
        "d": {"s": 8, "y": 6, "e": 8, "w": 8, "r": 8, "g": 8, "o": None},
        "ry": 2,
        "ra": {"l": "strong_adjacent", "e": "Similar role"},
        "da": {"l": "exact", "e": "Software domain"},
        "rr": [{"i": 0, "r": "medium", "x": "Some overlap"}],
        "sa": [{"i": 0, "s": "partial", "e": "Limited evidence", "y": 2, "m": 5}],
        "rc": [{"i": 0, "s": "partial", "e": "Limited evidence"}],
        "h": False,
        "hr": [],
        "r": "Potential fit.",
    }
    result = _assemble(jd, resume, pair)
    assert result["fit_score"] == 6
    assert result["fit_label"] == "Potential Fit"
    assert result["candidate_analysis"]["work_experiences"][0]["relevance_to_jd"] == "medium"
    assert result["skill_assessment"][0]["skill"] == "Python"
    assert result["unmet_mandatory_requirements"] == ["Python"]
    assert result["gaps"][0]["requirement"] == "Python"
    assert result["hard_requirements_met"] is False
    assert result["score_method"]["applied_caps"] == ["unmet_critical_mandatory_requirement"]


def test_optimized_assembly_rejects_missing_indexes():
    jd = {"all_jd_skills": [{"skill": "Python"}], "requirements": []}
    resume = {"work_experiences": []}
    pair = {"s": 5, "sa": [], "rc": [], "rr": []}
    try:
        _assemble(jd, resume, pair)
    except ValueError as exc:
        assert "omitted skill indices" in str(exc)
    else:
        raise AssertionError("Missing indexed assessments must be rejected")


def test_optimized_assembly_rejects_invalid_status_instead_of_silently_accepting_it():
    jd = {
        "all_jd_skills": [{"skill": "Python", "priority": "mandatory", "importance": "critical"}],
        "requirements": [
            {"category": "skill", "requirement": "Python", "priority": "mandatory", "importance": "critical"}
        ],
    }
    resume = {"work_experiences": []}
    pair = {
        "c": 0.8,
        "d": {"s": 5, "y": 5, "e": 5, "w": 5, "r": 5, "g": 5, "o": None},
        "ra": {"l": "unclear", "e": "No role evidence"},
        "da": {"l": "unclear", "e": "No domain evidence"},
        "rr": [],
        "sa": [{"i": 0, "s": "probably", "e": "", "y": None, "m": 5}],
        "rc": [{"i": 0, "s": "met", "e": "Python work"}],
        "h": True,
        "hr": ["insufficient evidence"],
    }
    with pytest.raises(ValueError, match="Invalid skill 0.status"):
        _assemble(jd, resume, pair)


def test_optimized_assembly_applies_critical_mandatory_score_caps():
    jd = {
        "all_jd_skills": [
            {"skill": "Python", "priority": "mandatory", "importance": "critical"},
            {"skill": "SQL", "priority": "mandatory", "importance": "critical"},
        ],
        "requirements": [
            {"category": "skill", "requirement": "Python", "priority": "mandatory", "importance": "critical"},
            {"category": "skill", "requirement": "SQL", "priority": "mandatory", "importance": "critical"},
        ],
    }
    resume = {"total_years_experience": 8, "work_experiences": []}
    pair = {
        "c": 0.8,
        "d": {"s": 8, "y": 9, "e": None, "w": 8, "r": 8, "g": 8, "o": None},
        "ry": 7,
        "ra": {"l": "exact", "e": "Same role"},
        "da": {"l": "exact", "e": "Same domain"},
        "rr": [],
        "sa": [
            {"i": 0, "s": "not_evidenced", "e": "No evidence", "y": None, "m": 0},
            {"i": 1, "s": "not_evidenced", "e": "No evidence", "y": None, "m": 0},
        ],
        "rc": [
            {"i": 0, "s": "not_evidenced", "e": "No evidence"},
            {"i": 1, "s": "not_evidenced", "e": "No evidence"},
        ],
        "h": False,
        "hr": [],
        "r": "Strong general background but both critical skills are not evidenced.",
    }

    result = _assemble(jd, resume, pair)
    assert result["fit_score"] == 4
    assert result["fit_label"] == "Potential Fit"
    assert result["score_method"]["applied_caps"] == [
        "two_or_more_unmet_critical_mandatory_requirements"
    ]


def test_extraction_rejects_freeform_domains():
    resume = {
        "primary_functional_domain": "Accounting",
        "secondary_functional_domains": [],
        "primary_industry_domain": "Utilities",
        "secondary_industry_domains": [],
        "work_experiences": [],
    }
    with pytest.raises(ValueError, match="primary_functional_domain"):
        _validate_extraction("resume", resume)


def test_parallel_optimized_scoring_analyzes_shared_entities_once(tmp_path):
    jd = {
        "job_title": "Engineer",
        "role_family": "Software Engineering",
        "seniority_level": "mid",
        "primary_functional_domain": "software_engineering",
        "secondary_functional_domains": [],
        "functional_subdomains": ["backend"],
        "primary_industry_domain": "technology_software",
        "secondary_industry_domains": [],
        "industry_subdomains": ["SaaS"],
        "domain_rationale": "Builds software services.",
        "minimum_years_experience": 2,
        "education_requirements": [],
        "certification_requirements": [],
        "all_jd_skills": [
            {
                "skill": "Python",
                "skill_type": "technical",
                "priority": "mandatory",
                "importance": "critical",
                "jd_evidence": "Python required",
            }
        ],
        "top_critical_skills": [
            {"skill": "Python", "why_critical": "Core implementation", "jd_evidence": "Python required"}
        ],
        "requirements": [
            {
                "category": "skill",
                "requirement": "Python",
                "priority": "mandatory",
                "importance": "critical",
                "jd_evidence": "Python required",
            }
        ],
    }
    resume = {
        "primary_role": "Software Engineer",
        "current_or_most_recent_role": "Software Engineer",
        "highest_demonstrated_seniority": "mid",
        "primary_functional_domain": "software_engineering",
        "secondary_functional_domains": [],
        "functional_subdomains": ["backend"],
        "primary_industry_domain": "technology_software",
        "secondary_industry_domains": [],
        "industry_subdomains": ["SaaS"],
        "domain_rationale": "Built backend services.",
        "total_years_experience": 3,
        "highest_education": "Not evidenced",
        "certifications": [],
        "work_experiences": [
            {
                "role_title": "Software Engineer",
                "employer": "Acme",
                "start_date": "2022",
                "end_date": "2025",
                "duration_years": 3,
                "functional_domains": ["software_engineering"],
                "industry_domains": ["technology_software"],
                "skills_evidenced": ["Python"],
                "responsibilities_evidenced": ["Built APIs"],
                "achievements_evidenced": [],
                "evidence": "Built Python APIs.",
            }
        ],
    }
    pair = {
        "c": 0.9,
        "d": {"s": 9, "y": 9, "w": 9, "r": 9, "g": 9, "e": None, "o": None},
        "ry": 3,
        "ra": {"l": "exact", "e": "Same role"},
        "da": {"l": "exact", "e": "Same domain"},
        "rr": [{"i": 0, "r": "high", "x": "Direct experience"}],
        "sa": [{"i": 0, "s": "met", "e": "Built Python APIs", "y": 3, "m": 9}],
        "rc": [{"i": 0, "s": "met", "e": "Built Python APIs"}],
        "h": False,
        "hr": [],
        "r": "Directly evidenced fit.",
    }

    class FakeClient:
        model = "fake-model"
        base_url = "http://fake"
        max_tokens = 1000
        generation_config = {"temperature": 0}

        def __init__(self):
            self.calls = {"jd": 0, "resume": 0, "pair": 0}
            self.lock = threading.Lock()

        def complete(self, _user_prompt, *, system_prompt, response_schema, stage=None):
            if system_prompt == JD_ANALYSIS_PROMPT:
                kind, payload = "jd", jd
            elif system_prompt == RESUME_ANALYSIS_PROMPT:
                kind, payload = "resume", resume
            elif system_prompt == PAIR_SCORING_PROMPT:
                kind, payload = "pair", pair
            else:  # pragma: no cover - defensive assertion
                raise AssertionError("Unexpected prompt")
            with self.lock:
                self.calls[kind] += 1
            time.sleep(0.01)
            return Completion(content=json.dumps(payload), usage={}, latency_seconds=0.01)

    records = [
        PairRecord("pair-1", "data.parquet", 0, "same resume", "same jd", None, {}),
        PairRecord("pair-2", "data.parquet", 1, "same resume", "same jd", None, {}),
    ]
    client = FakeClient()
    counters = score_dataset_optimized(
        records,
        client=client,  # type: ignore[arg-type]
        output_path=tmp_path / "results.jsonl",
        cache_path=tmp_path / "cache.jsonl",
        workers=2,
    )

    assert counters["scored"] == 2
    assert counters["jd_cache_misses"] == 1
    assert counters["resume_cache_misses"] == 1
    assert client.calls == {"jd": 1, "resume": 1, "pair": 2}
