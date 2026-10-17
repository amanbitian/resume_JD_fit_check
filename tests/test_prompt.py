import pytest

from candidate_fit.prompt import compact_text, label_for_score, normalize_prediction, parse_json_object


@pytest.mark.parametrize(
    ("score", "label"),
    [(0, "No Fit"), (3, "No Fit"), (4, "Potential Fit"), (6, "Potential Fit"), (7, "Good Fit"), (10, "Good Fit")],
)
def test_label_bands(score, label):
    assert label_for_score(score) == label


def test_parse_fenced_json_and_normalize_inconsistent_label():
    value = parse_json_object('```json\n{"fit_score": 8, "fit_label": "No Fit", "confidence": 1.2}\n```')
    normalized = normalize_prediction(value)
    assert normalized["fit_score"] == 8
    assert normalized["fit_label"] == "Good Fit"
    assert normalized["confidence"] == 1.0


def test_compact_text_keeps_head_and_tail():
    text = "HEAD" + ("x" * 1_000) + "TAIL"
    result = compact_text(text, 200)
    assert result.startswith("HEAD")
    assert result.endswith("TAIL")
    assert len(result) == 200


def test_hard_requirement_flag_is_derived_from_unmet_list():
    normalized = normalize_prediction(
        {
            "fit_score": 5,
            "unmet_mandatory_requirements": ["Required Yardi experience"],
        }
    )
    assert normalized["hard_requirements_met"] is False


def test_normalize_retains_all_audit_analysis_fields():
    normalized = normalize_prediction(
        {
            "fit_score": 6,
            "jd_analysis": {"primary_functional_domain": "software_engineering"},
            "candidate_analysis": {
                "primary_functional_domain": "data_analytics",
                "work_experiences": [{"role_title": "Analyst"}],
            },
            "skill_assessment": [{"skill": "Python", "status": "met"}],
        }
    )
    assert normalized["jd_analysis"]["primary_functional_domain"] == "software_engineering"
    assert normalized["candidate_analysis"]["work_experiences"][0]["role_title"] == "Analyst"
    assert normalized["skill_assessment"] == [{"skill": "Python", "status": "met"}]


def test_compact_wire_format_expands_to_public_contract():
    normalized = normalize_prediction(
        {
            "s": 8,
            "c": 0.9,
            "j": {
                "t": "Engineer", "lv": "senior", "pf": "software_engineering", "sf": [],
                "pi": "technology_software", "si": [], "dr": "Builds software", "my": 5,
                "ed": ["Bachelor's"],
                "sk": [{"n": "Python", "t": "technical", "p": "mandatory", "e": "Python required"}],
                "cs": [{"n": "Python", "w": "Core implementation", "e": "Python required"}],
            },
            "a": {
                "pf": "software_engineering", "sf": [], "pi": "technology_software", "si": [],
                "dr": "Software work", "ty": 7, "ry": 6, "ed": "Bachelor's", "ce": [],
                "we": [{"t": "Developer", "e": "Acme", "sd": "2020", "ed": "present", "y": 6,
                        "fd": ["software_engineering"], "id": ["technology_software"],
                        "sk": ["Python"], "r": "high", "x": "Directly relevant"}],
            },
            "d": {"s": 9, "y": 8, "e": 8, "w": 9},
            "sa": [{"n": "Python", "p": "mandatory", "s": "met", "e": "Developer role", "y": 6, "m": 9}],
            "rc": [{"c": "skill", "r": "Python", "p": "mandatory", "s": "met", "e": "Developer role"}],
            "u": [],
            "r": "Strong fit.",
        }
    )
    assert normalized["fit_score"] == 8
    assert normalized["fit_label"] == "Good Fit"
    assert normalized["jd_analysis"]["all_jd_skills"][0]["skill"] == "Python"
    assert normalized["candidate_analysis"]["work_experiences"][0]["role_title"] == "Developer"
    assert normalized["skill_assessment"][0]["match_score"] == 9

