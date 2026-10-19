import copy
import csv
import dataclasses
import importlib.util
import json

import pytest

from candidate_fit import tasks
from candidate_fit.tasks import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    Item,
    ValidationError,
    build_items,
    jd_prompt,
    jd_user_message,
    load_item_ids,
    output_task_dir,
    parse_strict_object,
    result_columns,
    task_fingerprint,
    validate_reply,
)


def write_csv(path, rows, id_col, text_col):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[id_col, text_col])
        writer.writeheader()
        writer.writerows({id_col: i, text_col: t} for i, t in rows)
    return path


SHARED_VERSIONS = ("v2.1", "v2.2", "v2.3")  # the versions every model uses; v2.3-gptoss is for gpt-oss-20b only


def package_fixture():
    """The valid rubric and its JD from the v2.1 package's own tests (it is valid under v2.2 as well)."""
    spec = importlib.util.spec_from_file_location("jd_pkg_fixture", tasks.JD_V21_PACKAGE / "tests" / "test_validator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.JD, module.good


def saved_v21_outputs():
    """The 12 rubrics the v2.1 prompt produced, each with its JD, as shipped with the v2.2 package for regression tests."""
    csv.field_size_limit(10**9)
    with (tasks.JD_V22_PACKAGE / "tests" / "fixtures" / "results_full_text.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {row["item_id"]: (row["jd"], row["output"]) for row in csv.DictReader(handle)}


def v22_item(jd, item_id="JD-1"):
    return Item(0, item_id, "s", "u", 100, {"jd": jd, "jd_prompt_version": "v2.2"})


RESUME_REPLY = {"input_status": "ok", "work_experiences": [{"id": "E1"}], "skills": [], "projects": [],
                "education": [], "certifications": []}


def test_strict_parse_accepts_one_object_with_optional_fence_and_rejects_trailing_text():
    assert parse_strict_object('```json\n{"a": 1}\n```') == {"a": 1}
    for bad in ('{"a": 1} trailing words', "[1, 2]", "not json", ""):
        with pytest.raises(ValidationError):
            parse_strict_object(bad)


def test_jd_items_use_the_package_input_layout_and_keep_the_jd_for_validation(tmp_path):
    source = write_csv(tmp_path / "jd.csv", [(f"JD-{i}", f"job text {i}") for i in range(20)], "jd_id", "jd_text")
    items = build_items("jd_extraction", source=source, limit=5, seed=7)
    again = build_items("jd_extraction", source=source, limit=5, seed=7)
    ids = [i.item_id for i in items]
    assert len(items) == 5 and ids == [i.item_id for i in again]
    assert ids == sorted(ids, key=lambda x: int(x.split("-")[1]))  # file order, so results line up between runs
    first = items[0]
    assert first.user == jd_user_message(first.context["jd"], "2026-10-17")
    assert first.user.startswith("REFERENCE_DATE: 2026-10-17\n<JOB_DESCRIPTION>\n") and first.user.endswith("\n</JOB_DESCRIPTION>")
    assert first.system.startswith("JD RUBRIC EXTRACTOR") and "OUTPUT CONTRACT" in first.system
    assert "v2.3" in first.system.splitlines()[0] and first.context["jd_prompt_version"] == "v2.3"  # v2.3 is the default
    assert first.system == tasks.JD_PROMPTS["v2.3"].full_prompt.read_text(encoding="utf-8")


def test_jd_extraction_gets_a_larger_output_cap_than_before():
    assert DEFAULT_MAX_OUTPUT_TOKENS["jd_extraction"] == 12288  # 6,144 cut off the longest JDs


def test_resume_items_fill_the_reference_date_and_resume_text(tmp_path):
    source = write_csv(tmp_path / "resume.csv", [("RES-1", "Python engineer, 3 years")], "resume_id", "resume_text")
    (item,) = build_items("resume_extraction", source=source, reference_date="2026-10-17")
    assert "2026-10-17" in item.user and "Python engineer, 3 years" in item.user and "{RESUME_TEXT}" not in item.user
    assert item.context == {"resume": "Python engineer, 3 years"}  # the raw text is kept for the CSV


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        build_items("bogus")


def test_specific_items_can_be_selected_by_id_in_file_order(tmp_path):
    source = write_csv(tmp_path / "jd.csv", [(f"JD-{i}", f"job {i}") for i in range(10)], "jd_id", "jd_text")
    items = build_items("jd_extraction", source=source, item_ids=["JD-7", "JD-2", "JD-5"])
    assert [i.item_id for i in items] == ["JD-2", "JD-5", "JD-7"] and [i.index for i in items] == [0, 1, 2]
    with pytest.raises(ValueError, match="not in the dataset"):
        build_items("jd_extraction", source=source, item_ids=["JD-2", "JD-404"])
    with pytest.raises(ValueError, match="either a list of item ids or a limit"):
        build_items("jd_extraction", source=source, item_ids=["JD-2"], limit=3)


def test_item_ids_load_from_json_objects_json_strings_or_plain_text(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([{"jd_id": "JD-1", "selection": "longest"}, {"jd_id": "JD-2"}]), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(["JD-3", "JD-4"]), encoding="utf-8")
    (tmp_path / "c.txt").write_text("# comment\nJD-5\n\nJD-6\n", encoding="utf-8")
    assert load_item_ids(tmp_path / "a.json") == ("JD-1", "JD-2")
    assert load_item_ids(tmp_path / "b.json") == ("JD-3", "JD-4")
    assert load_item_ids(tmp_path / "c.txt") == ("JD-5", "JD-6")
    (tmp_path / "d.json").write_text(json.dumps([{"nothing": 1}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_item_ids(tmp_path / "d.json")


def test_fingerprint_changes_with_the_task_the_cap_and_the_reference_date_and_not_otherwise():
    base = task_fingerprint("jd_extraction")
    assert task_fingerprint("jd_extraction") == base
    assert task_fingerprint("jd_extraction", max_output_tokens=999) != base
    assert task_fingerprint("jd_extraction", reference_date="2027-01-01") != base
    assert len({task_fingerprint(t) for t in tasks.TASK_NAMES}) == 3


@pytest.mark.parametrize("version", ["v2.1", "v2.2", "v2.3", "v2.3-gptoss"])
@pytest.mark.parametrize("part", ["full_prompt", "schema", "validator"])
def test_fingerprint_covers_the_jd_prompt_schema_and_validator_of_each_version(monkeypatch, tmp_path, version, part):
    base = task_fingerprint("jd_extraction", jd_prompt_version=version)
    edited = tmp_path / f"{version}-{part}.txt"
    edited.write_text(getattr(tasks.JD_PROMPTS[version], part).read_text(encoding="utf-8") + "\n# edited", encoding="utf-8")
    monkeypatch.setitem(tasks.JD_PROMPTS, version, dataclasses.replace(tasks.JD_PROMPTS[version], **{part: edited}))
    assert task_fingerprint("jd_extraction", jd_prompt_version=version) != base


def test_each_jd_prompt_version_has_its_own_fingerprint_and_output_folder_and_older_ones_keep_theirs():
    fingerprints = {v: task_fingerprint("jd_extraction", jd_prompt_version=v) for v in tasks.JD_PROMPTS}
    assert len(set(fingerprints.values())) == len(tasks.JD_PROMPTS) == 5
    assert task_fingerprint("jd_extraction") == fingerprints["v2.3"]  # v2.3 is the default
    assert output_task_dir("jd_extraction") == output_task_dir("jd_extraction", "v2.3") == "jd_extraction_v2_3"
    # Results already saved live in these folders under these fingerprints. If either changes on purpose, those results
    # can no longer be exported or resumed, so they are pinned.
    assert output_task_dir("jd_extraction", "v2.1") == "jd_extraction"
    assert fingerprints["v2.1"] == "5db430aa667826cc"
    assert output_task_dir("jd_extraction", "v2.2") == "jd_extraction_v2_2"
    assert fingerprints["v2.2"] == "86914dc10f00b4d8"
    assert output_task_dir("resume_extraction") == "resume_extraction" and output_task_dir("resume_jd_pair") == "resume_jd_pair"


def test_an_unknown_jd_prompt_version_is_refused_everywhere():
    for call in (lambda: jd_prompt("v9"), lambda: task_fingerprint("jd_extraction", jd_prompt_version="v9"),
                 lambda: build_items("jd_extraction", jd_prompt_version="v9"), lambda: output_task_dir("jd_extraction", "v9")):
        with pytest.raises(ValueError, match="unknown JD prompt version 'v9'.*v2.1, v2.2, v2.3"):
            call()


def test_items_carry_their_prompt_version_so_each_is_checked_by_its_own_validator(tmp_path):
    source = write_csv(tmp_path / "jd.csv", [("JD-1", "job text")], "jd_id", "jd_text")
    built = {v: build_items("jd_extraction", source=source, jd_prompt_version=v)[0] for v in SHARED_VERSIONS}
    for version, item in built.items():
        assert item.context["jd_prompt_version"] == version and version in item.system.splitlines()[0]
    assert len({item.system for item in built.values()}) == 3
    assert build_items("jd_extraction", source=source)[0].system == built["v2.3"].system


def test_a_valid_rubric_passes_the_package_validator():
    jd, good = package_fixture()
    item = Item(0, "JD-1", "s", "u", 100, {"jd": jd})
    fields = validate_reply("jd_extraction", item, json.dumps(good()))
    assert fields["json_valid"] is True and fields["input_status"] == "ok" and fields["valid_for_cache"] is True
    fenced = "```json\n" + json.dumps(good()) + "\n```"
    assert validate_reply("jd_extraction", item, fenced)["input_status"] == "ok"  # one whole fence is tolerated


def test_rubric_failures_name_the_validators_reasons_and_cap_the_list():
    jd, good = package_fixture()
    item = Item(0, "JD-1", "s", "u", 100, {"jd": jd})
    altered = good()
    altered["skills"][0]["source_quote"] = "AWS ... Azure"
    with pytest.raises(ValidationError, match="ALTERED_QUOTE"):
        validate_reply("jd_extraction", item, json.dumps(altered))
    drifted = good()
    drifted["input_validity"] = drifted.pop("input_status")  # the wrong layout the models produced before
    with pytest.raises(ValidationError, match="SCHEMA"):
        validate_reply("jd_extraction", item, json.dumps(drifted))
    many = good()
    for row in [*many["skills"], *many["key_responsibilities"]]:
        row["source_quote"] = "not in the jd"  # eight bad quotes, more than the six the message lists
    with pytest.raises(ValidationError, match=r"\(\+\d+ more\)"):
        validate_reply("jd_extraction", item, json.dumps(many))
    string_null = copy.deepcopy(good())
    string_null["skills"][0]["min_years"] = "null"
    with pytest.raises(ValidationError, match="SCHEMA"):
        validate_reply("jd_extraction", item, json.dumps(string_null))


# --- The v2.2 rules (enrollment, short hiring requests, generic other-skills, unusable inputs) ---

REVIEW_CODES = ("OTHER_SKILLS_PRIORITY_REVIEW", "HARD_GATE_QUOTE_WEAK")


def test_the_saved_v21_outputs_are_judged_exactly_as_the_v22_packages_own_audit_says():
    """The package audited these 12 old outputs: 7 usable and 5 to reject. The reasons must reach the retry message."""
    audit = {row["id"]: row for row in json.loads((tasks.JD_V22_PACKAGE / "historical_v22_audit.json").read_text(encoding="utf-8"))["results"]}
    outputs = saved_v21_outputs()
    assert set(outputs) == set(audit) and sum(row["valid_for_cache"] for row in audit.values()) == 7
    for jd_id, (jd, output) in outputs.items():
        expected = audit[jd_id]
        if expected["valid_for_cache"]:
            assert validate_reply("jd_extraction", v22_item(jd, jd_id), output)["valid_for_cache"] is True, jd_id
            continue
        with pytest.raises(ValidationError) as caught:
            validate_reply("jd_extraction", v22_item(jd, jd_id), output)
        message = str(caught.value)
        wanted = [text.split(":")[0] for text in expected["errors"]]
        wanted += [text.split(":")[0] for text in expected["warnings"] if text.startswith(REVIEW_CODES)]
        assert wanted and all(code in message for code in wanted), (jd_id, message)


def test_the_old_prompts_judgement_is_unchanged_for_v21_so_its_results_stay_comparable():
    for jd_id, (jd, output) in saved_v21_outputs().items():
        item = Item(0, jd_id, "s", "u", 100, {"jd": jd, "jd_prompt_version": "v2.1"})
        assert validate_reply("jd_extraction", item, output)["valid_for_cache"] is True, jd_id  # v2.1 accepted all 12


def test_a_review_only_warning_is_rejected_with_a_reason_and_not_an_empty_message():
    """v2.2 can block with warnings alone. The validator then has no errors, and the reply must still say why."""
    jd, output = saved_v21_outputs()["JD-001951"]
    with pytest.raises(ValidationError) as caught:
        validate_reply("jd_extraction", v22_item(jd), output)
    assert "OTHER_SKILLS_PRIORITY_REVIEW" in str(caught.value) and "Effective Communication" in str(caught.value)


def test_enrollment_is_a_condition_and_not_a_completed_degree():
    jd, output = saved_v21_outputs()["JD-000016"]
    with pytest.raises(ValidationError, match="ENROLLMENT_CONDITION_MISSING.*ENROLLMENT_AS_COMPLETED_DEGREE"):
        validate_reply("jd_extraction", v22_item(jd), output)
    rubric = json.loads(output)
    rubric["education_requirement"] = {"min_level": None, "accepted_fields": [], "priority": None,
                                       "equivalent_experience_accepted": False, "hard_gate_explicit": False, "jd_evidence": None}
    phrase = "Currently studying for a degree in Computer Science, Software Engineering, or Computer Engineering"
    assert phrase in jd
    rubric["requirements"].append({
        "id": "R3", "category": "other", "requirement": "Currently studying toward a degree in Computer Science, Software "
        "Engineering or Computer Engineering", "priority": "mandatory", "importance": "important", "priority_origin": "heading",
        "hard_gate_explicit": False, "min_years": None, "jd_evidence": "Current enrollment in specified degree fields",
        "source_quote": phrase})
    assert validate_reply("jd_extraction", v22_item(jd), json.dumps(rubric))["valid_for_cache"] is True


def test_a_short_hiring_request_is_partial_and_not_not_a_jd():
    jd, output = saved_v21_outputs()["JD-000314"]
    with pytest.raises(ValidationError, match="MISCLASSIFIED_INFORMAL_JD"):
        validate_reply("jd_extraction", v22_item(jd), output)
    rubric = json.loads(output)
    rubric.update(input_status="partial", warnings=["Informal work request; qualifications unstated"],
                  role_summary="Accounting and tax support for a small business", primary_functional_domain="finance_accounting")
    rubric["key_responsibilities"] = [{"id": "D1", "responsibility": "Help with accounting and taxes", "importance": "core",
                                       "source_quote": "needs help with accountingtaxes"}]
    assert validate_reply("jd_extraction", v22_item(jd), json.dumps(rubric))["valid_for_cache"] is True


def test_generic_other_skills_are_accepted_once_they_are_no_longer_mandatory():
    jd, output = saved_v21_outputs()["JD-001951"]
    rubric = json.loads(output)
    for skill in rubric["skills"]:
        if skill["name"].lower() in ("problem solving", "effective communication", "technical writing"):
            skill.update(priority="preferred", priority_origin="inferred", importance="supporting")
    assert validate_reply("jd_extraction", v22_item(jd), json.dumps(rubric))["valid_for_cache"] is True


def test_a_genuine_not_a_jd_answer_is_final_but_marked_unusable_and_content_in_it_is_still_rejected():
    jd, good = package_fixture()
    empty = good()
    empty.update(input_status="not_a_jd", skills=[], key_responsibilities=[], requirements=[], any_of_groups=[],
                 top_critical_skill_ids=[], critical_skill_rationale=[])
    fields = validate_reply("jd_extraction", v22_item("Lorem ipsum dolor sit amet."), json.dumps(empty))
    assert fields["input_status"] == "not_a_jd" and fields["valid_for_cache"] is False  # final: asking again cannot help
    assert fields["validator_warnings"].startswith("NOT_A_RUBRIC: input_status=not_a_jd")
    legacy = Item(0, "JD-1", "s", "u", 100, {"jd": "Lorem ipsum dolor sit amet.", "jd_prompt_version": "v2.1"})
    assert validate_reply("jd_extraction", legacy, json.dumps(empty))["valid_for_cache"] is True  # v2.1 had no such flag
    full = good()
    full["input_status"] = "multiple_roles"  # abstention must leave the requirement lists empty
    with pytest.raises(ValidationError, match="INVALID_INPUT_CONTENT"):
        validate_reply("jd_extraction", v22_item(jd), json.dumps(full))


def test_review_warnings_are_taken_from_the_validator_so_the_list_lives_in_one_place():
    report_class = tasks.load_rubric_validator("v2.2").Report
    report = report_class(warnings=["TOO_MANY_WARNINGS: long", "HARD_GATE_QUOTE_WEAK: R1: weak", "OTHER_SKILLS_PRIORITY_REVIEW: S2 x"])
    assert tasks._blocking_warnings(report) == ["HARD_GATE_QUOTE_WEAK: R1: weak", "OTHER_SKILLS_PRIORITY_REVIEW: S2 x"]
    old = tasks.load_rubric_validator("v2.1").Report(warnings=["HARD_GATE_QUOTE_WEAK: R1: weak"])
    assert tasks._blocking_warnings(old) == []  # v2.1 never blocked on warnings


def test_the_validators_load_side_by_side_without_clashing():
    loaded = {v: tasks.load_rubric_validator(v) for v in SHARED_VERSIONS}
    assert len({id(m) for m in loaded.values()}) == 3
    assert len({m.versioned_cache_key("jd", "m", "x") for m in loaded.values()}) == 3  # each stamps its own schema version
    assert tasks.load_rubric_validator() is loaded["v2.3"]  # no version means the default


def test_resume_replies_need_the_schema_fields_and_unusable_documents_only_a_status():
    item = Item(0, "RES-1", "s", "u", 100)
    assert validate_reply("resume_extraction", item, json.dumps(RESUME_REPLY))["json_valid"] is True
    assert validate_reply("resume_extraction", item, '{"input_status": "empty_or_unreadable"}')["input_status"] == "empty_or_unreadable"
    with pytest.raises(ValidationError, match="projects"):
        validate_reply("resume_extraction", item, json.dumps({k: v for k, v in RESUME_REPLY.items() if k != "projects"}))
    with pytest.raises(ValidationError, match="not one of the schema"):
        validate_reply("resume_extraction", item, json.dumps({**RESUME_REPLY, "input_status": "great"}))
    with pytest.raises(ValidationError, match="duplicate ids"):
        validate_reply("resume_extraction", item, json.dumps({**RESUME_REPLY, "work_experiences": [{"id": "E1"}, {"id": "E1"}]}))
    with pytest.raises(ValidationError):
        validate_reply("resume_extraction", item, '{"input_status": "ok"}')


def test_pair_reply_is_scored_by_the_calculator_and_bad_quotes_are_rejected():
    jd = "We need a Python developer with 5 years of relevant experience."
    resume = "Software Engineer at Acme, 2019-2024. Built Python services."
    item = Item(0, "pair-1", "s", "u", 100, {"job_description": jd, "resume": resume})
    good = {
        "status": "scored",
        "checks": [
            {"id": "R1", "jd": "Python developer", "kind": "skill", "priority": "mandatory", "status": "met",
             "quote": "Built Python services"},
            {"id": "R2", "jd": "5 years of relevant experience", "kind": "experience", "priority": "mandatory",
             "status": "met", "quote": "Software Engineer at Acme, 2019-2024"},
        ],
        "work": 7, "role": 8,
        "years": {"required": 5, "relevant": 5, "certain": True},
        "reason": "Python services over five years.",
        "review": False,
    }
    fields = validate_reply("resume_jd_pair", item, json.dumps(good))
    assert fields["verdict"] in {"strong_fit", "good_fit", "potential_fit", "weak_fit", "no_fit"}
    assert "final_score" in fields and fields["model_reason"] == "Python services over five years."
    bad = json.loads(json.dumps(good))
    bad["checks"][0]["quote"] = "Led a team of ten"
    with pytest.raises(ValidationError):
        validate_reply("resume_jd_pair", item, json.dumps(bad))
    # The calculator raises a TypeError (not a ValueError) for a dict where the top-level status string belongs.
    weird = json.loads(json.dumps(good))
    weird["status"] = {"x": 1}
    with pytest.raises(ValidationError, match="TypeError"):
        validate_reply("resume_jd_pair", item, json.dumps(weird))


def test_pair_items_are_streamed_from_parquet_and_the_seeded_sample_keeps_file_order(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "pairs.parquet"
    n = 50
    pq.write_table(pa.table({"pair_id": [f"P-{i}" for i in range(n)], "resume_text": [f"resume {i}" for i in range(n)],
                             "jd_text": [f"jd {i}" for i in range(n)]}), path)
    everything = build_items("resume_jd_pair", source=path)
    assert len(everything) == n and everything[3].context == {"resume": "resume 3", "job_description": "jd 3"}
    sample = build_items("resume_jd_pair", source=path, limit=6, seed=3)
    again = build_items("resume_jd_pair", source=path, limit=6, seed=3)
    ids = [i.item_id for i in sample]
    assert len(ids) == 6 and ids == [i.item_id for i in again]
    assert ids == sorted(ids, key=lambda x: int(x.split("-")[1])) and [i.index for i in sample] == list(range(6))
    assert [i.item_id for i in build_items("resume_jd_pair", source=path, limit=4)] == ["P-0", "P-1", "P-2", "P-3"]
    chosen = build_items("resume_jd_pair", source=path, item_ids=["P-40", "P-5"])
    assert [i.item_id for i in chosen] == ["P-5", "P-40"]
    with pytest.raises(ValueError, match="not in the dataset"):
        build_items("resume_jd_pair", source=path, item_ids=["P-5", "P-999"])


def test_result_columns_put_the_model_output_last_and_pair_columns_in_place():
    assert result_columns("resume_jd_pair")[-1] == "output"
    assert "final_score" in result_columns("resume_jd_pair")
    assert "final_score" not in result_columns("jd_extraction") and "validator_warnings" in result_columns("jd_extraction")
    columns = result_columns("jd_extraction")
    assert columns.index("input_status") < columns.index("valid_for_cache") < columns.index("logistics_conditions") < columns.index("validator_warnings")
    assert "valid_for_cache" not in result_columns("resume_extraction")  # only JD rubrics are cached
    assert "logistics_conditions" not in result_columns("resume_extraction")
    assert result_columns("jd_extraction")[:2] == ["jd_id", "jd"]
    assert result_columns("resume_extraction")[:2] == ["resume_id", "resume"]
    assert result_columns("resume_jd_pair")[:3] == ["pair_id", "resume", "jd"]


# --- The v2.3 rules (open-category examples, explicitly listed generic skills, employment logistics) ---

REVIEW_CODES_V23 = (*REVIEW_CODES, "LOGISTICS_POLICY_REVIEW")


def saved_v22_outputs():
    """The 12 rubrics the v2.2 prompt produced, each with its JD, as shipped with the v2.3 package for regression tests."""
    csv.field_size_limit(10**9)
    with (tasks.JD_V23_PACKAGE / "tests" / "fixtures" / "results_v22.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {row["jd_id"]: (row["jd"], row["output"]) for row in csv.DictReader(handle)}


def v23_item(jd, item_id="JD-1"):
    return Item(0, item_id, "s", "u", 100, {"jd": jd, "jd_prompt_version": "v2.3"})


def salesforce_rubric(*, logistics_hard_gates):
    """JD-000002 with the v2.3 fixes: the two named platforms in one open group instead of a generic 'SaaS Development'."""
    jd, output = saved_v22_outputs()["JD-000002"]
    rubric = json.loads(output)
    template = rubric["skills"][-1]
    assert template["name"] == "SaaS Development"
    quote = "Familiarity with developing on an enterprise-level SaaS (Salesforce, ServiceNow, etc.)"
    assert quote in jd
    template.update(name="Salesforce", raw="Salesforce", any_of_group="G1", source_quote=quote, jd_evidence="Example enterprise SaaS platform")
    second_id = f"S{len(rubric['skills']) + 1}"
    rubric["skills"].append({**copy.deepcopy(template), "id": second_id, "name": "ServiceNow", "raw": "ServiceNow"})
    rubric["any_of_groups"] = [{
        "id": "G1", "description": "Enterprise SaaS platforms", "member_skill_ids": [template["id"], second_id], "min_required": 1,
        "open_category": True, "category_label": "enterprise_saas", "priority": template["priority"],
        "importance": template["importance"], "priority_origin": template["priority_origin"], "hard_gate_explicit": False}]
    for requirement in rubric["requirements"]:
        if requirement["category"] in ("location", "work_authorization"):
            requirement["hard_gate_explicit"] = logistics_hard_gates
    return jd, rubric


def test_the_saved_v22_outputs_are_judged_exactly_as_the_v23_packages_own_audit_says():
    """The package re-audited these 12 outputs: 9 usable and 3 to reject. The reasons must reach the retry message."""
    audit_file = tasks.JD_V23_PACKAGE / "v22_outputs_under_v23_validator.json"
    audit = {row["jd_id"]: row for row in json.loads(audit_file.read_text(encoding="utf-8"))["results"]}
    outputs = saved_v22_outputs()
    assert set(outputs) == set(audit) and sum(row["valid_for_cache"] for row in audit.values()) == 9
    for jd_id, (jd, output) in outputs.items():
        expected = audit[jd_id]
        if expected["valid_for_cache"]:
            assert validate_reply("jd_extraction", v23_item(jd, jd_id), output)["valid_for_cache"] is True, jd_id
            continue
        with pytest.raises(ValidationError) as caught:
            validate_reply("jd_extraction", v23_item(jd, jd_id), output)
        wanted = [text.split(":")[0] for text in expected["errors"]]
        wanted += [text.split(":")[0] for text in expected["warnings"] if text.startswith(REVIEW_CODES_V23)]
        assert wanted and all(code in str(caught.value) for code in wanted), (jd_id, str(caught.value))


def test_v22_still_accepts_all_twelve_of_its_outputs_so_the_versions_really_differ():
    for jd_id, (jd, output) in saved_v22_outputs().items():
        assert validate_reply("jd_extraction", v22_item(jd, jd_id), output)["valid_for_cache"] is True, jd_id


def test_named_platform_examples_must_share_one_open_group_with_one_required_member():
    jd, rubric = salesforce_rubric(logistics_hard_gates=True)
    assert validate_reply("jd_extraction", v23_item(jd), json.dumps(rubric))["valid_for_cache"] is True
    closed = copy.deepcopy(rubric)
    closed["any_of_groups"][0]["open_category"] = False
    closed["any_of_groups"][0]["category_label"] = None
    with pytest.raises(ValidationError, match="INVALID_OPEN_ALTERNATIVE_GROUP"):
        validate_reply("jd_extraction", v23_item(jd), json.dumps(closed))
    both_required = copy.deepcopy(rubric)
    both_required["any_of_groups"][0]["min_required"] = 2
    with pytest.raises(ValidationError, match="INVALID_OPEN_ALTERNATIVE_GROUP"):
        validate_reply("jd_extraction", v23_item(jd), json.dumps(both_required))
    one_missing = copy.deepcopy(rubric)
    one_missing["skills"] = [s for s in one_missing["skills"] if s["name"] != "ServiceNow"]
    one_missing["any_of_groups"] = []
    one_missing["skills"][-1]["any_of_group"] = None
    with pytest.raises(ValidationError, match="MISSING_OPEN_EXAMPLE"):
        validate_reply("jd_extraction", v23_item(jd), json.dumps(one_missing))


def test_an_explicitly_listed_generic_skill_may_not_be_dropped_but_is_never_a_hard_gate():
    jd, output = saved_v22_outputs()["JD-008861"]
    with pytest.raises(ValidationError, match="MISSING_EXPLICIT_OTHER_SKILL.*problem solving"):
        validate_reply("jd_extraction", v23_item(jd), output)
    rubric = json.loads(output)
    rubric["skills"].append({
        "id": f"S{len(rubric['skills']) + 1}", "name": "Problem Solving", "raw": "problem solving", "skill_type": "interpersonal",
        "priority": "preferred", "importance": "supporting", "priority_origin": "inferred", "hard_gate_explicit": False,
        "any_of_group": None, "min_years": None, "expected_level": "unspecified", "jd_evidence": "Other listed competency",
        "source_quote": "problem solving"})
    assert validate_reply("jd_extraction", v23_item(jd), json.dumps(rubric))["valid_for_cache"] is True
    as_gate = copy.deepcopy(rubric)
    as_gate["skills"][-1].update(priority="mandatory", priority_origin="explicit", hard_gate_explicit=True)
    with pytest.raises(ValidationError, match="GENERIC_SKILL_HARD_GATE"):
        validate_reply("jd_extraction", v23_item(jd), json.dumps(as_gate))


def test_a_sponsorship_or_relocation_policy_is_kept_as_an_explicit_condition_or_the_reply_is_sent_back():
    """The review warning alone blocks (no errors), so its reason has to be in the message."""
    jd, rubric = salesforce_rubric(logistics_hard_gates=False)
    with pytest.raises(ValidationError) as caught:
        validate_reply("jd_extraction", v23_item(jd), json.dumps(rubric))
    assert "LOGISTICS_POLICY_REVIEW" in str(caught.value) and "technical-fit gate" in str(caught.value)
    missing = copy.deepcopy(rubric)
    missing["requirements"] = [r for r in missing["requirements"] if r["category"] != "work_authorization"]
    for index, requirement in enumerate(missing["requirements"], 1):
        requirement["id"] = f"R{index}"
    with pytest.raises(ValidationError, match="MISSING_EXPLICIT_LOGISTICS"):
        validate_reply("jd_extraction", v23_item(jd), json.dumps(missing))


def test_logistics_are_listed_apart_from_job_fit_and_older_versions_do_not_pretend_to_separate_them():
    jd, rubric = salesforce_rubric(logistics_hard_gates=True)
    fields = validate_reply("jd_extraction", v23_item(jd), json.dumps(rubric))
    assert "work_authorization" in fields["logistics_conditions"] and "location" in fields["logistics_conditions"]
    fit, logistics = tasks.split_requirements(rubric)
    assert {r["category"] for r in logistics} == {"work_authorization", "location"}
    assert logistics and not any(r["category"] in ("work_authorization", "location", "travel_shift") for r in fit)
    assert len(fit) + len(logistics) == len(rubric["requirements"])  # nothing is dropped, only separated
    plain_jd, plain_output = saved_v22_outputs()["JD-003899"]
    assert validate_reply("jd_extraction", v23_item(plain_jd), plain_output)["logistics_conditions"] == ""
    assert "logistics_conditions" not in validate_reply("jd_extraction", v22_item(plain_jd), plain_output)
    with pytest.raises(ValueError, match="does not separate logistics"):
        tasks.split_requirements(rubric, "v2.2")


def test_logistics_warnings_are_blocking_for_v23_and_not_for_v22():
    review = "LOGISTICS_POLICY_REVIEW: Explicit location policy should preserve non-negotiable wording"
    assert tasks._blocking_warnings(tasks.load_rubric_validator("v2.3").Report(warnings=[review])) == [review]
    assert tasks._blocking_warnings(tasks.load_rubric_validator("v2.2").Report(warnings=[review])) == []


# --- gpt-oss-20b: its own JD prompt and settings, and nothing changes for any other model ---


def test_the_gptoss_version_carries_its_settings_and_no_shared_version_sets_any():
    gptoss = tasks.JD_PROMPTS["v2.3-gptoss"]
    assert (gptoss.models, gptoss.max_output_tokens, gptoss.reject_duplicate_keys, gptoss.retry_temperature) == (
        ("gpt-oss-20b",), 8192, True, 0.3)
    assert output_task_dir("jd_extraction", "v2.3-gptoss") == "jd_extraction_v2_3_gptoss"
    for version in SHARED_VERSIONS:  # every model-specific setting is off by default, so these versions run as they always did
        shared = tasks.JD_PROMPTS[version]
        assert (shared.models, shared.max_output_tokens, shared.reject_duplicate_keys, shared.retry_temperature) == (None, None, False, None)
    assert tasks.default_jd_prompt_version("gpt-oss-20b") == "v2.3-gptoss"
    for other in ("qwen3.8-27b", "gemma-4-12b", "qwen3.5-4b", None):
        assert tasks.default_jd_prompt_version(other) is None  # they keep the shared default


def test_the_gptoss_items_use_the_standalone_prompt_the_lower_cap_and_the_retry_setting_and_qwens_do_not(tmp_path):
    source = write_csv(tmp_path / "jd.csv", [("JD-1", "job text")], "jd_id", "jd_text")
    (gptoss,) = build_items("jd_extraction", source=source, jd_prompt_version="v2.3-gptoss")
    (qwen,) = build_items("jd_extraction", source=source)
    assert gptoss.system == tasks.jd_system_prompt(tasks.JD_PROMPTS["v2.3-gptoss"])  # the standalone prompt, as it is sent
    assert gptoss.max_output_tokens == 8192 and gptoss.retry_temperature == 0.3
    assert qwen.system == tasks.JD_PROMPTS["v2.3"].full_prompt.read_text(encoding="utf-8")
    assert qwen.max_output_tokens == 12288 and qwen.retry_temperature is None
    assert gptoss.user == qwen.user  # the JD goes to the model the same way
    (explicit,) = build_items("jd_extraction", source=source, jd_prompt_version="v2.3-gptoss", max_output_tokens=5000)
    assert explicit.max_output_tokens == 5000  # a cap given on the command line still wins
    assert task_fingerprint("jd_extraction", jd_prompt_version="v2.3-gptoss", max_output_tokens=5000) != task_fingerprint(
        "jd_extraction", jd_prompt_version="v2.3-gptoss")


def test_the_gptoss_package_reuses_the_v23_schema_and_validator_so_the_v23_checks_apply_unchanged():
    gptoss, v23 = tasks.JD_PROMPTS["v2.3-gptoss"], tasks.JD_PROMPTS["v2.3"]
    assert gptoss.schema.read_bytes() == v23.schema.read_bytes() and gptoss.validator.read_bytes() == v23.validator.read_bytes()


def test_gptoss_rubrics_are_judged_exactly_as_v23_judges_them():
    for jd_id, (jd, output) in saved_v22_outputs().items():
        outcomes = []
        for version in ("v2.3", "v2.3-gptoss"):
            item = Item(0, jd_id, "s", "u", 100, {"jd": jd, "jd_prompt_version": version})
            try:
                outcomes.append(("accepted", validate_reply("jd_extraction", item, output)))
            except ValidationError as exc:
                outcomes.append(("rejected", str(exc)))
        assert outcomes[0] == outcomes[1], jd_id


def test_only_the_gptoss_version_refuses_a_reply_that_repeats_a_json_key():
    jd, good = package_fixture()
    repeated = json.dumps(good()).replace('"input_status": "ok"', '"input_status": "ok", "input_status": "ok"', 1)
    assert repeated.count('"input_status"') == 2
    gptoss = Item(0, "JD-1", "s", "u", 100, {"jd": jd, "jd_prompt_version": "v2.3-gptoss"})
    with pytest.raises(ValidationError, match="repeats the JSON key 'input_status'"):
        validate_reply("jd_extraction", gptoss, repeated)
    for version in (None, *SHARED_VERSIONS):  # every other prompt reads it as it always did, keeping the last value
        context = {"jd": jd} if version is None else {"jd": jd, "jd_prompt_version": version}
        assert validate_reply("jd_extraction", Item(0, "JD-1", "s", "u", 100, context), repeated)["valid_for_cache"] is True
    assert tasks.parse_strict_object(repeated)["input_status"] == "ok"  # and the strict parser is lenient unless asked
    with pytest.raises(ValidationError, match="repeats"):
        tasks.parse_strict_object(repeated, reject_duplicate_keys=True)


def test_the_gptoss_fingerprint_changes_with_its_settings_and_the_shared_ones_do_not_depend_on_them(monkeypatch):
    base = task_fingerprint("jd_extraction", jd_prompt_version="v2.3-gptoss")
    for change in ({"retry_temperature": 0.5}, {"reject_duplicate_keys": False}):
        monkeypatch.setitem(tasks.JD_PROMPTS, "v2.3-gptoss", dataclasses.replace(tasks.JD_PROMPTS["v2.3-gptoss"], **change))
        assert task_fingerprint("jd_extraction", jd_prompt_version="v2.3-gptoss") != base, change
        monkeypatch.undo()
    # The shared versions' fingerprints are pinned elsewhere; this shows a model-specific setting cannot reach them.
    shared = {v: task_fingerprint("jd_extraction", jd_prompt_version=v) for v in SHARED_VERSIONS}
    monkeypatch.setitem(tasks.JD_PROMPTS, "v2.3-gptoss", dataclasses.replace(tasks.JD_PROMPTS["v2.3-gptoss"], retry_temperature=0.9))
    assert shared == {v: task_fingerprint("jd_extraction", jd_prompt_version=v) for v in SHARED_VERSIONS}


def test_the_gptoss_prompt_is_sent_without_its_response_formats_block_and_the_package_file_is_left_as_shipped():
    """On our model server that block makes gpt-oss answer in a form the server rejects with HTTP 500 (peg-native format)."""
    gptoss = tasks.JD_PROMPTS["v2.3-gptoss"]
    shipped = gptoss.full_prompt.read_text(encoding="utf-8")
    sent = tasks.jd_system_prompt(gptoss)
    assert "# Response Formats" in shipped and "# Response Formats" not in sent  # the file is untouched; only what is sent differs
    removed = [line for line in shipped.splitlines() if line not in sent.splitlines()]
    assert removed == ["# Response Formats", "## jd_rubric", "// Final response: a single object conforming to the existing JD rubric schema."]
    assert sent.replace("\n\n# Allowed values", "\n\n# Response Formats\n## jd_rubric\n// Final response: a single object conforming "
                        "to the existing JD rubric schema.\n\n# Allowed values") == shipped  # nothing else changed
    for version in SHARED_VERSIONS:  # every shared version is sent exactly as its file reads
        shared = tasks.JD_PROMPTS[version]
        assert tasks.jd_system_prompt(shared) == shared.full_prompt.read_text(encoding="utf-8")


def test_a_prompt_that_should_have_a_response_formats_block_but_does_not_is_an_error_and_not_a_silent_no_op(tmp_path):
    plain = tmp_path / "plain.txt"
    plain.write_text("a prompt with no such block", encoding="utf-8")
    broken = dataclasses.replace(tasks.JD_PROMPTS["v2.3-gptoss"], full_prompt=plain)
    with pytest.raises(ValueError, match="expects a '# Response Formats' block"):
        tasks.jd_system_prompt(broken)


def test_turning_the_block_removal_off_changes_the_gptoss_fingerprint_so_older_failed_results_are_not_reused(monkeypatch):
    base = task_fingerprint("jd_extraction", jd_prompt_version="v2.3-gptoss")
    monkeypatch.setitem(tasks.JD_PROMPTS, "v2.3-gptoss", dataclasses.replace(tasks.JD_PROMPTS["v2.3-gptoss"], without_response_formats=False))
    assert task_fingerprint("jd_extraction", jd_prompt_version="v2.3-gptoss") != base


# --- gemma-4-26B-A4B: the v2.3 prompt plus the Gemma field rules, for that model only ---


def test_the_gemma26_version_is_the_v23_prompt_plus_the_package_rules_and_nothing_else():
    import importlib.util

    gemma, v23 = tasks.JD_PROMPTS["v2.3-gemma26"], tasks.JD_PROMPTS["v2.3"]
    base = v23.full_prompt.read_text(encoding="utf-8")
    sent = tasks.jd_system_prompt(gemma)
    assert sent.startswith(base.rstrip()) and len(sent) > len(base)  # Qwen's prompt is untouched; rules come after it
    spec = importlib.util.spec_from_file_location("gemma_pkg_core", tasks.JD_GEMMA_PACKAGE / "gemma_jd_fix" / "core.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    assert sent == package.apply_gemma_rules(base, "targeted")  # exactly what the package's own function builds
    assert sent.endswith(package.GEMMA_RULES) and "key_responsibilities[].importance: core / secondary ONLY" in sent
    # the same schema, validator, output cap and retry as Qwen: the rules are the only difference
    assert (gemma.schema, gemma.validator) == (v23.schema, v23.validator)
    assert (gemma.max_output_tokens, gemma.reject_duplicate_keys, gemma.retry_temperature, gemma.without_response_formats) == (
        None, False, None, False)
    assert gemma.models == ("gemma-4-26b-a4b",) and output_task_dir("jd_extraction", "v2.3-gemma26") == "jd_extraction_v2_3_gemma26"
    assert tasks.jd_system_prompt(v23) == base  # Qwen's version is still sent exactly as its file reads


def test_only_the_26b_gemma_gets_the_gemma_rules_by_default():
    assert tasks.default_jd_prompt_version("gemma-4-26b-a4b") == "v2.3-gemma26"
    for other in ("gemma-4-12b", "qwen3.8-27b", "qwen3.5-4b", None):
        assert tasks.default_jd_prompt_version(other) in (None,)  # they keep the shared v2.3
    assert tasks.default_jd_prompt_version("gpt-oss-20b") == "v2.3-gptoss"
    for version in SHARED_VERSIONS:
        assert tasks.JD_PROMPTS[version].addendum_from is None  # no shared version carries an addendum


def test_the_gemma_rules_are_part_of_the_fingerprint_so_edited_rules_never_reuse_older_results(monkeypatch, tmp_path):
    base = task_fingerprint("jd_extraction", jd_prompt_version="v2.3-gemma26")
    assert base != task_fingerprint("jd_extraction", jd_prompt_version="v2.3")
    edited = tmp_path / "core.py"
    edited.write_text('GEMMA_RULES = "different rules"\n', encoding="utf-8")
    monkeypatch.setitem(tasks.JD_PROMPTS, "v2.3-gemma26", dataclasses.replace(
        tasks.JD_PROMPTS["v2.3-gemma26"], addendum_from=(edited, "GEMMA_RULES")))
    assert task_fingerprint("jd_extraction", jd_prompt_version="v2.3-gemma26") != base
    items = build_items("jd_extraction", item_ids=["JD-000314"], jd_prompt_version="v2.3")
    assert items[0].system == tasks.JD_PROMPTS["v2.3"].full_prompt.read_text(encoding="utf-8")


def test_gemma_items_send_the_rules_and_gemma_replies_are_judged_by_the_unchanged_v23_checks(tmp_path):
    source = write_csv(tmp_path / "jd.csv", [("JD-1", "job text")], "jd_id", "jd_text")
    (gemma,) = build_items("jd_extraction", source=source, jd_prompt_version="v2.3-gemma26")
    (qwen,) = build_items("jd_extraction", source=source)
    assert gemma.system == tasks.jd_system_prompt(tasks.JD_PROMPTS["v2.3-gemma26"]) and gemma.system != qwen.system
    assert gemma.user == qwen.user and gemma.max_output_tokens == qwen.max_output_tokens == 12288
    assert gemma.retry_temperature is None and gemma.context["jd_prompt_version"] == "v2.3-gemma26"
    for jd_id, (jd, output) in saved_v22_outputs().items():  # the same accept/reject decisions as v2.3
        outcomes = []
        for version in ("v2.3", "v2.3-gemma26"):
            item = Item(0, jd_id, "s", "u", 100, {"jd": jd, "jd_prompt_version": version})
            try:
                outcomes.append(("accepted", validate_reply("jd_extraction", item, output)))
            except ValidationError as exc:
                outcomes.append(("rejected", str(exc)))
        assert outcomes[0] == outcomes[1], jd_id


def test_length_order_runs_shortest_first_and_the_default_keeps_dataset_order(tmp_path):
    source = write_csv(tmp_path / "jd.csv", [("JD-1", "x" * 900), ("JD-2", "x" * 100), ("JD-3", "x" * 500)], "jd_id", "jd_text")
    default = [i.item_id for i in build_items("jd_extraction", source=source, item_ids=["JD-1", "JD-2", "JD-3"])]
    shortest = [i.item_id for i in build_items("jd_extraction", source=source, item_ids=["JD-1", "JD-2", "JD-3"], order="length")]
    assert default == ["JD-1", "JD-2", "JD-3"] and shortest == ["JD-2", "JD-3", "JD-1"]
