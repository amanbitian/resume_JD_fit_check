"""The three production tasks: resume extraction, JD extraction and resume-JD pair scoring.

Each task builds one request per item (a resume, a JD, or a resume/JD pair) and validates the model's reply.
Validation is strict: a reply that is not a single JSON object, lacks the schema's required fields, or is rejected
by the task's own validator is recorded as an error with the reason. It never becomes a score or a cached rubric.

Prompt sources (all inside this repository):
  resume_extraction  prompt/old prompt/1_resume_extraction_system_prompt.txt  (+ user template)
  jd_extraction      a versioned rubric package (JD_PROMPTS): the FULL prompt, checked by its JSON schema and
                     rubric_validator.py (exact quotes, ids, groups, hard-gate provenance). The default is v2.3,
                     prompt/jd_rubric_extractor_v2_3_targeted/jd_rubric_v2_3/. v2.2 added the enrollment, short hiring
                     request and generic other-skills rules; v2.3 adds open-category examples, explicitly listed generic
                     skills and employment logistics kept apart from job fit. Older versions stay selectable so their saved
                     results can be reproduced. Every version writes to its own output folder, so a new prompt never
                     overwrites old results.
  resume_jd_pair     prompt/Prompt_v2/prompt.txt, scored by prompt/Prompt_v2/scoring.py

Each JD package also ships a shorter SCHEMA_MODE prompt for backends that enforce a JSON response schema. Neither the dual
GPU service (on its evaluation chat path) nor the cloud batch clients forward a response schema, so the FULL prompt is used.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "prompt"
RESUME_SYSTEM_PROMPT = PROMPT_DIR / "old prompt" / "1_resume_extraction_system_prompt.txt"
RESUME_USER_TEMPLATE = PROMPT_DIR / "old prompt" / "1_resume_extraction_user_prompt_template.txt"
PAIR_SYSTEM_PROMPT = PROMPT_DIR / "Prompt_v2" / "prompt.txt"
PAIR_KIT = PROMPT_DIR / "Prompt_v2" / "scoring.py"
INPUT_DIR = REPO_ROOT / "Input"
RESUME_CSV = INPUT_DIR / "resume_dataset.csv"
JD_CSV = INPUT_DIR / "jd_dataset.csv"
PAIR_PARQUET = INPUT_DIR / "Candidate_JD_eval_159k.parquet"
DEFAULT_REFERENCE_DATE = "2026-10-17"
# JD extraction was cut off at 6,144 tokens on the longest JDs, so its cap is doubled.
DEFAULT_MAX_OUTPUT_TOKENS = {"resume_extraction": 8192, "jd_extraction": 12288, "resume_jd_pair": 4096}


@dataclass(frozen=True, slots=True)
class JDPrompt:
    """One version of the JD rubric extractor: its prompt, schema and validator, and the folder its results go to."""

    version: str
    full_prompt: Path
    schema: Path
    validator: Path
    output_dir: str  # under the output root; one folder per version, so a new prompt never overwrites older results
    # Settings for a version written for one model. Every default below is the behaviour of the shared versions
    # (v2.1 to v2.3), so a version that sets none of them runs exactly as before.
    models: tuple[str, ...] | None = None  # local model keys it is written for; None means any model
    max_output_tokens: int | None = None  # output cap for this version; None means the task default
    reject_duplicate_keys: bool = False  # refuse a reply whose JSON repeats an object key
    retry_temperature: float | None = None  # change the decoding on a retry; None keeps the first attempt's settings
    # Drop the prompt's trailing "# Response Formats" block before sending it. On our model server it makes gpt-oss answer in
    # a form the server's own parser rejects (HTTP 500, "does not match the expected peg-native format"); the package file
    # itself is left as shipped.
    without_response_formats: bool = False
    # Append a block of model-specific rules to the shared prompt: (a Python file, the name of the string in it). The text is
    # read from that file when the prompt is built, so the package stays the one source of it.
    addendum_from: tuple[Path, str] | None = None


JD_V21_PACKAGE = PROMPT_DIR / "Prompt_v2" / "jd_rubric_extractor_v2_1_fixed" / "jd_rubric_v2_1"
JD_V22_PACKAGE = PROMPT_DIR / "jd_rubric_extractor_v2_2_semantic_fixed" / "jd_rubric_v2_2"
JD_V23_PACKAGE = PROMPT_DIR / "jd_rubric_extractor_v2_3_targeted" / "jd_rubric_v2_3"
JD_GPTOSS_PACKAGE = PROMPT_DIR / "gptoss20b_candidate_processing_optimized_v1" / "gptoss20b_rubric_v1"
JD_GEMMA_PACKAGE = PROMPT_DIR / "gemma26_jd_v23_fix"
JD_PROMPTS = {
    # v2.1 keeps the folder name it always had, so the results made with it stay where they are.
    "v2.1": JDPrompt("v2.1", JD_V21_PACKAGE / "JD_Rubric_Extractor_v2.1_FULL.txt", JD_V21_PACKAGE / "jd_rubric.schema.json",
                     JD_V21_PACKAGE / "rubric_validator.py", "jd_extraction"),
    "v2.2": JDPrompt("v2.2", JD_V22_PACKAGE / "JD_Rubric_Extractor_v2.2_FULL.txt", JD_V22_PACKAGE / "jd_rubric.schema.json",
                     JD_V22_PACKAGE / "rubric_validator.py", "jd_extraction_v2_2"),
    "v2.3": JDPrompt("v2.3", JD_V23_PACKAGE / "JD_Rubric_Extractor_v2.3_FULL.txt", JD_V23_PACKAGE / "jd_rubric.schema.json",
                     JD_V23_PACKAGE / "rubric_validator.py", "jd_extraction_v2_3"),
    # The v2.3 rubric for gpt-oss-20b only. Same schema and validator as v2.3; a prompt written for that model (the
    # standalone one, because our evaluation path cannot send a response schema), a lower output cap, and two rules for
    # the model's observed failures: a reply that repeats a JSON key is refused, and a retry changes the decoding state
    # because repeating the same greedy request repeats the same mistake. Qwen and the other models never use it.
    "v2.3-gptoss": JDPrompt("v2.3-gptoss", JD_GPTOSS_PACKAGE / "GPTOSS20B_JD_EXTRACTOR_STANDALONE.txt",
                            JD_GPTOSS_PACKAGE / "jd_rubric.schema.json", JD_GPTOSS_PACKAGE / "rubric_validator.py",
                            "jd_extraction_v2_3_gptoss", models=("gpt-oss-20b",), max_output_tokens=8192,
                            reject_duplicate_keys=True, retry_temperature=0.3, without_response_formats=True),
    # The v2.3 prompt, schema and validator exactly as Qwen uses them, plus the Gemma field rules from the gemma26 package,
    # appended to the prompt. For gemma-4-26B-A4B only: those rules answer the schema drift it showed (the wrong `importance`
    # words for responsibilities, a `source_quote` added to the education and experience objects). Nothing else differs.
    "v2.3-gemma26": JDPrompt("v2.3-gemma26", JD_V23_PACKAGE / "JD_Rubric_Extractor_v2.3_FULL.txt",
                             JD_V23_PACKAGE / "jd_rubric.schema.json", JD_V23_PACKAGE / "rubric_validator.py",
                             "jd_extraction_v2_3_gemma26", models=("gemma-4-26b-a4b",),
                             addendum_from=(JD_GEMMA_PACKAGE / "gemma_jd_fix" / "core.py", "GEMMA_RULES")),
}
DEFAULT_JD_PROMPT_VERSION = "v2.3"
# A local model with a JD prompt written for it uses that one unless a version is chosen. Other models use the default.
MODEL_JD_PROMPT = {"gpt-oss-20b": "v2.3-gptoss", "gemma-4-26b-a4b": "v2.3-gemma26"}

TASK_NAMES = ("resume_extraction", "jd_extraction", "resume_jd_pair")
TASK_LABELS = {
    "resume_extraction": "Run resume extraction",
    "jd_extraction": "Run JD extraction",
    "resume_jd_pair": "Run resume-JD pair scoring",
}
DIM_LABELS = {"skills": "Skills", "work": "Work", "role": "Role", "years": "Years", "education": "Education"}

RESUME_INVALID_STATUS = {"not_a_resume", "empty_or_unreadable", "multiple_candidates"}
RESUME_REQUIRED_LISTS = ("work_experiences", "skills", "projects", "education", "certifications")
ID_KEYS = ("item_id", "jd_id", "resume_id", "pair_id")
MAX_REPORTED_ERRORS = 6


class ValidationError(ValueError):
    """A model reply that cannot be used. The message is stored with the item so the failure is visible."""


@dataclass(frozen=True, slots=True)
class Item:
    index: int
    item_id: str
    system: str
    user: str
    max_output_tokens: int
    # Source texts kept for validation: the JD for JD extraction, the resume and JD for pairs.
    context: dict[str, str] = field(default_factory=dict, compare=False)
    # Set only by a JD prompt version that changes the decoding on a retry (see JDPrompt.retry_temperature).
    retry_temperature: float | None = field(default=None, compare=False)


def prompt_fingerprint(*texts: str) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def jd_user_message(jd_text: str, reference_date: str) -> str:
    return f"REFERENCE_DATE: {reference_date}\n<JOB_DESCRIPTION>\n{jd_text}\n</JOB_DESCRIPTION>"


def jd_prompt(version: str | None = None) -> JDPrompt:
    """The JD prompt version to use; None means the default."""
    chosen = version or DEFAULT_JD_PROMPT_VERSION
    if chosen not in JD_PROMPTS:
        raise ValueError(f"unknown JD prompt version {chosen!r}; choose one of {', '.join(JD_PROMPTS)}")
    return JD_PROMPTS[chosen]


_RESPONSE_FORMATS_BLOCK = re.compile(r"# Response Formats\n.*?\n(?=# )", re.S)


def jd_system_prompt(jd: JDPrompt) -> str:
    """The system prompt of a JD prompt version as it is sent.

    A version that asks for it has a block removed and/or a block of model-specific rules appended. A version that asks for
    neither (v2.1 to v2.3) is sent exactly as its file reads.
    """
    text = jd.full_prompt.read_text(encoding="utf-8")
    if jd.without_response_formats:
        text, removed = _RESPONSE_FORMATS_BLOCK.subn("", text, count=1)
        if not removed:
            raise ValueError(f"JD prompt {jd.version} expects a '# Response Formats' block but its prompt file has none")
    if jd.addendum_from is not None:
        path, name = jd.addendum_from
        rules = getattr(_load_module(f"addendum_{path}", path, f"addendum_{jd.version.replace('.', '').replace('-', '_')}"), name)
        text = text.rstrip() + "\n\n" + rules  # the same join the package uses (apply_gemma_rules, mode "targeted")
    return text


def default_jd_prompt_version(local_model: str | None) -> str | None:
    """The JD prompt version written for a local model, or None when the model uses the shared default."""
    return MODEL_JD_PROMPT.get(local_model) if local_model else None


def _output_cap(task: str, max_output_tokens: int | None, jd_prompt_version: str | None) -> int:
    """An explicit cap wins, then the cap of a JD prompt version that sets its own, then the task default."""
    if max_output_tokens:
        return max_output_tokens
    if task == "jd_extraction" and jd_prompt(jd_prompt_version).max_output_tokens:
        return jd_prompt(jd_prompt_version).max_output_tokens
    return DEFAULT_MAX_OUTPUT_TOKENS[task]


def output_task_dir(task: str, jd_prompt_version: str | None = None) -> str:
    """The folder name for a task's results. Each JD prompt version has its own, so a new prompt cannot overwrite old results."""
    return jd_prompt(jd_prompt_version).output_dir if task == "jd_extraction" else task


def task_fingerprint(
    task: str,
    *,
    reference_date: str = DEFAULT_REFERENCE_DATE,
    max_output_tokens: int | None = None,
    jd_prompt_version: str | None = None,
) -> str:
    """Identifies everything that changes what a reply means. Results made under another fingerprint are not reused.

    The recipe for each task is fixed: changing it would orphan results already saved under the old fingerprint.
    """
    cap = _output_cap(task, max_output_tokens, jd_prompt_version)
    parts = [task, str(cap)]
    if task == "resume_extraction":
        parts += [RESUME_SYSTEM_PROMPT.read_text(encoding="utf-8"), RESUME_USER_TEMPLATE.read_text(encoding="utf-8"),
                  reference_date]
    elif task == "jd_extraction":
        jd = jd_prompt(jd_prompt_version)
        parts += [jd_system_prompt(jd), jd.schema.read_text(encoding="utf-8"),
                  jd.validator.read_text(encoding="utf-8"), reference_date]
        if jd.reject_duplicate_keys or jd.retry_temperature is not None or jd.without_response_formats:  # only versions that set them
            parts.append(f"reject_duplicate_keys={jd.reject_duplicate_keys};retry_temperature={jd.retry_temperature};"
                         f"without_response_formats={jd.without_response_formats}")
    else:
        parts += [PAIR_SYSTEM_PROMPT.read_text(encoding="utf-8"), PAIR_KIT.read_text(encoding="utf-8")]
    return prompt_fingerprint(*parts)


def load_item_ids(path: Path) -> tuple[str, ...]:
    """Read ids from a JSON list (strings, or objects with item_id/jd_id/resume_id/pair_id) or a text file, one per line."""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return tuple(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must hold a JSON list of ids")
    ids = []
    for entry in data:
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, dict) and any(isinstance(entry.get(k), str) for k in ID_KEYS):
            ids.append(next(entry[k] for k in ID_KEYS if isinstance(entry.get(k), str)))
        else:
            raise ValueError(f"cannot read an id from {entry!r} in {path}")
    return tuple(ids)


def _read_csv(path: Path, id_col: str, text_col: str) -> list[tuple[str, str]]:
    csv.field_size_limit(10**9)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [(row[id_col], row[text_col]) for row in csv.DictReader(handle) if row.get(text_col, "").strip()]


def _select(rows: list[tuple[str, str]], limit: int | None, seed: int | None, item_ids: Iterable[str] | None) -> list[tuple[str, str]]:
    """Specific ids (kept in file order), the first `limit` rows, or a seeded random sample of that size in file order."""
    if item_ids is not None:
        wanted = set(item_ids)
        chosen = [row for row in rows if row[0] in wanted]
        missing = wanted - {row[0] for row in chosen}
        if missing:
            raise ValueError(f"{len(missing)} requested ids are not in the dataset, for example {sorted(missing)[:3]}")
        return chosen
    if not limit or limit >= len(rows):
        return rows
    if seed is None:
        return rows[:limit]
    return [rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), limit))]


def build_items(
    task: str,
    *,
    limit: int | None = None,
    reference_date: str = DEFAULT_REFERENCE_DATE,
    max_output_tokens: int | None = None,
    source: Path | None = None,
    seed: int | None = None,
    item_ids: Iterable[str] | None = None,
    jd_prompt_version: str | None = None,
) -> list[Item]:
    """Turn a task and its dataset into requests, in file order. `jd_prompt_version` applies to JD extraction only."""
    if task not in TASK_NAMES:
        raise ValueError(f"unknown task {task!r}; choose one of {', '.join(TASK_NAMES)}")
    if item_ids is not None and limit:
        raise ValueError("give either a list of item ids or a limit, not both")
    cap = _output_cap(task, max_output_tokens, jd_prompt_version)
    if task == "resume_extraction":
        system = RESUME_SYSTEM_PROMPT.read_text(encoding="utf-8")
        template = RESUME_USER_TEMPLATE.read_text(encoding="utf-8")
        rows = _select(_read_csv(source or RESUME_CSV, "resume_id", "resume_text"), limit, seed, item_ids)
        return [
            Item(i, rid, system, template.replace("{REFERENCE_DATE}", reference_date).replace("{RESUME_TEXT}", text), cap,
                 {"resume": text})
            for i, (rid, text) in enumerate(rows)
        ]
    if task == "jd_extraction":
        jd = jd_prompt(jd_prompt_version)
        system = jd_system_prompt(jd)
        rows = _select(_read_csv(source or JD_CSV, "jd_id", "jd_text"), limit, seed, item_ids)
        # The prompt version travels with each item so the reply is checked by that version's own validator.
        return [Item(i, jid, system, jd_user_message(text, reference_date), cap, {"jd": text, "jd_prompt_version": jd.version},
                      jd.retry_temperature)
                for i, (jid, text) in enumerate(rows)]
    return _pair_items(source or PAIR_PARQUET, limit, seed, cap, item_ids)


def _pair_items(path: Path, limit: int | None, seed: int | None, cap: int, item_ids: Iterable[str] | None) -> list[Item]:
    """Stream the parquet in batches and keep only the wanted rows, instead of loading every text first."""
    import pyarrow.parquet as pq

    system = PAIR_SYSTEM_PROMPT.read_text(encoding="utf-8")
    parquet = pq.ParquetFile(path)
    total = parquet.metadata.num_rows
    wanted_rows: set[int] | None = None
    wanted_ids = set(item_ids) if item_ids is not None else None
    if wanted_ids is None and limit and limit < total:
        chosen = random.Random(seed).sample(range(total), limit) if seed is not None else range(limit)
        wanted_rows = set(chosen)
    last_wanted = max(wanted_rows) if wanted_rows else total
    items: list[Item] = []
    offset = 0
    for batch in parquet.iter_batches(batch_size=4096, columns=["pair_id", "resume_text", "jd_text"]):
        ids, resumes, jds = (batch[c].to_pylist() for c in ("pair_id", "resume_text", "jd_text"))
        for i, pair_id in enumerate(ids):
            if wanted_rows is not None and offset + i not in wanted_rows:
                continue
            if wanted_ids is not None and pair_id not in wanted_ids:
                continue
            resume, jd = resumes[i] or "", jds[i] or ""
            user = json.dumps({"job_description": jd, "resume": resume}, ensure_ascii=False)
            items.append(Item(len(items), pair_id, system, user, cap, {"resume": resume, "job_description": jd}))
        offset += len(ids)
        if offset > last_wanted:
            break
    if wanted_ids is not None and len(items) != len(wanted_ids):
        missing = wanted_ids - {item.item_id for item in items}
        raise ValueError(f"{len(missing)} requested ids are not in the dataset, for example {sorted(missing)[:3]}")
    return items


def _no_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """json.loads silently keeps the last of two equal keys. This refuses the reply instead of hiding the repeat."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"reply repeats the JSON key {key!r}")
        result[key] = value
    return result


def parse_strict_object(text: str, *, reject_duplicate_keys: bool = False) -> dict[str, Any]:
    """The whole reply must be one JSON object. One whole code fence is tolerated; trailing text is not.

    `reject_duplicate_keys` is off unless a JD prompt version turns it on, so every other model parses as before.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.IGNORECASE).strip()
    try:
        value = json.loads(cleaned, object_pairs_hook=_no_repeated_keys) if reject_duplicate_keys else json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"reply is not valid JSON: {exc.msg} at character {exc.pos}") from exc
    if not isinstance(value, dict):
        raise ValidationError("reply is not a JSON object")
    return value


_module_cache: dict[str, Any] = {}


def _load_module(key: str, path: Path, name: str) -> Any:
    """Import a module from the prompt folders by path, once, so generic file names cannot collide."""
    if key not in _module_cache:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses with postponed annotations look their module up here
        spec.loader.exec_module(module)
        _module_cache[key] = module
    return _module_cache[key]


def load_pair_kit() -> Any:
    return _load_module("pair_kit", PAIR_KIT, "prompt_v2_scoring")


def load_rubric_validator(version: str | None = None) -> Any:
    """The rubric validator of one JD prompt version. It needs the `jsonschema` package."""
    jd = jd_prompt(version)
    name = jd.version.replace(".", "").replace("-", "_")  # a module name: v2.1 gives v21 (unchanged), v2.3-gptoss gives v23_gptoss
    return _load_module(f"rubric_validator_{jd.version}", jd.validator, f"jd_rubric_validator_{name}")


def _check_resume(value: dict[str, Any]) -> dict[str, Any]:
    status = value.get("input_status")
    if status in RESUME_INVALID_STATUS:
        return {"json_valid": True, "input_status": status}
    if status not in {"ok", "partial"}:
        raise ValidationError(f"input_status {status!r} is not one of the schema's values")
    for name in RESUME_REQUIRED_LISTS:
        if not isinstance(value.get(name), list):
            raise ValidationError(f"required list '{name}' is missing or not a list")
        ids = [entry.get("id") for entry in value[name] if isinstance(entry, dict) and "id" in entry]
        if len(ids) != len(set(ids)):
            raise ValidationError(f"duplicate ids in '{name}'")
    return {"json_valid": True, "input_status": status}


def _blocking_warnings(report: Any) -> list[str]:
    """Warnings the validator itself says need a review before the rubric is published.

    v2.2 blocks on a mandatory skill taken from the generic `otherskills` text and on a hard gate whose quote has no
    hard-minimum wording; v2.3 also blocks on a sponsorship or relocation policy that is not recorded as an explicit
    condition. Asking the validator keeps that list in one place. v2.1 has no such warnings.
    """
    report_class = type(report)
    if not hasattr(report_class, "needs_rubric_review"):
        return []
    return [warning for warning in report.warnings if report_class(warnings=[warning]).needs_rubric_review]


def split_requirements(rubric: dict[str, Any], jd_prompt_version: str | None = None) -> tuple[list[dict], list[dict]]:
    """Split a rubric's `requirements` into (job-fit requirements, logistical conditions).

    Sponsorship, relocation, work-authorisation, shift and location policies are logistics, not technical fit. Anything
    that scores a candidate against a rubric must use the first list for job fit and send the second list to a separate,
    human-reviewed step. The split comes from the validator of the prompt version, which defines it from v2.3 on.
    """
    validator = load_rubric_validator(jd_prompt_version)
    if not hasattr(validator, "partition_requirements_for_scoring"):
        raise ValueError(f"JD prompt {jd_prompt(jd_prompt_version).version} does not separate logistics; use v2.3 or later")
    return validator.partition_requirements_for_scoring(rubric)


def _logistics_text(rubric: dict[str, Any], version: str) -> str | None:
    """The rubric's logistical conditions as one line for review, or None if this prompt version does not separate them."""
    if not hasattr(load_rubric_validator(version), "partition_requirements_for_scoring"):
        return None
    _fit, logistics = split_requirements(rubric, version)
    return "; ".join(f"{row['id']} {row['category']}: {row['requirement']}" for row in logistics)


def _check_rubric(value: dict[str, Any], jd_text: str, jd_prompt_version: str | None = None) -> dict[str, Any]:
    """The validator of the chosen prompt version: strict schema, exact quotes, ids, groups, critical links and
    hard-gate provenance. v2.2 also checks enrollment, short hiring requests and generic other-skills; v2.3 also
    checks open-category examples, explicitly listed generic skills and employment logistics.

    Errors and review warnings reject the reply with their reasons, so the retry can act on them. A reply that is
    valid but says the input is not a usable JD is final, because asking again cannot help, and is marked
    valid_for_cache=False so it is never cached or scored against.
    """
    version = jd_prompt(jd_prompt_version).version
    report = load_rubric_validator(version).validate_rubric(jd_text, value)
    reasons = [*report.errors, *_blocking_warnings(report)]
    if reasons:
        shown = reasons[:MAX_REPORTED_ERRORS]
        extra = len(reasons) - len(shown)
        raise ValidationError("; ".join(shown) + (f"; (+{extra} more)" if extra > 0 else ""))
    usable = bool(report.valid_for_cache)
    notes = list(report.warnings)
    if not usable:  # not_a_jd, empty_or_unreadable or multiple_roles
        notes.insert(0, f"NOT_A_RUBRIC: input_status={value['input_status']}; nothing to cache")
    fields = {"json_valid": True, "input_status": value["input_status"], "valid_for_cache": usable,
              "validator_warnings": "; ".join(notes)}
    logistics = _logistics_text(value, version)
    if logistics is not None:  # versions before v2.3 do not separate logistics, so they leave the field out
        fields["logistics_conditions"] = logistics
    return fields


def validate_reply(task: str, item: Item, text: str) -> dict[str, Any]:
    """Return the fields to store for a valid reply. Any problem, including an unexpected exception from a
    validator, becomes a ValidationError so one odd reply can never abort a whole batch."""
    try:
        repeats_refused = task == "jd_extraction" and jd_prompt(item.context.get("jd_prompt_version")).reject_duplicate_keys
        value = parse_strict_object(text, reject_duplicate_keys=repeats_refused)
        if task == "resume_extraction":
            return _check_resume(value)
        if task == "jd_extraction":
            return _check_rubric(value, item.context["jd"], item.context.get("jd_prompt_version"))
        kit = load_pair_kit()
        result = kit.calculate(value, item.context["job_description"], item.context["resume"])
    except ValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - the kit raises AssessmentError (a ValueError) and, rarely, TypeError
        raise ValidationError(f"{type(exc).__name__}: {exc}") from exc
    return {
        "json_valid": True,
        "verdict": result["fit"] if result["status"] == "scored" else "unscorable",
        "final_score": result["score"],
        "model_reason": result["reason"],
        "needs_human_review": result["review"],
        "score_breakdown": build_breakdown(result, kit.WEIGHTS),
    }


def build_breakdown(result: dict[str, Any], weights: dict[str, float]) -> str:
    """How the calculator turned the evidence into the score, shown beside the model's own reason."""
    if result["status"] != "scored":
        return f"Unscorable: {result['reason']} No score assigned."
    parts = []
    for key, weight in weights.items():
        value = result["dimensions"][key]
        parts.append(
            f"{DIM_LABELS[key]} n/a (weight redistributed)"
            if value is None
            else f"{DIM_LABELS[key]} {value:g}/10 x {weight:.0%} = {weight * value:.2f}"
        )
    text = f"{'; '.join(parts)}. Weighted before caps = {result['weighted_before_caps']}."
    if result["caps"]:
        text += " Caps: " + "; ".join(f"<= {cap['max']} ({cap['because']})" for cap in result["caps"])
        text += f". Final = {result['score']}."
    else:
        text += f" No caps. Final = {result['score']}."
    text += f" Band: {result['fit']}."
    if result["review_reasons"]:
        text += " Review: " + "; ".join(result["review_reasons"]) + "."
    return text


RESULT_COLUMNS_COMMON = [
    "item_id",
    "model",
    "status",
    "attempt",
    "finish_reason",
    "json_valid",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "tokens_per_sec",
    "prefill_tokens_per_sec",
    "latency_seconds",
    "error",
]
RESULT_COLUMNS_PAIR = ["verdict", "final_score", "model_reason", "needs_human_review", "score_breakdown"]
RESULT_COLUMNS_EXTRACTION = ["input_status", "validator_warnings"]
# valid_for_cache is the rubric's go/no-go: False for a valid answer that is not a usable JD rubric (see _check_rubric).
# logistics_conditions lists the sponsorship, relocation, work-authorisation, shift and location policies (v2.3 on); they
# are not job fit and need their own human-reviewed handling.
RESULT_COLUMNS_JD = ["input_status", "valid_for_cache", "logistics_conditions", "validator_warnings"]


TEXT_COLUMNS = {"resume_extraction": ("resume",), "jd_extraction": ("jd",), "resume_jd_pair": ("resume", "jd")}
# The CSV names its id column after the dataset it came from, so it joins back to that dataset.
ID_COLUMNS = {"resume_extraction": "resume_id", "jd_extraction": "jd_id", "resume_jd_pair": "pair_id"}


def item_texts(task: str, item: Item) -> dict[str, str]:
    """The complete source text of an item, keyed by its CSV column."""
    source = {"resume": item.context.get("resume", ""), "jd": item.context.get("jd", item.context.get("job_description", ""))}
    return {column: source[column] for column in TEXT_COLUMNS[task]}


def result_columns(task: str) -> list[str]:
    extra = {"resume_jd_pair": RESULT_COLUMNS_PAIR, "jd_extraction": RESULT_COLUMNS_JD}.get(task, RESULT_COLUMNS_EXTRACTION)
    return [ID_COLUMNS[task], *TEXT_COLUMNS[task], *RESULT_COLUMNS_COMMON[1:6], *extra,
            *RESULT_COLUMNS_COMMON[6:], "output"]
