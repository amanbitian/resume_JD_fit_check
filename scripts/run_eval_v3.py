"""Score the selected pairs with the v3 prompt package and its deterministic calculator.

Per pair and model (each lane runs on its own GPU):
  1. resume extraction   prompt/1_resume_extraction_system_prompt.txt
  2. JD extraction       prompt/2_jd_extraction_system_prompt.txt
  3. evidence matching   prompt/3_relevancy_scoring_system_prompt.txt   (evidence only, no arithmetic)
  4. score_calculator.recompute() + merge_assessment()                  (Python owns the final score)
  5. evidence_audit.annotate_quotes() verifies extractor quotes against the raw text

The model never produces the score. A pair that cannot be validated stays an error row and is never
turned into a 1/10; an unreadable resume or JD is reported as "unscorable".

Stages: run (resumable) and export (CSV with the final score, reasoning and raw output last).
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V3_DIR = ROOT / "prompt" / "old prompt"  # the v3 package: three prompts, score_calculator.py, evidence_audit.py
sys.path.insert(0, str(V3_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402
from evidence_audit import annotate_quotes  # noqa: E402
from score_calculator import DIM_WEIGHTS, INVALID_JD, INVALID_RESUME, MAX_SKILL_SCORE, merge_assessment, recompute  # noqa: E402

from candidate_fit.client import DualGPUClient, OutputTokenLimitError  # noqa: E402
from run_eval_extremes import CLIENT_CONFIG, DATASET, MODELS, ORCHESTRATOR, _lane_urls, _load_records, _load_selection, _read_jsonl  # noqa: E402

PROMPT_DIR = V3_DIR
OUT_DIR = Path("outputs/eval-v3")
REFERENCE_DATE = "2026-10-17"
CALC_ERRORS = (ValueError, KeyError, TypeError, AttributeError, IndexError)
DIM_LABELS = {
    "skills": "Skills",
    "work_experience": "Work",
    "role_alignment": "Role",
    "domain_alignment": "Domain",
    "years": "Years",
    "education_certifications": "Education/certs",
}
CSV_COLUMNS = [
    "pair_id",
    "resume",
    "jd",
    "model",
    "verdict",
    "final_score",
    "reasoning",
    "needs_human_review",
    "token_count",
    "token_per_sec",
    "output_tokens",
    "output",
]


class StageError(RuntimeError):
    def __init__(self, stage: str, message: str, metrics: dict[str, Any], raw_output: str = "") -> None:
        super().__init__(f"{stage}: {message}")
        self.stage, self.metrics, self.raw_output = stage, metrics, raw_output


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def parse_first_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object in a reply, ignoring code fences and any trailing text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Model response did not contain a JSON object")
    value, _end = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("Model response JSON must be an object")
    return value


def _render(template: str, reference_date: str, **fields: str) -> str:
    text = template.replace("{REFERENCE_DATE}", reference_date)
    for key, value in fields.items():
        text = text.replace("{" + key + "}", value)
    return text


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_resume(value: dict[str, Any]) -> None:
    status = value.get("input_status")
    if status not in {"ok", "partial"} | INVALID_RESUME:
        raise ValueError(f"Invalid resume input_status: {status!r}")
    if status in {"ok", "partial"}:
        for key in ("work_experiences", "skills", "projects", "education", "certifications"):
            if not isinstance(value.get(key), list):
                raise ValueError(f"Resume extraction is missing list {key}")


def validate_jd(value: dict[str, Any]) -> None:
    status = value.get("input_status")
    if status not in {"ok", "partial"} | INVALID_JD:
        raise ValueError(f"Invalid JD input_status: {status!r}")
    if status in {"ok", "partial"}:
        for key in ("skills", "any_of_groups", "key_responsibilities", "requirements"):
            if not isinstance(value.get(key), list):
                raise ValueError(f"JD extraction is missing list {key}")


def check_coverage(jd: dict[str, Any], resume: dict[str, Any], match: dict[str, Any]) -> None:
    """Name the exact IDs a matching reply left out, so the retry note is actionable.

    The calculator enforces the same one-entry-per-ID rule but reports it without the IDs.
    """
    status = (match.get("input_check") or {}).get("status")
    if status != "ok":
        # Matching only runs when both extractions are usable, and the prompt says to continue on partial input.
        raise ValueError(
            f"input_check.status must be 'ok' (got {status!r}): the resume ({resume.get('input_status')}) and JD "
            f"({jd.get('input_status')}) extractions are usable; only not_a_resume, empty_or_unreadable, "
            "multiple_candidates, not_a_jd or multiple_roles make an input insufficient"
        )
    checks = (
        ("skill", jd.get("skills"), "id", match.get("skill_matches"), "jd_skill_id"),
        ("group", jd.get("any_of_groups"), "id", match.get("group_matches"), "group_id"),
        ("responsibility", jd.get("key_responsibilities"), "id", match.get("responsibility_matches"), "jd_responsibility_id"),
        ("requirement", jd.get("requirements"), "id", match.get("requirement_matches"), "jd_requirement_id"),
        ("experience relevance", resume.get("work_experiences"), "id", match.get("experience_relevance"), "resume_experience_id"),
    )
    problems = []
    for label, expected_items, expected_key, got_items, got_key in checks:
        expected = [item.get(expected_key) for item in expected_items or []]
        got = [item.get(got_key) for item in got_items or [] if isinstance(item, dict)]
        missing = sorted(set(expected) - set(got), key=str)
        extra = sorted(set(got) - set(expected), key=str)
        duplicated = sorted({i for i in got if got.count(i) > 1}, key=str)
        for kind, ids in (("missing", missing), ("unexpected", extra), ("duplicated", duplicated)):
            if ids:
                problems.append(f"{label} entries {kind}: {', '.join(map(str, ids))}")
    if problems:
        raise ValueError("every ID needs exactly one entry; " + "; ".join(problems))
    allowed_refs = {"skills_section", "summary"}
    for key in ("work_experiences", "projects", "education", "certifications"):
        allowed_refs |= {item.get("id") for item in resume.get(key) or []}
    for item in match.get("skill_matches") or []:
        unknown = sorted({str(ref) for ref in item.get("resume_refs") or []} - {str(ref) for ref in allowed_refs})
        if unknown:
            problems.append(f"{item.get('jd_skill_id')} cites unknown resume_refs {', '.join(unknown)}; allowed: {', '.join(sorted(map(str, allowed_refs)))}")
    for item in match.get("skill_matches") or []:
        band = MAX_SKILL_SCORE.get(item.get("status"))
        score = item.get("match_score")
        if band and isinstance(score, (int, float)) and not band[0] <= score <= band[1]:
            problems.append(f"{item.get('jd_skill_id')} status {item['status']} needs match_score {band[0]}-{band[1]} (got {score})")
    if problems:
        raise ValueError("; ".join(problems))


def _new_metrics(stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "attempts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_seconds": 0.0,
        "finish_reason": None,
        "cached": False,
        "skipped": False,
    }


def run_stage(client: Any, *, stage: str, system_prompt: str, user_prompt: str, validate, attempts: int = 2):
    """Call the model until its reply parses and validates; token and latency totals include retries."""
    metrics = _new_metrics(stage)
    last: Exception = ValueError("no attempt")
    raw = ""
    for attempt in range(1, attempts + 1):
        note = ""
        if attempt > 1:
            note = (
                "\nYour previous response was invalid or incomplete "
                f"({type(last).__name__}: {str(last)[:200]}). Return one complete, valid JSON object only."
            )
        try:
            completion = client.complete(user_prompt + note, system_prompt=system_prompt, response_schema=None, stage=stage)
        except OutputTokenLimitError as exc:
            metrics["attempts"] = attempt
            raise StageError(stage, f"OutputTokenLimitError: {exc}", metrics, raw) from exc
        usage = completion.usage or {}
        metrics["attempts"] = attempt
        metrics["input_tokens"] += int(usage.get("input_tokens") or 0)
        metrics["output_tokens"] += int(usage.get("output_tokens") or 0)
        metrics["latency_seconds"] = round(metrics["latency_seconds"] + completion.latency_seconds, 4)
        metrics["finish_reason"] = completion.finish_reason
        raw = completion.content
        try:
            value = parse_first_json_object(raw)
            validate(value)
            return value, raw, metrics
        except CALC_ERRORS as exc:
            last = exc
    raise StageError(stage, f"{type(last).__name__}: {last}", metrics, raw)


def _audit_summary(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue["status"]] = counts.get(issue["status"], 0) + 1
    return counts


def process_pair(client: Any, record: Any, *, prompts: dict[str, str], cache: dict[str, Any], cache_path: Path, reference_date: str) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    base = {
        "pair_id": record.pair_id,
        "selection": record.metadata.get("selection"),
        "gold_label": record.gold_label,
        "model": client.model,
        "lane": client.lane,
        "reference_date": reference_date,
    }
    raw_outputs: dict[str, str] = {}

    def extract(kind: str, text: str, system_prompt: str, template: str, field: str, validate):
        key = hashlib.sha256((kind + "\x00" + system_prompt + "\x00" + reference_date + "\x00" + text).encode("utf-8")).hexdigest()
        if key in cache:
            hit = cache[key]
            stages.append({**hit["metrics"], "cached": True})
            raw_outputs[f"{kind}_extraction"] = hit["raw_output"]
            return hit["value"]
        try:
            value, raw, metrics = run_stage(
                client,
                stage=f"{kind}_extraction",
                system_prompt=system_prompt,
                user_prompt=_render(template, reference_date, **{field: text}),
                validate=validate,
            )
        except StageError as exc:
            stages.append(exc.metrics)
            raw_outputs[exc.stage] = exc.raw_output
            raise
        stages.append(metrics)
        raw_outputs[f"{kind}_extraction"] = raw
        entry = {"key": key, "kind": kind, "value": value, "metrics": metrics, "raw_output": raw}
        cache[key] = entry
        with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        return value

    try:
        resume = extract("resume", record.resume, prompts["resume_system"], prompts["resume_user"], "RESUME_TEXT", validate_resume)
        jd = extract("jd", record.job_description, prompts["jd_system"], prompts["jd_user"], "JD_TEXT", validate_jd)

        invalid = None
        if resume["input_status"] in INVALID_RESUME:
            invalid = ("insufficient_resume", f"Resume extractor reported {resume['input_status']}")
        elif jd["input_status"] in INVALID_JD:
            invalid = ("insufficient_jd", f"JD extractor reported {jd['input_status']}")
        if invalid:
            # prompt 3 returns empty arrays on invalid input; the calculator then reports 'unscorable'.
            match = {"input_check": {"status": invalid[0], "notes": invalid[1]}, "needs_human_review": True}
            calc = recompute(jd, resume, match, reference_date)
            skipped = _new_metrics("relevancy_matching")
            skipped["skipped"] = True
            stages.append(skipped)
            raw_outputs["relevancy_matching"] = f"(stage skipped: {invalid[1]})"
        else:
            holder: dict[str, Any] = {}
            match, raw, metrics = run_stage(
                client,
                stage="relevancy_matching",
                system_prompt=prompts["match_system"],
                user_prompt=_render(prompts["match_user"], reference_date, JD_EXTRACTION_JSON=_compact(jd), RESUME_EXTRACTION_JSON=_compact(resume)),
                validate=lambda value: (check_coverage(jd, resume, value), holder.update(calc=recompute(jd, resume, value, reference_date))),
            )
            stages.append(metrics)
            raw_outputs["relevancy_matching"] = raw
            calc = holder["calc"]
    except StageError as exc:
        if exc.stage == "relevancy_matching":
            stages.append(exc.metrics)
            raw_outputs[exc.stage] = exc.raw_output
        return {**base, "status": "error", "error": str(exc), "failed_stage": exc.stage, "stages": stages, "raw_outputs": raw_outputs}

    _, resume_issues = annotate_quotes(resume, record.resume, "resume")
    _, jd_issues = annotate_quotes(jd, record.job_description, "jd")
    audit = {"resume": _audit_summary(resume_issues), "jd": _audit_summary(jd_issues)}
    reasons = list(calc["review_reasons"])
    unverified = sum(audit[k].get(s, 0) for k in audit for s in ("unverified", "ambiguous"))
    if unverified and calc["final_score"] is not None:
        reasons.append(f"{unverified} extractor evidence quote(s) are unverified or ambiguous against the source text")
    result = merge_assessment(match, calc)
    result["review_reasons"] = sorted(set(reasons))
    result["needs_human_review"] = bool(reasons)
    if result["assessment_status"] == "scored" and reasons:
        result["assessment_status"] = "needs_review"
    return {
        **base,
        "status": "ok",
        "result": result,
        "calc": {k: calc[k] for k in ("dimension_scores", "weighted_score", "base_score", "caps", "final_score", "fit_band", "years")},
        "audit": audit,
        "resume_extraction": resume,
        "jd_extraction": jd,
        "stages": stages,
        "raw_outputs": raw_outputs,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_cache(path: Path) -> dict[str, Any]:
    return {row["key"]: row for row in _read_jsonl(path)}


def _load_prompts() -> dict[str, str]:
    read = lambda name: (PROMPT_DIR / name).read_text(encoding="utf-8")  # noqa: E731
    return {
        "resume_system": read("1_resume_extraction_system_prompt.txt"),
        "resume_user": read("1_resume_extraction_user_prompt_template.txt"),
        "jd_system": read("2_jd_extraction_system_prompt.txt"),
        "jd_user": read("2_jd_extraction_user_prompt_template.txt"),
        "match_system": read("3_relevancy_scoring_system_prompt.txt"),
        "match_user": read("3_relevancy_scoring_user_prompt_template.txt"),
    }


def stage_run(args: argparse.Namespace) -> None:
    _lane_urls()  # fail fast if either lane is not the expected model
    config = json.loads(CLIENT_CONFIG.read_text(encoding="utf-8"))
    prompts = _load_prompts()
    records = _load_records(args.dataset)
    if args.selection:
        records = [r for r in records if r.metadata["selection"] in args.selection]
    records.sort(key=lambda r: (not r.metadata["selection"].startswith("smallest"), r.metadata["selection"]))  # quick pairs first
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def run_lane(lane: str) -> dict[str, int]:
        model, tag = MODELS[lane]
        client = DualGPUClient(
            orchestrator_url=ORCHESTRATOR,
            lane=lane,
            model=model,
            client_id=config["client_id"],
            task_label="candidate_fit_eval_v3",
            timeout=args.timeout,
            retries=1,
            max_tokens=args.max_output_tokens,
        )
        out_path, cache_path = OUT_DIR / f"{tag}-v3.jsonl", OUT_DIR / f"{tag}-v3-extractions.jsonl"
        done = {row["pair_id"] for row in _read_jsonl(out_path) if row.get("status") == "ok"}
        cache = _load_cache(cache_path)
        counters = {"scored": 0, "errors": 0, "skipped": 0}
        for record in records:
            if record.pair_id in done:
                counters["skipped"] += 1
                continue
            log(f"{tag}: {record.metadata['selection']} ({record.pair_id}) started")
            result = process_pair(client, record, prompts=prompts, cache=cache, cache_path=cache_path, reference_date=args.reference_date)
            with out_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            if result["status"] == "ok":
                counters["scored"] += 1
                log(f"{tag}: {record.metadata['selection']} final_score={result['result']['final_score']} band={result['result']['fit_band']}")
            else:
                counters["errors"] += 1
                log(f"{tag}: {record.metadata['selection']} ERROR {result['error'][:200]}")
        return {"lane": lane, **counters}

    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        print(json.dumps(list(pool.map(run_lane, MODELS)), indent=2))


def build_reasoning(record: dict[str, Any]) -> str:
    """Plain-text explanation of how Python reached the final score, next to the score in the sheet."""
    if record.get("status") != "ok":
        return f"Pipeline failed at {record.get('failed_stage')}: {record.get('error')}. No score assigned."
    calc, result = record["calc"], record["result"]
    reasons = "; ".join(result["review_reasons"])
    if calc["final_score"] is None:
        return f"Unscorable: {reasons}. No score assigned; this is not a 1/10 and must not enter the reject cohort."
    parts, total, weight_sum = [], 0.0, 0.0
    for key, weight in DIM_WEIGHTS.items():
        value = (calc["dimension_scores"] or {}).get(key)
        label = DIM_LABELS[key]
        if value is None:
            parts.append(f"{label} n/a (unknown, excluded and weights renormalized)")
            continue
        parts.append(f"{label} {value:g}/10 x {weight:.0%} = {weight * value:.2f}")
        total += weight * value
        weight_sum += weight
    text = f"{'; '.join(parts)}. Weighted = {total:.2f}/{weight_sum:.2f} = {calc['weighted_score']} -> base score {calc['base_score']}."
    caps = [tuple(cap) for cap in calc["caps"]]
    if caps:
        text += " Caps: " + ", ".join(f"{name} <= {value}" for name, value in caps) + f". Final = min(base, caps) = {calc['final_score']}."
    else:
        text += f" No hard caps applied. Final = {calc['final_score']}."
    text += f" Band: {calc['fit_band']}."
    years = calc.get("years") or {}
    if years.get("required_min_years") is not None:
        text += f" Years: {years.get('years_used_for_requirement')} counted vs {years.get('required_min_years')} required (ratio {years.get('ratio')})."
    skills = (record.get("result") or {}).get("skill_matches") or []
    if skills:
        counts = {s: sum(1 for m in skills if m.get("status") == s) for s in ("met", "partial", "not_evidenced")}
        text += f" Skills: {counts['met']} met, {counts['partial']} partial, {counts['not_evidenced']} not evidenced."
    if result.get("rationale"):
        text += f" Model rationale: {result['rationale']}"
    if reasons:
        text += f" Needs human review: {reasons}."
    return text


def stage_export(args: argparse.Namespace) -> None:
    selection = _load_selection()["selection"]
    table = pq.read_table(args.dataset, columns=["resume_text", "jd_text"])
    rows: list[dict[str, Any]] = []
    latest: dict[str, dict[str, dict[str, Any]]] = {}
    for _model, tag in MODELS.values():
        latest[tag] = {}
        for record in _read_jsonl(OUT_DIR / f"{tag}-v3.jsonl"):  # last record wins, so a retry supersedes an error
            latest[tag][record["pair_id"]] = record
    for item in selection:
        for model, tag in MODELS.values():
            record = latest[tag].get(item["pair_id"])
            stages = (record or {}).get("stages") or []
            input_tokens = sum(s.get("input_tokens") or 0 for s in stages)
            output_tokens = sum(s.get("output_tokens") or 0 for s in stages)
            seconds = sum(s.get("latency_seconds") or 0 for s in stages)
            ok = bool(record) and record.get("status") == "ok"
            result = (record or {}).get("result") or {}
            if not record:
                output, reasoning, verdict = "NOT RUN", "Not run.", "not run"
            else:
                reasoning = build_reasoning(record)
                verdict = result.get("fit_band") if ok else "error"
                raw = (record.get("raw_outputs") or {})
                output = raw.get("relevancy_matching") or next(iter(reversed(raw.values())), "") if raw else ""
                if not ok:
                    output = f"ERROR: {record.get('error')}"
            rows.append(
                {
                    "pair_id": item["pair_id"],
                    "resume": table["resume_text"][item["parquet_row"]].as_py(),
                    "jd": table["jd_text"][item["parquet_row"]].as_py(),
                    "model": model,
                    "verdict": verdict,
                    "final_score": "" if result.get("final_score") is None else result["final_score"],
                    "reasoning": reasoning,
                    "needs_human_review": result.get("needs_human_review", "") if ok else "",
                    "token_count": input_tokens or "",
                    "token_per_sec": round(output_tokens / seconds, 2) if ok and seconds and output_tokens else "",
                    "output_tokens": output_tokens or "",
                    "output": output,
                }
            )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:  # BOM so Excel reads the UTF-8 text
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {target.resolve()}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["run", "export"])
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--reference-date", default=REFERENCE_DATE)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-output-tokens", type=int, default=8_192)
    parser.add_argument("--selection", nargs="*", help="only these selection tags, e.g. smallest_1 (smoke test)")
    parser.add_argument("--output", default=str(OUT_DIR / "eval_v3_scored.csv"))
    args = parser.parse_args()
    {"run": stage_run, "export": stage_export}[args.stage](args)


if __name__ == "__main__":
    main()
