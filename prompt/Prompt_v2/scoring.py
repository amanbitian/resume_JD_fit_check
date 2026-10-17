#!/usr/bin/env python3
"""Compact JD–resume scoring kit. Python 3.10+, standard library only.

Score existing assessment:
  python scoring.py --jd jd.txt --resume resume.txt --assessment model.json

Call an OpenAI-compatible API (e.g., LM Studio):
  python scoring.py --jd jd.txt --resume resume.txt --model MODEL \
    --api-base http://localhost:1234/v1

Cloud API: supply --api-base and an API key in OPENAI_API_KEY.
No network request is made unless --model is provided.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

WEIGHTS = {"skills": 0.40, "work": 0.30, "role": 0.15, "years": 0.10, "education": 0.05}
STATUS_VALUES = {"met": 10.0, "partial": 5.0, "not_evidenced": 0.0, "uncertain": 5.0}
PRIORITY_VALUES = {"mandatory": 1.0, "preferred": 0.4, "unclear": 0.4}
KINDS = {"skill", "responsibility", "education", "experience", "other"}

class AssessmentError(ValueError):
    pass


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssessmentError(message)


def _contains(source: str, quote: str) -> bool:
    # Accept whitespace/case normalization, but not invented paraphrases.
    norm = lambda s: re.sub(r"\s+", " ", s).strip().casefold()
    return bool(quote.strip() and norm(quote) in norm(source))


def _int_0_10(value: object, field: str) -> None:
    _assert(type(value) is int and 0 <= value <= 10, f"{field} must be integer 0..10")


def validate(data: dict, jd: str, resume: str) -> dict:
    _assert(isinstance(data, dict), "Model output must be a JSON object")
    required_keys = {"status", "checks", "work", "role", "years", "reason", "review"}
    _assert(set(data) == required_keys, f"Expected keys {sorted(required_keys)}")
    _assert(data["status"] in {"scored", "unscorable"}, "Bad status")
    _assert(type(data["review"]) is bool, "review must be boolean")
    _assert(isinstance(data["reason"], str) and 1 <= len(data["reason"].split()) <= 35,
            "reason must be 1..35 words")
    years = data["years"]
    _assert(isinstance(years, dict) and set(years) == {"required", "relevant", "certain"},
            "years must have required,relevant,certain")
    _assert(type(years["certain"]) is bool, "years.certain must be boolean")
    for field in ("required", "relevant"):
        value = years[field]
        _assert(value is None or (type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 70),
                f"years.{field} must be null or finite 0..70")
    if years["certain"]:
        _assert(years["required"] is not None and years["relevant"] is not None,
                "Certain years must have required and relevant")
    if not years["certain"]:
        _assert(years["relevant"] is None, "Uncertain relevant years must be null, not an unsupported estimate")
    if years["required"] is None:
        _assert(years["relevant"] is None and not years["certain"],
                "No explicit years minimum: relevant=null, certain=false")
    checks = data["checks"]
    _assert(isinstance(checks, list), "checks must be a list")
    if data["status"] == "unscorable":
        _assert(not checks and data["work"] is None and data["role"] is None and data["review"],
                "Unscorable: empty checks, null ratings, review=true")
        return data
    _int_0_10(data["work"], "work")
    _int_0_10(data["role"], "role")
    _assert(len(checks) > 0, "No requirements assessed; do not silently give a score")
    _assert(len(checks) <= 120, "More than 120 requirements: split or review input")
    for i, check in enumerate(checks, 1):
        _assert(isinstance(check, dict) and set(check) ==
                {"id", "jd", "kind", "priority", "status", "quote"},
                f"Check {i}: missing/unexpected keys")
        _assert(check["id"] == f"R{i}", f"Check {i}: expected R{i} (unique, ordered)")
        _assert(isinstance(check["jd"], str) and _contains(jd, check["jd"]),
                f"{check['id']}: JD substring not found")
        _assert(isinstance(check["kind"], str) and check["kind"] in KINDS, f"{check['id']}: bad kind")
        _assert(isinstance(check["priority"], str) and check["priority"] in PRIORITY_VALUES, f"{check['id']}: bad priority")
        _assert(isinstance(check["status"], str) and check["status"] in STATUS_VALUES, f"{check['id']}: bad match status")
        if check["status"] in ("met", "partial"):
            _assert(isinstance(check["quote"], str) and _contains(resume, check["quote"]),
                    f"{check['id']}: required resume evidence quote not found")
        else:
            _assert(check["quote"] is None, f"{check['id']}: absent/uncertain needs quote=null")
    return data


def _dimension(checks: list[dict], kind: str) -> float | None:
    eligible = [r for r in checks if r["kind"] == kind]
    if not eligible:
        return None
    weight_sum = sum(PRIORITY_VALUES[r["priority"]] for r in eligible)
    return sum(STATUS_VALUES[r["status"]] * PRIORITY_VALUES[r["priority"]]
               for r in eligible) / weight_sum


def _years_score(required: float | None, relevant: float | None, certain: bool) -> float | None:
    if required is None or required == 0:
        return None
    if not certain or relevant is None:
        return 5.0  # Neutral provisional value, not a claim of qualification.
    ratio = relevant / required
    if ratio >= 1: return 10.0
    if ratio >= .8: return 7.0
    if ratio >= .6: return 5.0
    if ratio >= .4: return 3.0
    return 1.0 if ratio > 0 else 0.0


def _round_half_down(value: float) -> int:
    return math.ceil(value - 0.5 - 1e-9)


def calculate(data: dict, jd: str, resume: str) -> dict:
    validate(data, jd, resume)
    if data["status"] == "unscorable":
        return {"status": "unscorable", "score": None, "fit": "unscorable", "review": True,
                "reason": data["reason"], "checks": 0, "caps": [], "provisional": True}

    checks = data["checks"]
    y = data["years"]
    work_from_checks = _dimension(checks, "responsibility")
    dims = {
        "skills": _dimension(checks, "skill"),
        "work": work_from_checks if work_from_checks is not None else float(data["work"]),
        "role": float(data["role"]),
        "years": _years_score(y["required"], y["relevant"], y["certain"]),
        "education": _dimension(checks, "education"),
    }
    applicable_weight = sum(WEIGHTS[k] for k, v in dims.items() if v is not None)
    _assert(applicable_weight > 0, "No applicable scoring dimensions")
    raw = sum(WEIGHTS[k] * v for k, v in dims.items() if v is not None) / applicable_weight
    base = max(1, min(10, _round_half_down(raw)))
    caps = []
    review_reasons = []
    missing_mandatory = 0
    for check in checks:
        if check["status"] == "uncertain":
            review_reasons.append(f"{check['id']}: uncertain evidence")
        if check["priority"] != "mandatory":
            continue
        if check["kind"] == "other":
            if check["status"] != "met":
                review_reasons.append(f"{check['id']}: mandatory condition needs review")
            continue  # No automated penalties for work-location/authorization conditions.
        if check["kind"] == "experience" and y["required"] is not None:
            continue  # Experience is scored once using the numeric years rule below.
        if check["status"] == "not_evidenced":
            missing_mandatory += 1
            caps.append({"max": 6, "because": f"{check['id']}: explicit must-have not evidenced"})
            review_reasons.append(f"{check['id']}: mandatory qualification not evidenced")
        elif check["status"] in ("partial", "uncertain"):
            caps.append({"max": 8, "because": f"{check['id']}: mandatory item not fully verified"})
            review_reasons.append(f"{check['id']}: mandatory qualification not fully evidenced")
    if missing_mandatory >= 2:
        caps.append({"max": 4, "because": "Two or more explicit must-haves not evidenced"})
    if y["required"] is not None and y["required"] > 0:
        if not y["certain"]:
            caps.append({"max": 8, "because": "Required experience is uncertain"})
            review_reasons.append("Relevant years could not be verified")
        else:
            ratio = y["relevant"] / y["required"]
            if ratio < .5:
                caps.append({"max": 4, "because": "Less than 50% of required relevant years evidenced"})
                review_reasons.append("Shortfall against required relevant years")
            elif ratio < .8:
                caps.append({"max": 6, "because": "Less than 80% of required relevant years evidenced"})
                review_reasons.append("Shortfall against required relevant years")
    score = min([base] + [c["max"] for c in caps])
    bands = {1: "no_fit", 2: "no_fit", 3: "weak_fit", 4: "weak_fit",
             5: "potential_fit", 6: "potential_fit", 7: "good_fit", 8: "good_fit",
             9: "strong_fit", 10: "strong_fit"}
    review = bool(data["review"] or review_reasons or any(c["max"] < 10 for c in caps))
    return {
        "status": "scored", "score": score, "fit": bands[score],
        "weighted_before_caps": round(raw, 3),
        "dimensions": {k: (round(v, 3) if v is not None else None) for k, v in dims.items()},
        "caps": caps, "review": review, "review_reasons": sorted(set(review_reasons)),
        "reason": data["reason"], "checks": len(checks),
        "provisional": True, "human_decision_required": True,
        "score_policy": "compact-v4.1",
    }


def call_model(prompt: str, jd: str, resume: str, model: str, base: str,
               key: str, max_tokens: int, json_mode: bool) -> tuple[dict, dict]:
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({"job_description": jd, "resume": resume}, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + key,
    })
    try:
        with urllib.request.urlopen(request, timeout=300) as reply:
            response = json.loads(reply.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise AssessmentError(f"API HTTP {exc.code}: {detail}") from exc
    choices = response.get("choices") or []
    _assert(bool(choices), "API returned no choices")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise AssessmentError("Model reached output token limit; reject partial JSON")
    text = (choice.get("message") or {}).get("content")
    _assert(isinstance(text, str) and bool(text.strip()), "Missing model JSON content")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssessmentError(f"Invalid model JSON: {exc}") from exc
    return parsed, response.get("usage") or {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact JD–resume scoring, validation and optional API inference")
    ap.add_argument("--jd", required=True, help="UTF-8 job description text file")
    ap.add_argument("--resume", required=True, help="UTF-8 resume text file")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--assessment", help="Existing model JSON assessment file")
    source.add_argument("--model", help="Model ID on an OpenAI-compatible server")
    ap.add_argument("--api-base", default="http://localhost:1234/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--json-mode", action="store_true", help="Request server JSON object mode if supported")
    ap.add_argument("--output", help="Optional final result path (JSON)")
    args = ap.parse_args()
    try:
        jd = Path(args.jd).read_text(encoding="utf-8")
        resume = Path(args.resume).read_text(encoding="utf-8")
        _assert(jd.strip() and resume.strip(), "JD and resume must be nonempty")
        usage = {}
        if args.assessment:
            assessment = json.loads(Path(args.assessment).read_text(encoding="utf-8"))
        else:
            prompt = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
            assessment, usage = call_model(prompt, jd, resume, args.model, args.api_base,
                os.environ.get(args.api_key_env, "local-not-needed"), args.max_tokens, args.json_mode)
        result = calculate(assessment, jd, resume)
        if usage:
            result["api_usage"] = usage
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(output)
        return 0
    except (AssessmentError, OSError, json.JSONDecodeError, urllib.error.URLError) as exc:
        print(f"SCORING_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
