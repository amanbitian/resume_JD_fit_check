"""Score the selected pairs with the single-prompt Prompt_v2 kit (compact-v4.1) and export a CSV.

One model call per pair and model (each lane runs on its own GPU):
  1. prompt/Prompt_v2/prompt.txt is the system prompt; the user message is
     {"job_description": ..., "resume": ...} as the kit's README specifies
  2. prompt/Prompt_v2/scoring.py validates the reply (every JD/resume quote must appear verbatim)
     and computes the score, fit band, caps and review flags in Python

The model never produces the score. A reply that fails validation is retried once with the validation
message; if it still fails the pair is an error row, never a 1/10.

dual_gpu_setup contract (API_CONTRACT.md / CLIENT_API.md):
  - step 0: launch_service.py is run first (one-shot and idempotent), so the service is up
  - step 2: each lane is deployed with the real token budget (input_tokens + max_output_tokens)
  - step 7: each lane this script loaded is released with {"lane", "model"} when that lane's pairs are
    done, on success and on failure. A model that was already serving the lane is never unloaded.

Results are tagged with a hash of prompt.txt. A run only reuses results from the same prompt, and the
export only shows results from the current prompt.

Stages: run (resumable) and export (CSV with the score, the model's reason and the raw output last).
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402

from candidate_fit.client import _LOCAL_OPENER, DualGPUClient, OutputTokenLimitError, require_loopback  # noqa: E402
from run_eval_extremes import CLIENT_CONFIG, DATASET, MODELS, ORCHESTRATOR, _get, _load_records, _load_selection, _read_jsonl  # noqa: E402

KIT_DIR = ROOT / "prompt" / "Prompt_v2"
OUT_DIR = Path("outputs/eval-prompt-v2")
SERVICE_DIR = Path("F:/Projects/Dual_gpu_setup")
SERVICE_CONFIG = SERVICE_DIR / "example.dual_gpu.toml"
CONTEXT_WINDOW = 24_576
PROMPT_OVERHEAD_TOKENS = 1_500  # system prompt + JSON wrapper around the resume and JD
DIM_LABELS = {"skills": "Skills", "work": "Work", "role": "Role", "years": "Years", "education": "Education"}
CSV_COLUMNS = [
    "pair_id",
    "resume",
    "jd",
    "model",
    "verdict",
    "final_score",
    "model_reason",
    "needs_human_review",
    "score_breakdown",
    "token_count",
    "token_per_sec",
    "output_tokens",
    "output",
]


def load_kit() -> Any:
    """Import prompt/Prompt_v2/scoring.py by path so its generic module name cannot collide."""
    spec = importlib.util.spec_from_file_location("prompt_v2_scoring", KIT_DIR / "scoring.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]


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


# --- dual_gpu_setup client contract -------------------------------------------------------------------


def api_post(path: str, body: dict[str, Any], timeout: float = 900.0) -> tuple[int, dict[str, Any]]:
    """POST JSON to the orchestrator; returns (status, payload) and never raises for an HTTP error status."""
    req = request.Request(
        require_loopback(ORCHESTRATOR) + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _LOCAL_OPENER.open(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(text)
        except ValueError:
            payload = {"error": text[:500]}
        return exc.code, payload if isinstance(payload, dict) else {"error": text[:500]}


def ensure_service(config: Path = SERVICE_CONFIG) -> None:
    """CLIENT_API step 0: one-shot, idempotent launcher; exits at once if the service is already healthy."""
    result = subprocess.run(
        [sys.executable, str(SERVICE_DIR / "launch_service.py"), "--config", str(config), "--no-browser"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise SystemExit(f"launch_service.py could not confirm the service is up:\n{result.stdout}\n{result.stderr}")
    log(f"dual_gpu_setup service ready: {result.stdout.strip()[:200]}")


def get_status(attempts: int = 6) -> dict[str, Any]:
    """GET /api/status, riding out the resets a just-started service gives its first requests."""
    for attempt in range(attempts):
        try:
            return _get("/api/status")
        except (OSError, ValueError):  # URLError and ConnectionResetError are OSErrors
            if attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def ensure_lane(lane: str, model: str, *, input_tokens: int, max_output_tokens: int) -> bool:
    """Deploy `model` on `lane` unless it already serves it. True only if this call loaded it (so we must release it)."""
    server = {s["target"]: s for s in get_status().get("servers", [])}.get(lane)
    if server and server.get("alive") and server.get("model") == model:
        log(f"{lane}: {model} already loaded; reusing it and leaving it loaded afterwards")
        return False
    status, payload = api_post(
        "/api/clients/deploy_lane",
        {
            "model": model,
            "lane": lane,
            "context_window": CONTEXT_WINDOW,
            "input_tokens": input_tokens,
            "max_output_tokens": max_output_tokens,
        },
        timeout=1800,
    )
    if status != 200:
        raise RuntimeError(f"deploy_lane {lane} {model}: HTTP {status}: {payload.get('error')}")
    log(f"{lane}: loaded {model}")
    return True


def release_lane(lane: str, model: str) -> bool:
    """CLIENT_API step 7: unload `model` from `lane` only if it still serves it. Never the bare {} / force form."""
    for _ in range(5):
        status, payload = api_post("/api/unload", {"lane": lane, "model": model}, timeout=120)
        if status != 409:  # 409: a load or unload is in progress, retry shortly
            return status == 200 and bool(payload.get("released"))
        time.sleep(2)
    return False


# --- scoring ------------------------------------------------------------------------------------------


def score_pair(client: Any, record: Any, *, kit: Any, system_prompt: str, attempts: int = 2) -> dict[str, Any]:
    """One model call (plus one corrective retry); the score always comes from kit.calculate()."""
    base = {
        "pair_id": record.pair_id,
        "selection": record.metadata.get("selection"),
        "gold_label": record.gold_label,
        "model": client.model,
        "lane": client.lane,
        "prompt_version": "Prompt_v2 compact-v4.1",
        "prompt_sha256": prompt_hash(system_prompt),
    }
    metrics: dict[str, Any] = {"attempts": 0, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0, "finish_reason": None}
    user_prompt = json.dumps({"job_description": record.job_description, "resume": record.resume}, ensure_ascii=False)
    raw = ""
    last_error: Exception = ValueError("no attempt")
    for attempt in range(1, attempts + 1):
        note = ""
        if attempt > 1:
            note = f"\n\nYour previous JSON was rejected: {last_error}. Return one corrected JSON object only, in exactly the required structure."
        metrics["attempts"] = attempt
        try:
            completion = client.complete(user_prompt + note, system_prompt=system_prompt, response_schema=None, stage="prompt_v2_scoring")
        except OutputTokenLimitError as exc:
            return {**base, "status": "error", "error": f"OutputTokenLimitError: {exc}", "raw_output": raw, **metrics}
        except Exception as exc:  # noqa: BLE001 - one failed pair must not stop the lane
            return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}", "raw_output": raw, **metrics}
        usage = completion.usage or {}
        metrics["input_tokens"] += int(usage.get("input_tokens") or 0)
        metrics["output_tokens"] += int(usage.get("output_tokens") or 0)
        metrics["latency_seconds"] = round(metrics["latency_seconds"] + completion.latency_seconds, 4)
        metrics["finish_reason"] = completion.finish_reason
        raw = completion.content
        try:
            assessment = parse_first_json_object(raw)
            result = kit.calculate(assessment, record.job_description, record.resume)
        except (kit.AssessmentError, ValueError, TypeError, KeyError) as exc:
            last_error = exc
            continue
        return {**base, "status": "ok", "assessment": assessment, "result": result, "raw_output": raw, **metrics,
                "scored_at": datetime.now(timezone.utc).isoformat()}
    return {**base, "status": "error", "error": f"{type(last_error).__name__}: {last_error}", "raw_output": raw, **metrics}


def build_breakdown(result: dict[str, Any], weights: dict[str, float]) -> str:
    """How Python turned the model's evidence into the score, shown beside the model's own reason."""
    if result["status"] != "scored":
        return f"Unscorable: {result['reason']} No score assigned; this is not a 1/10."
    parts = []
    for key, weight in weights.items():
        value = result["dimensions"][key]
        if value is None:
            parts.append(f"{DIM_LABELS[key]} n/a (weight redistributed)")
        else:
            parts.append(f"{DIM_LABELS[key]} {value:g}/10 x {weight:.0%} = {weight * value:.2f}")
    text = f"{'; '.join(parts)}. Weighted before caps = {result['weighted_before_caps']}."
    if result["caps"]:
        text += " Caps: " + "; ".join(f"<= {cap['max']} ({cap['because']})" for cap in result["caps"])
        text += f". Final = min(rounded weighted score, caps) = {result['score']}."
    else:
        text += f" No caps. Final = {result['score']}."
    text += f" Band: {result['fit']}."
    if result["review_reasons"]:
        text += " Review: " + "; ".join(result["review_reasons"]) + "."
    return text


def current_records(path: Path, system_prompt_hash: str) -> dict[str, dict[str, Any]]:
    """Latest record per pair, counting only results produced by the current prompt.txt."""
    latest: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):  # last record wins, so a retry supersedes an error
        if record.get("prompt_sha256") == system_prompt_hash:
            latest[record["pair_id"]] = record
    return latest


def stage_run(args: argparse.Namespace) -> None:
    config = json.loads(CLIENT_CONFIG.read_text(encoding="utf-8"))
    kit = load_kit()
    system_prompt = (KIT_DIR / "prompt.txt").read_text(encoding="utf-8")
    system_hash = prompt_hash(system_prompt)
    records = _load_records(args.dataset)
    if args.selection:
        records = [r for r in records if r.metadata["selection"] in args.selection]
    records.sort(key=lambda r: (not r.metadata["selection"].startswith("smallest"), r.metadata["selection"]))  # quick pairs first
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    largest = max(item["qwen3.8-27b_tokens"] for item in _load_selection()["selection"])
    input_tokens = largest + PROMPT_OVERHEAD_TOKENS  # largest resume+JD pair plus the prompt, for the deploy sizing check
    log(f"prompt {system_hash}; deploy token budget: {input_tokens} in + {args.max_output_tokens} out")

    ensure_service(Path(args.service_config))

    def run_lane(lane: str) -> dict[str, Any]:
        model, tag = MODELS[lane]
        counters: dict[str, Any] = {"lane": lane, "scored": 0, "errors": 0, "skipped": 0}
        loaded = False
        try:
            loaded = ensure_lane(lane, model, input_tokens=input_tokens, max_output_tokens=args.max_output_tokens)
            client = DualGPUClient(
                orchestrator_url=ORCHESTRATOR,
                lane=lane,
                model=model,
                client_id=config["client_id"],
                task_label="candidate_fit_eval_prompt_v2",
                timeout=args.timeout,
                retries=1,
                max_tokens=args.max_output_tokens,
            )
            out_path = OUT_DIR / f"{tag}-prompt_v2.jsonl"
            done = {pair_id for pair_id, row in current_records(out_path, system_hash).items() if row.get("status") == "ok"}
            for record in records:
                if record.pair_id in done:
                    counters["skipped"] += 1
                    continue
                log(f"{tag}: {record.metadata['selection']} ({record.pair_id}) started")
                result = score_pair(client, record, kit=kit, system_prompt=system_prompt)
                with out_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                if result["status"] == "ok":
                    counters["scored"] += 1
                    log(f"{tag}: {record.metadata['selection']} score={result['result']['score']} fit={result['result']['fit']}")
                else:
                    counters["errors"] += 1
                    log(f"{tag}: {record.metadata['selection']} ERROR {result['error'][:200]}")
        except Exception as exc:  # noqa: BLE001 - report it, but let the other lane finish
            counters["failed"] = f"{type(exc).__name__}: {exc}"
            log(f"{tag}: lane failed: {counters['failed'][:300]}")
        finally:
            if loaded:  # only what this script loaded, and per lane as soon as that lane's queue is empty
                counters["released"] = release_lane(lane, model)
                log(f"{tag}: released {lane} ({model}): {counters['released']}")
        return counters

    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        print(json.dumps(list(pool.map(run_lane, MODELS)), indent=2))


def _cell(value: Any) -> Any:
    return "" if value is None else value


def stage_export(args: argparse.Namespace) -> None:
    kit = load_kit()
    system_hash = prompt_hash((KIT_DIR / "prompt.txt").read_text(encoding="utf-8"))
    selection = _load_selection()["selection"]
    table = pq.read_table(args.dataset, columns=["resume_text", "jd_text"])
    latest = {tag: current_records(OUT_DIR / f"{tag}-prompt_v2.jsonl", system_hash) for _model, tag in MODELS.values()}
    rows: list[dict[str, Any]] = []
    for item in selection:
        for model, tag in MODELS.values():
            record = latest[tag].get(item["pair_id"])
            ok = bool(record) and record.get("status") == "ok"
            result = record["result"] if ok else {}
            output_tokens = (record or {}).get("output_tokens") or 0
            seconds = (record or {}).get("latency_seconds") or 0
            if not record:
                verdict, reason, breakdown, output = "not run", "Not run with the current prompt.", "", "NOT RUN"
            elif ok:
                verdict, reason = result["fit"], result["reason"]
                breakdown, output = build_breakdown(result, kit.WEIGHTS), record["raw_output"]
            else:
                verdict, reason = "error", f"Pipeline failed: {record['error']}. No score assigned."
                breakdown, output = "", record.get("raw_output") or f"ERROR: {record['error']}"
            rows.append(
                {
                    "pair_id": item["pair_id"],
                    "resume": table["resume_text"][item["parquet_row"]].as_py(),
                    "jd": table["jd_text"][item["parquet_row"]].as_py(),
                    "model": model,
                    "verdict": verdict,
                    "final_score": _cell(result.get("score")),
                    "model_reason": reason,
                    "needs_human_review": result.get("review", "") if ok else "",
                    "score_breakdown": breakdown,
                    "token_count": (record or {}).get("input_tokens") or "",
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
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-output-tokens", type=int, default=4_096)
    parser.add_argument("--service-config", default=str(SERVICE_CONFIG), help="dual_gpu_setup config passed to launch_service.py")
    parser.add_argument("--selection", nargs="*", help="only these selection tags, e.g. smallest_1 (smoke test)")
    parser.add_argument("--output", default=str(OUT_DIR / "eval_prompt_v2_scored.csv"))
    args = parser.parse_args()
    {"run": stage_run, "export": stage_export}[args.stage](args)


if __name__ == "__main__":
    main()
