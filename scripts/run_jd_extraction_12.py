"""Extract a rubric from 12 job descriptions on two models and save the outputs with metrics as CSV.

Models (one lane each, both loaded at once with the thinking budget set at deploy time):
  r9700   Qwen3.8-27B-Q4_K_M
  9070xt  gemma-4-12B-it-QAT-Q4_0

Selection from Input/jd_dataset.csv: the 2 longest JDs, the 2 shortest, and 8 chosen at random (seeded).
Prompt: prompt/Prompt_v2/Jd_extraction_prompt.txt as the system message; the JD goes in the user message
inside <JOB_DESCRIPTION> tags, as the earlier extraction template did.

dual_gpu_setup contract (API_CONTRACT.md / CLIENT_API.md):
  - launch_service.py starts the service if needed
  - /api/deploy (parallel_models) loads both models with reasoning_budget=512; the script checks that
    the status reports budget 512 before it sends any JD
  - /api/chat for every JD, with the registered client_id
  - /api/unload {"lane", "model"} releases each lane after its queue is finished, on success and failure

Stages: run (resumable per JD and model) and export (CSV: jd_id, jd, output, then the metrics).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval_extremes import CLIENT_CONFIG, _read_jsonl  # noqa: E402
from run_eval_prompt_v2 import api_post, ensure_service, get_status, release_lane  # noqa: E402

JD_CSV = ROOT / "Input" / "jd_dataset.csv"
PROMPT = ROOT / "prompt" / "Prompt_v2" / "Jd_extraction_prompt.txt"
OUT_DIR = ROOT / "outputs" / "jd-extraction-12"
SEED = 20261017
N_EXTREME_EACH_END = 2
N_RANDOM = 8
THINKING_BUDGET = 512
MAX_OUTPUT_TOKENS = 6144
CONTEXT_WINDOW = 16384
INPUT_TOKEN_BUDGET = 5000  # longest JD (8,171 chars) plus the extraction prompt is well under this
LANES = {
    "r9700": ("Qwen3.8-27B-Q4_K_M", "qwen3.8-27b"),
    "9070xt": ("gemma-4-12B-it-QAT-Q4_0", "gemma-4-12b"),
}
CSV_COLUMNS = [
    "jd_id",
    "jd",
    "output",
    "model",
    "status",
    "finish_reason",
    "json_valid",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "cached_input_tokens",
    "tokens_per_sec",
    "prefill_tokens_per_sec",
    "time_to_first_token_ms",
    "prefill_duration_ms",
    "decode_duration_ms",
    "end_to_end_duration_ms",
    "selection",
    "jd_chars",
    "run_id",
    "error",
]


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def select_jds(path: Path = JD_CSV) -> list[dict[str, Any]]:
    csv.field_size_limit(10**9)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [{"jd_id": r["jd_id"], "jd_text": r["jd_text"]} for r in csv.DictReader(handle)]
    ranked = sorted(rows, key=lambda r: (len(r["jd_text"]), r["jd_id"]))
    longest = list(reversed(ranked[-N_EXTREME_EACH_END:]))
    shortest = ranked[:N_EXTREME_EACH_END]
    taken = {r["jd_id"] for r in longest + shortest}
    pool = [r for r in ranked if r["jd_id"] not in taken]
    randoms = random.Random(SEED).sample(pool, N_RANDOM)
    picks = [("longest", r) for r in longest] + [("shortest", r) for r in shortest] + [("random", r) for r in randoms]
    return [{"selection": tag, **row, "jd_chars": len(row["jd_text"])} for tag, row in picks]


def ensure_thinking_budget(models: dict[str, str]) -> None:
    """Load both models with the thinking budget set, then check the service reports it for each lane."""
    entries = [
        {"model": model, "lane": lane, "context_window": CONTEXT_WINDOW, "input_tokens": INPUT_TOKEN_BUDGET,
         "max_output_tokens": MAX_OUTPUT_TOKENS, "reasoning_budget": THINKING_BUDGET}
        for lane, model in models.items()
    ]
    status, payload = api_post("/api/deploy", {"mode": "parallel_models", "models": entries}, timeout=1800)
    if status != 200:
        raise SystemExit(f"/api/deploy failed: HTTP {status}: {payload.get('error')}")
    by_lane = {s["target"]: s for s in get_status().get("servers", [])}
    for lane, model in models.items():
        server = by_lane.get(lane) or {}
        if server.get("model") != model or not server.get("alive"):
            raise SystemExit(f"{lane}: expected {model} alive, got {server.get('model')!r} alive={server.get('alive')}")
        if server.get("reasoning_budget") != THINKING_BUDGET:
            raise SystemExit(f"{lane}: thinking budget is {server.get('reasoning_budget')}, expected {THINKING_BUDGET}")
        log(f"{lane}: {model} loaded, reasoning_mode={server.get('reasoning_mode')} budget={server.get('reasoning_budget')}")


def chat_one(lane: str, client_id: str, system_prompt: str, jd_text: str) -> dict[str, Any]:
    """One /api/chat call. Length-capped and failed answers are kept with their usage, not dropped."""
    body = {
        "client_id": client_id,
        "target": lane,
        "task_label": "jd_extraction_eval_12",
        "workload_kind": "evaluation",
        "connection_type": "batch",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<JOB_DESCRIPTION>\n{jd_text}\n</JOB_DESCRIPTION>"},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "timeout_seconds": 900,
        "store_history": True,
    }
    status, payload = api_post("/api/chat", body, timeout=1000)
    if status != 200:
        return {"status": "failed", "error": f"HTTP {status}: {payload.get('error')}"}
    result = (payload.get("results") or {}).get(lane) or {}
    return {
        "status": result.get("status"),
        "output": result.get("output_text") or "",
        "finish_reason": result.get("finish_reason"),
        "usage": result.get("usage") or {},
        "timing": result.get("timing") or {},
        "run_id": result.get("run_id"),
        "error": result.get("error_text"),
    }


def strict_json_valid(text: str) -> bool:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        json.loads(cleaned)
        return True
    except ValueError:
        return False


def run_lane(lane: str, client_id: str, system_prompt: str, jds: list[dict[str, Any]], out_path: Path) -> dict[str, int]:
    model, tag = LANES[lane]
    done = {row["jd_id"] for row in _read_jsonl(out_path) if row.get("status") == "completed"}
    counts = {"completed": 0, "other": 0, "skipped": 0}
    for jd in jds:
        if jd["jd_id"] in done:
            counts["skipped"] += 1
            continue
        log(f"{tag}: {jd['jd_id']} ({jd['selection']}, {jd['jd_chars']} chars) started")
        started = time.perf_counter()
        result = chat_one(lane, client_id, system_prompt, jd["jd_text"])
        record = {
            "jd_id": jd["jd_id"], "selection": jd["selection"], "jd_chars": jd["jd_chars"], "model": model, "lane": lane,
            **result, "wall_seconds": round(time.perf_counter() - started, 2),
        }
        with out_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        counts["completed" if result["status"] == "completed" else "other"] += 1
        log(f"{tag}: {jd['jd_id']} {result['status']} finish={result.get('finish_reason')} out={(result.get('usage') or {}).get('output_tokens')}")
    return counts


def stage_run(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jds = select_jds()
    (OUT_DIR / "selection.json").write_text(json.dumps(jds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log("selected: " + ", ".join(f"{j['jd_id']}({j['selection']},{j['jd_chars']})" for j in jds))
    system_prompt = PROMPT.read_text(encoding="utf-8")
    client_id = json.loads(CLIENT_CONFIG.read_text(encoding="utf-8"))["client_id"]
    models = {lane: model for lane, (model, _tag) in LANES.items()}

    ensure_service()
    loaded = False
    try:
        ensure_thinking_budget(models)
        loaded = True
        with ThreadPoolExecutor(max_workers=len(LANES)) as pool:
            futures = {
                lane: pool.submit(run_lane, lane, client_id, system_prompt, jds, OUT_DIR / f"{tag}.jsonl")
                for lane, (_model, tag) in LANES.items()
            }
            for lane, future in futures.items():
                log(f"{lane}: {future.result()}")
    finally:
        if loaded:
            for lane, model in models.items():
                released = release_lane(lane, model)
                log(f"{lane}: released {model}: {released}")


def stage_export(args: argparse.Namespace) -> None:
    jds = {j["jd_id"]: j for j in json.loads((OUT_DIR / "selection.json").read_text(encoding="utf-8"))}
    rows: list[dict[str, Any]] = []
    for lane, (model, tag) in LANES.items():
        latest = {r["jd_id"]: r for r in _read_jsonl(OUT_DIR / f"{tag}.jsonl")}  # last record per JD wins
        for jd_id, jd in jds.items():
            r = latest.get(jd_id)
            if r is None:
                rows.append({"jd_id": jd_id, "jd": jd["jd_text"], "output": "NOT RUN", "model": model, "selection": jd["selection"], "jd_chars": jd["jd_chars"]})
                continue
            u, t = r.get("usage") or {}, r.get("timing") or {}
            seconds = (r.get("wall_seconds") or 0)
            output = r.get("output") or ""
            rows.append({
                "jd_id": jd_id,
                "jd": jd["jd_text"],
                "output": output,
                "model": model,
                "status": r.get("status"),
                "finish_reason": r.get("finish_reason") or "",
                "json_valid": strict_json_valid(output) if output else False,
                "input_tokens": _blank(u.get("input_tokens")),
                "output_tokens": _blank(u.get("output_tokens")),
                "thinking_tokens": _blank(u.get("thinking_tokens")),
                "cached_input_tokens": _blank(u.get("cached_input_tokens")),
                "tokens_per_sec": _blank(t.get("tokens_per_second")),
                "prefill_tokens_per_sec": _blank(t.get("prefill_tokens_per_second")),
                "time_to_first_token_ms": _blank(t.get("time_to_first_token_ms")),
                "prefill_duration_ms": _blank(t.get("prefill_duration_ms")),
                "decode_duration_ms": _blank(t.get("decode_duration_ms")),
                "end_to_end_duration_ms": _blank(t.get("end_to_end_duration_ms")),
                "selection": r.get("selection"),
                "jd_chars": r.get("jd_chars"),
                "run_id": r.get("run_id") or "",
                "error": r.get("error") or "",
            })
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {target.resolve()}")


def _blank(value: Any) -> Any:
    return "" if value is None else value


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["run", "export"])
    parser.add_argument("--output", default=str(OUT_DIR / "jd_extraction_12.csv"))
    args = parser.parse_args()
    {"run": stage_run, "export": stage_export}[args.stage](args)


if __name__ == "__main__":
    main()
