"""Score a 4-row evaluation sample with both prompts on both dual-GPU lanes and export to Excel.

Stages (run in order, each is resumable):
  select  pick shortest / longest (by exact token count) plus two seeded-random pairs
  run     score the sample with the single-pass and optimized prompts on each loaded lane
  export  combine every JSONL result into one Excel workbook

Only the resume and JD text reach the model; labels and provenance columns stay local.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

import pyarrow.parquet as pq

from candidate_fit.client import _LOCAL_OPENER, DualGPUClient, require_loopback
from candidate_fit.dataset import PairRecord
from candidate_fit.optimized import score_dataset_optimized
from candidate_fit.scorer import ScoreOptions, score_dataset

ORCHESTRATOR = "http://127.0.0.1:8090"
DATASET = "Input/Candidate_JD_eval_159k.parquet"
OUT_DIR = Path("outputs/eval-sample")
CLIENT_CONFIG = Path(".candidate-fit-client.json")
THREE_CLASS = {"No Fit", "Potential Fit", "Good Fit"}
SHORTLIST = 3000  # candidates per extreme to tokenize exactly; char length only shortlists, bounds are checked below
PROMPTS = {"single_pass": "Single-pass prompt", "optimized": "Optimized 3-stage prompt"}
# lane -> (model catalog name, short tag used in file names and sheet names)
MODELS = {
    "r9700": ("Qwen3.8-27B-Q4_K_M", "qwen3.8-27b"),
    "9070xt": ("gemma-4-12B-it-QAT-Q4_0", "gemma-4-12b"),
}


def _get(path: str) -> Any:
    with _LOCAL_OPENER.open(require_loopback(ORCHESTRATOR) + path, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _tokenize_count(base_url: str, text: str) -> int:
    from urllib import request

    req = request.Request(
        require_loopback(base_url) + "/tokenize",
        data=json.dumps({"content": text, "add_special": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _LOCAL_OPENER.open(req, timeout=60) as response:
        return len(json.loads(response.read().decode("utf-8"))["tokens"])


def _lane_urls() -> dict[str, str]:
    servers = {s["target"]: s for s in _get("/api/status")["servers"]}
    missing = set(MODELS) - set(servers)
    if missing:
        raise SystemExit(f"Lane(s) not deployed: {sorted(missing)}. Deploy with /api/clients/deploy_lane first.")
    for lane, (model, _tag) in MODELS.items():
        if servers[lane]["model"] != model or not servers[lane]["alive"]:
            raise SystemExit(f"Lane {lane} is running {servers[lane]['model']!r}, expected {model!r}")
    return {lane: servers[lane]["base_url"] for lane in MODELS}


def _char_lengths(table_path: str) -> list[tuple[int, int]]:
    import pyarrow.compute as pc

    lengths: list[tuple[int, int]] = []
    parquet = pq.ParquetFile(table_path)
    for batch in parquet.iter_batches(batch_size=8192, columns=["resume_text", "jd_text"]):
        resume = pc.utf8_length(batch["resume_text"]).fill_null(0).to_pylist()
        jd = pc.utf8_length(batch["jd_text"]).fill_null(0).to_pylist()
        lengths.extend(zip(resume, jd))
    return lengths


def stage_select(args: argparse.Namespace) -> None:
    urls = _lane_urls()
    lengths = _char_lengths(args.dataset)
    order = sorted(range(len(lengths)), key=lambda i: (sum(lengths[i]), i))
    table = pq.read_table(args.dataset)
    cols = {name: table[name] for name in ("resume_text", "jd_text")}

    def tokens(index: int, lane: str) -> int:
        return sum(_tokenize_count(urls[lane], cols[c][index].as_py() or "") for c in cols)

    # Exact token counts under the primary (Qwen) tokenizer decide the extremes; ties break on row index.
    primary = "r9700"
    low_pool, high_pool = order[:SHORTLIST], order[-SHORTLIST:]
    with ThreadPoolExecutor(max_workers=8) as pool:
        counted = dict(zip(low_pool + high_pool, pool.map(lambda i: tokens(i, primary), low_pool + high_pool)))
    shortest = min(low_pool, key=lambda i: (counted[i], i))
    longest = max(high_pool, key=lambda i: (counted[i], -i))
    # Proof the shortlist is sufficient: a row outside it has at most max_chars_outside characters, so it
    # cannot exceed that many chars times the densest tokens/char seen; symmetrically for the minimum.
    outside_high = sum(lengths[order[-SHORTLIST - 1]])
    outside_low = sum(lengths[order[SHORTLIST]])
    max_density = max(counted[i] / max(sum(lengths[i]), 1) for i in high_pool)
    min_density = min(counted[i] / max(sum(lengths[i]), 1) for i in low_pool)
    longest_bound, shortest_bound = outside_high * max_density, outside_low * min_density
    print(
        f"longest={counted[longest]} tokens (any row outside the shortlist is <= {longest_bound:.0f}); "
        f"shortest={counted[shortest]} tokens (any row outside is >= {shortest_bound:.0f})"
    )
    if longest_bound >= counted[longest] or shortest_bound <= counted[shortest]:
        raise SystemExit("Shortlist cannot prove the extremes; raise SHORTLIST")
    rng = random.Random(args.seed)
    pool = [i for i in range(len(lengths)) if i not in (shortest, longest)]
    randoms = rng.sample(pool, 2)
    picks = [("shortest", shortest), ("longest", longest), ("random_1", randoms[0]), ("random_2", randoms[1])]

    selection = []
    for tag, index in picks:
        meta = {c: table[c][index].as_py() for c in ("source_dataset", "source_file", "source_row", "label")}
        selection.append(
            {
                "selection": tag,
                "parquet_row": index,
                "pair_id": f"{Path(args.dataset).name}:{index}",
                **meta,
                "resume_chars": lengths[index][0],
                "jd_chars": lengths[index][1],
                **{f"{MODELS[lane][1]}_tokens": tokens(index, lane) for lane in MODELS},
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": args.dataset, "seed": args.seed, "rows": len(lengths), "selection": selection}
    (OUT_DIR / "selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


def _load_records(dataset: str) -> list[tuple[str, PairRecord]]:
    selection = json.loads((OUT_DIR / "selection.json").read_text(encoding="utf-8"))["selection"]
    table = pq.read_table(dataset)
    records = []
    for item in selection:
        i = item["parquet_row"]
        records.append(
            (
                item["selection"],
                PairRecord(
                    pair_id=item["pair_id"],
                    source_file=str(Path(dataset).resolve()),
                    row_index=i,
                    resume=table["resume_text"][i].as_py(),
                    job_description=table["jd_text"][i].as_py(),
                    gold_label=table["label"][i].as_py(),
                    # Local provenance only: metadata is never sent to the model.
                    metadata={
                        "selection": item["selection"],
                        "origin_dataset": item["source_dataset"],
                        "origin_file": item["source_file"],
                        "origin_row": item["source_row"],
                    },
                ),
            )
        )
    return records


def stage_run(args: argparse.Namespace) -> None:
    _lane_urls()  # fail fast if either lane is not the expected model
    config = json.loads(CLIENT_CONFIG.read_text(encoding="utf-8"))
    records = [record for _tag, record in _load_records(args.dataset)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "deployment.json").write_text(
        json.dumps({"captured_at": datetime.now(timezone.utc).isoformat(), "status": _get("/api/status")}, indent=2) + "\n",
        encoding="utf-8",
    )

    def run_lane(lane: str) -> dict[str, Any]:
        model, tag = MODELS[lane]
        client = DualGPUClient(
            orchestrator_url=ORCHESTRATOR,
            lane=lane,
            model=model,
            client_id=config["client_id"],
            task_label="candidate_fit_eval_sample",
            timeout=args.timeout,
            retries=1,
            max_tokens=args.max_output_tokens,
        )
        single = score_dataset(
            records, client=client, output_path=OUT_DIR / f"{tag}-single_pass.jsonl", options=ScoreOptions(max_errors=4)
        )
        optimized = score_dataset_optimized(
            records,
            client=client,
            output_path=OUT_DIR / f"{tag}-optimized.jsonl",
            cache_path=OUT_DIR / f"{tag}-analysis-cache.jsonl",
            max_errors=4,
        )
        return {"lane": lane, "model": model, "single_pass": single, "optimized": optimized}

    # One worker per lane: the lanes are independent GPUs, rows within a lane stay sequential.
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        results = list(pool.map(run_lane, MODELS))
    print(json.dumps(results, indent=2))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _flatten(model: str, lane: str, mode: str, row: dict[str, Any], cache: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    gold = row.get("gold_label")
    label = row.get("fit_label")
    comparable = gold in THREE_CLASS
    skills = row.get("skill_assessment") or []
    status_counts = {s: sum(1 for item in skills if item.get("status") == s) for s in ("met", "partial", "not_evidenced")}
    jd, cand = row.get("jd_analysis") or {}, row.get("candidate_analysis") or {}
    usage = row.get("usage") or {}
    out: dict[str, Any] = {
        "Model": model,
        "Lane": lane,
        "Prompt": PROMPTS[mode],
        "Selection": row.get("selection"),
        "Pair ID": row.get("pair_id"),
        "Origin dataset": row.get("origin_dataset"),
        "Origin row": row.get("origin_row"),
        "Gold label": gold,
        "Fit score (0-10)": row.get("fit_score"),
        "Predicted label": label,
        "Matches gold": ("yes" if label == gold else "no") if comparable and label else "n/a (non 3-class gold)",
        "Confidence": row.get("confidence"),
        "Hard requirements met": row.get("hard_requirements_met"),
        "Unmet mandatory (count)": len(row.get("unmet_mandatory_requirements") or []),
        "Unmet mandatory": "; ".join(row.get("unmet_mandatory_requirements") or []),
        "Rationale": row.get("rationale"),
        "JD title": jd.get("job_title") or jd.get("primary_role"),
        "JD functional domain": jd.get("primary_functional_domain"),
        "JD industry domain": jd.get("primary_industry_domain"),
        "Candidate functional domain": cand.get("primary_functional_domain"),
        "Candidate industry domain": cand.get("primary_industry_domain"),
        "Candidate total years": cand.get("total_years_experience"),
        "JD skills assessed": len(skills),
        "Skills met": status_counts["met"],
        "Skills partial": status_counts["partial"],
        "Skills not evidenced": status_counts["not_evidenced"],
        "Requirement checks": len(row.get("requirement_checks") or []),
        "Resume chars": row.get("input_resume_chars"),
        "JD chars": row.get("input_jd_chars"),
        "Input tokens (scoring call)": usage.get("input_tokens"),
        "Output tokens (scoring call)": usage.get("output_tokens"),
        "Scoring call latency (s)": row.get("latency_seconds"),
        "Total pipeline latency (s)": row.get("pipeline_latency_seconds", row.get("latency_seconds")),
        "Scoring attempts": row.get("pair_attempts", 1),
        "Finish reason": row.get("finish_reason"),
        "Status": row.get("status"),
        "Error": row.get("error"),
        "dgpu run id": row.get("dgpu_run_id"),
    }
    for key, value in (row.get("dimension_scores") or {}).items():
        out[f"Dim: {key}"] = value
    if mode == "optimized" and row.get("analysis_cache"):
        for kind, label_ in (("jd", "JD extraction"), ("resume", "Resume extraction")):
            entry = cache.get((kind, row["analysis_cache"][f"{kind}_hash"]))
            if entry:
                out[f"{label_} latency (s)"] = round(entry.get("latency_seconds") or 0, 3)
                out[f"{label_} output tokens"] = (entry.get("usage") or {}).get("output_tokens")
    out["Raw JSON"] = json.dumps(
        {k: row.get(k) for k in ("jd_analysis", "candidate_analysis", "skill_assessment", "requirement_checks")},
        ensure_ascii=False,
    )[:32000]
    return out


def stage_export(args: argparse.Namespace) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    selection = json.loads((OUT_DIR / "selection.json").read_text(encoding="utf-8"))
    order = {item["selection"]: n for n, item in enumerate(selection["selection"])}
    prompt_order = {name: n for n, name in enumerate(PROMPTS.values())}
    by_pair = {item["pair_id"]: item for item in selection["selection"]}
    per_model: dict[str, list[dict[str, Any]]] = {}
    for lane, (model, tag) in MODELS.items():
        cache = {(r["kind"], r["text_hash"]): r for r in _read_jsonl(OUT_DIR / f"{tag}-analysis-cache.jsonl") if r.get("status") == "ok"}
        rows = []
        for mode in PROMPTS:
            raw = _read_jsonl(OUT_DIR / f"{tag}-{mode}.jsonl")
            latest: dict[str, dict[str, Any]] = {}
            for row in raw:  # keep the last record per pair so a successful retry supersedes an error
                latest[row["pair_id"]] = row
            for row in latest.values():  # error rows carry no dataset metadata; backfill from the selection
                item = by_pair[row["pair_id"]]
                row.setdefault("selection", item["selection"])
                row.setdefault("gold_label", item["label"])
                row.setdefault("origin_dataset", item["source_dataset"])
                row.setdefault("origin_row", item["source_row"])
            rows.extend(_flatten(model, lane, mode, row, cache) for row in latest.values())
        rows.sort(key=lambda r: (order.get(r["Selection"], 99), prompt_order[r["Prompt"]]))
        per_model[tag] = rows
    combined = sorted(
        (row for rows in per_model.values() for row in rows),
        key=lambda r: (order.get(r["Selection"], 99), prompt_order[r["Prompt"]], r["Model"]),
    )

    # Side-by-side: one line per (row, prompt) comparing the two models.
    side: list[dict[str, Any]] = []
    index = {(r["Selection"], r["Prompt"], r["Lane"]): r for r in combined}
    for item in selection["selection"]:
        for prompt in PROMPTS.values():
            q, g = index.get((item["selection"], prompt, "r9700")), index.get((item["selection"], prompt, "9070xt"))
            if not q and not g:
                continue
            qs, gs = (q or {}).get("Fit score (0-10)"), (g or {}).get("Fit score (0-10)")
            side.append(
                {
                    "Selection": item["selection"],
                    "Prompt": prompt,
                    "Gold label": item["label"],
                    "Qwen score": qs,
                    "Qwen label": (q or {}).get("Predicted label"),
                    "Gemma score": gs,
                    "Gemma label": (g or {}).get("Predicted label"),
                    "Models agree on label": ("yes" if (q or {}).get("Predicted label") == (g or {}).get("Predicted label") else "no")
                    if qs is not None and gs is not None
                    else "n/a",
                    "Score difference (Qwen - Gemma)": (qs - gs) if qs is not None and gs is not None else None,
                    "Qwen latency (s)": (q or {}).get("Total pipeline latency (s)"),
                    "Gemma latency (s)": (g or {}).get("Total pipeline latency (s)"),
                }
            )

    summary = []
    for tag, rows in per_model.items():
        for prompt in PROMPTS.values():
            sub = [r for r in rows if r["Prompt"] == prompt]
            ok = [r for r in sub if r["Status"] == "ok"]
            scorable = [r for r in ok if r["Matches gold"] in ("yes", "no")]
            summary.append(
                {
                    "Model": sub[0]["Model"] if sub else tag,
                    "Prompt": prompt,
                    "Rows scored OK": f"{len(ok)}/{len(sub)}",
                    "Gold comparable rows": len(scorable),
                    "Matches gold": sum(r["Matches gold"] == "yes" for r in scorable),
                    "Mean score": round(sum(r["Fit score (0-10)"] for r in ok) / len(ok), 2) if ok else None,
                    "Mean latency (s)": round(sum(r["Total pipeline latency (s)"] or 0 for r in ok) / len(ok), 1) if ok else None,
                }
            )

    contract = [
        {"Item": "Run date (UTC)", "Value": datetime.now(timezone.utc).isoformat()},
        {"Item": "Dataset", "Value": f"{selection['dataset']} ({selection['rows']:,} rows)"},
        {"Item": "Random seed", "Value": selection["seed"]},
        {"Item": "Endpoint", "Value": f"{ORCHESTRATOR}/api/chat (loopback only, proxies bypassed, store_history per client default; see README)"},
        {"Item": "Sampling", "Value": "temperature 0, seed 42, reasoning off, max_tokens 6144"},
        {"Item": "Qwen", "Value": "Qwen3.8-27B-Q4_K_M on r9700, context 24576"},
        {"Item": "Gemma", "Value": "gemma-4-12B-it-QAT-Q4_0 on 9070xt, context 24576"},
        {"Item": "Single-pass prompt", "Value": "candidate_fit.prompt.ACTIVE_SYSTEM_PROMPT (compact wire format)"},
        {"Item": "Optimized prompt", "Value": "candidate_fit.optimized JD_ANALYSIS_PROMPT -> RESUME_ANALYSIS_PROMPT -> PAIR_SCORING_PROMPT"},
        {"Item": "Gold label caveat", "Value": "Rows from the Vaishnavi source carry binary 0/1 labels; they are not compared against the 3-class predictions."},
    ]

    wb = Workbook()
    wb.remove(wb.active)
    header_fill = PatternFill("solid", fgColor="1F3864")

    def sheet(name: str, rows: list[dict[str, Any]], wrap: tuple[str, ...] = ()) -> None:
        ws = wb.create_sheet(name[:31])
        if not rows:
            ws.append(["No rows"])
            return
        columns = list(dict.fromkeys(key for row in rows for key in row))
        ws.append(columns)
        for row in rows:
            ws.append([row.get(c) if not isinstance(row.get(c), (dict, list)) else json.dumps(row.get(c)) for c in columns])
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = header_fill
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for n, column in enumerate(columns, 1):
            width = 60 if column in wrap else min(max(len(str(column)) + 2, 14), 28)
            ws.column_dimensions[get_column_letter(n)].width = width
            if column in wrap:
                for cell in ws[get_column_letter(n)][1:]:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

    wrap = ("Rationale", "Unmet mandatory", "Raw JSON")
    sheet("Summary", summary)
    sheet("Selection", selection["selection"])
    sheet("Combined (16 rows)", combined, wrap)
    sheet("Qwen vs Gemma", side)
    for tag, rows in per_model.items():
        sheet(tag, rows, wrap)
    sheet("Run settings", contract, ("Value",))
    wb["Run settings"].column_dimensions["A"].width = 22
    wb["Run settings"].column_dimensions["B"].width = 110

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    wb.save(target)
    print(f"Wrote {target.resolve()} with sheets: {wb.sheetnames}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["select", "run", "export"])
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-output-tokens", type=int, default=6_144)
    parser.add_argument("--output", default="outputs/candidate_fit_eval_sample.xlsx")
    args = parser.parse_args()
    {"select": stage_select, "run": stage_run, "export": stage_export}[args.stage](args)


if __name__ == "__main__":
    main()
