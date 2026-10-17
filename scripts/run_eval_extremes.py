"""Score the 4 largest and 4 smallest resume/JD pairs on both dual-GPU lanes and export a CSV.

Stages (run in order, each is resumable):
  select  rank the pairs by exact token count (Qwen tokenizer) and keep the 4 largest and 4 smallest
  run     score the 8 pairs with the single-pass and optimized prompts on each loaded lane
  export  write one CSV row per (pair, model, prompt)

Only the resume and JD text reach the model; the source labels stay local.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
from urllib import request

import pyarrow.compute as pc
import pyarrow.parquet as pq

from candidate_fit.client import _LOCAL_OPENER, DualGPUClient, require_loopback
from candidate_fit.dataset import PairRecord
from candidate_fit.optimized import score_dataset_optimized
from candidate_fit.scorer import ScoreOptions, score_dataset

ORCHESTRATOR = "http://127.0.0.1:8090"
DATASET = "Input/Candidate_JD_eval_159k.parquet"
OUT_DIR = Path("outputs/eval-extremes-8")
CLIENT_CONFIG = Path(".candidate-fit-client.json")
PER_END = 4  # pairs taken from each end of the token-count ranking
SHORTLIST = 3000  # candidates per end counted exactly; character length only shortlists them
PROMPTS = {"single_pass": "single-pass", "optimized": "optimized"}
# lane -> (model catalog name, short tag used in file names)
MODELS = {
    "r9700": ("Qwen3.8-27B-Q4_K_M", "qwen3.8-27b"),
    "9070xt": ("gemma-4-12B-it-QAT-Q4_0", "gemma-4-12b"),
}
CSV_COLUMNS = [
    "pair_id",
    "resume",
    "jd",
    "verdict",
    "model",
    "prompt",
    "token_count",
    "token_per_sec",
    "output_tokens",
    "output",
]


def _get(path: str) -> Any:
    with _LOCAL_OPENER.open(require_loopback(ORCHESTRATOR) + path, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _tokenize_count(base_url: str, text: str) -> int:
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
    lengths: list[tuple[int, int]] = []
    parquet = pq.ParquetFile(table_path)
    for batch in parquet.iter_batches(batch_size=8192, columns=["resume_text", "jd_text"]):
        resume = pc.utf8_length(batch["resume_text"]).fill_null(0).to_pylist()
        jd = pc.utf8_length(batch["jd_text"]).fill_null(0).to_pylist()
        lengths.extend(zip(resume, jd))
    return lengths


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_select(args: argparse.Namespace) -> None:
    urls = _lane_urls()
    lengths = _char_lengths(args.dataset)
    order = sorted(range(len(lengths)), key=lambda i: (sum(lengths[i]), i))
    table = pq.read_table(args.dataset)
    text = {c: table[c] for c in ("resume_text", "jd_text")}

    primary = "r9700"  # the Qwen tokenizer decides the ranking
    low_pool, high_pool = order[:SHORTLIST], order[-SHORTLIST:]
    candidates = low_pool + high_pool
    with ThreadPoolExecutor(max_workers=8) as pool:
        counted = dict(
            zip(
                candidates,
                pool.map(
                    lambda i: sum(_tokenize_count(urls[primary], text[c][i].as_py() or "") for c in text),
                    candidates,
                ),
            )
        )
    largest = sorted(high_pool, key=lambda i: (-counted[i], i))[:PER_END]
    smallest = sorted(low_pool, key=lambda i: (counted[i], i))[:PER_END]

    # Proof the shortlist is sufficient: a row outside it has at most that many characters, so it cannot
    # exceed them times the densest tokens-per-character seen in the pool; symmetrically for the minimum.
    density = {i: counted[i] / max(sum(lengths[i]), 1) for i in candidates}
    longest_bound = sum(lengths[order[-SHORTLIST - 1]]) * max(density[i] for i in high_pool)
    shortest_bound = sum(lengths[order[SHORTLIST]]) * min(density[i] for i in low_pool)
    if longest_bound >= counted[largest[-1]] or shortest_bound <= counted[smallest[-1]]:
        raise SystemExit("Shortlist cannot prove the 4 largest and 4 smallest pairs; raise SHORTLIST")

    picks = [(f"largest_{n}", i) for n, i in enumerate(largest, 1)]
    picks += [(f"smallest_{n}", i) for n, i in enumerate(smallest, 1)]
    selection = []
    for tag, index in picks:
        selection.append(
            {
                "selection": tag,
                "parquet_row": index,
                "pair_id": f"{Path(args.dataset).name}:{index}",
                **{c: table[c][index].as_py() for c in ("source_dataset", "source_file", "source_row", "label")},
                "resume_chars": lengths[index][0],
                "jd_chars": lengths[index][1],
                "qwen3.8-27b_tokens": counted[index],
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "rows": len(lengths),
        "rule": f"{PER_END} largest and {PER_END} smallest by exact Qwen token count (resume + JD)",
        "selection": selection,
    }
    (OUT_DIR / "selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for item in selection:
        print(f"{item['selection']:<12} row {item['parquet_row']:>7}  {item['qwen3.8-27b_tokens']:>6} tokens  {item['label']}")


def _load_selection() -> dict[str, Any]:
    return json.loads((OUT_DIR / "selection.json").read_text(encoding="utf-8"))


def _load_records(dataset: str) -> list[PairRecord]:
    table = pq.read_table(dataset)
    records = []
    for item in _load_selection()["selection"]:
        i = item["parquet_row"]
        records.append(
            PairRecord(
                pair_id=item["pair_id"],
                source_file=str(Path(dataset).resolve()),
                row_index=i,
                resume=table["resume_text"][i].as_py(),
                job_description=table["jd_text"][i].as_py(),
                gold_label=table["label"][i].as_py(),
                # Local provenance only: metadata is never sent to the model.
                metadata={"selection": item["selection"]},
            )
        )
    return records


def stage_run(args: argparse.Namespace) -> None:
    _lane_urls()  # fail fast if either lane is not the expected model
    config = json.loads(CLIENT_CONFIG.read_text(encoding="utf-8"))
    records = _load_records(args.dataset)
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
            task_label="candidate_fit_eval_extremes",
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


def _cell(value: Any) -> Any:
    return "" if value is None else value


def stage_export(args: argparse.Namespace) -> None:
    selection = _load_selection()
    order = [item["pair_id"] for item in selection["selection"]]
    by_pair = {item["pair_id"]: item for item in selection["selection"]}
    table = pq.read_table(args.dataset, columns=["resume_text", "jd_text"])
    text = {item["pair_id"]: (table["resume_text"][item["parquet_row"]].as_py(), table["jd_text"][item["parquet_row"]].as_py()) for item in selection["selection"]}

    rows: list[dict[str, Any]] = []
    for pair_id in order:
        for model, tag in MODELS.values():
            for mode, mode_label in PROMPTS.items():
                latest: dict[str, dict[str, Any]] = {}
                for record in _read_jsonl(OUT_DIR / f"{tag}-{mode}.jsonl"):  # last record wins, so a retry supersedes an error
                    latest[record["pair_id"]] = record
                record = latest.get(pair_id)
                usage = (record or {}).get("usage") or {}
                output_tokens = usage.get("output_tokens")
                latency = (record or {}).get("latency_seconds")
                ok = bool(record) and record.get("status") == "ok"
                if not record:
                    output = "NOT RUN"
                elif ok:
                    output = record.get("raw_output") or ""
                else:
                    output = f"ERROR: {record.get('error') or record.get('status')}"
                rows.append(
                    {
                        "pair_id": pair_id,
                        "resume": text[pair_id][0],
                        "jd": text[pair_id][1],
                        "verdict": _cell(record.get("fit_label")) if ok else "",
                        "model": model,
                        "prompt": mode_label,
                        "token_count": _cell(usage.get("input_tokens")),
                        "token_per_sec": round(output_tokens / latency, 2) if ok and output_tokens and latency else "",
                        "output_tokens": _cell(output_tokens),
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
    parser.add_argument("stage", choices=["select", "run", "export"])
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-output-tokens", type=int, default=6_144)
    parser.add_argument("--output", default=str(OUT_DIR / "eval_extremes_8.csv"))
    args = parser.parse_args()
    {"select": stage_select, "run": stage_run, "export": stage_export}[args.stage](args)


if __name__ == "__main__":
    main()
