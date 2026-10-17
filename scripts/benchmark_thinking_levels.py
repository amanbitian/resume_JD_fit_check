"""Run a controlled local-model thinking-budget benchmark on representative resume/JD pairs.

The benchmark uses three shortest, three median-length, and three longest pairs from one parquet
shard. It deploys the same thinking budget to Qwen and GPT-OSS, scores the same nine labelled
pairs, stores each condition in an independent resumable JSONL, and restores reasoning-off.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import sys
from typing import Any
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from candidate_fit.client import DualGPUClient
from candidate_fit.dataset import PairRecord, iter_pairs
from candidate_fit.dual_gpu import load_client_config
from candidate_fit.metrics import evaluate_jsonl
from candidate_fit.optimized import score_dataset_optimized


QWEN_MODEL = "lmstudio:lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
QWEN_CLIENT_MODEL = "Qwen3.8-27B-Q4_K_M"
GPT_MODEL = "gpt-oss-20b-GGUF"
LEVELS = {"off": 0, "low": 512, "medium": 2048, "high": 4096}


def _total_chars(record: PairRecord) -> int:
    return len(record.resume) + len(record.job_description)


def select_records(dataset: str) -> tuple[list[PairRecord], list[dict[str, Any]]]:
    """Select three shortest, median-nearest, and three longest distinct pairs."""
    lengths: list[tuple[int, str, int]] = []
    for record in iter_pairs(dataset):
        lengths.append((_total_chars(record), record.pair_id, record.row_index))
    if len(lengths) < 9:
        raise ValueError("Dataset must have at least nine pairs")

    ordered = sorted(lengths)
    chosen: dict[str, str] = {}
    for _, pair_id, _ in ordered[:3]:
        chosen[pair_id] = "small"
    for _, pair_id, _ in reversed(ordered[-3:]):
        chosen[pair_id] = "long"

    median = statistics.median(length for length, _, _ in lengths)
    for _, pair_id, _ in sorted(lengths, key=lambda item: (abs(item[0] - median), item[2])):
        if pair_id not in chosen:
            chosen[pair_id] = "medium"
            if sum(group == "medium" for group in chosen.values()) == 3:
                break

    records: list[PairRecord] = []
    for record in iter_pairs(dataset):
        if record.pair_id in chosen:
            records.append(record)
    records.sort(key=lambda record: ({"small": 0, "medium": 1, "long": 2}[chosen[record.pair_id]], record.row_index))
    selection = [
        {
            "group": chosen[record.pair_id],
            "pair_id": record.pair_id,
            "row_index": record.row_index,
            "resume_chars": len(record.resume),
            "jd_chars": len(record.job_description),
            "total_chars": _total_chars(record),
            "gold_label": record.gold_label,
        }
        for record in records
    ]
    return records, selection


def deploy(orchestrator_url: str, budget: int, max_output_tokens: int) -> dict[str, Any]:
    """Use the documented deployment endpoint so both lanes have matching thinking policy."""
    body = {
        "mode": "parallel_models",
        "models": [
            {
                "model": QWEN_MODEL,
                "lane": "r9700",
                "context_window": 16384,
                "input_tokens": 5000,
                "max_output_tokens": max_output_tokens,
                "reasoning_budget": budget,
            },
            {
                "model": GPT_MODEL,
                "lane": "9070xt",
                "context_window": 16384,
                "input_tokens": 5000,
                "max_output_tokens": max_output_tokens,
                "reasoning_budget": budget,
            },
        ],
    }
    request = Request(
        f"{orchestrator_url.rstrip('/')}/api/deploy",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=900) as response:  # nosec B310 - local user-selected endpoint
        return json.loads(response.read().decode("utf-8"))


def summarize(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [row for row in rows if row.get("status") == "ok"]
    latency = [float(row["latency_seconds"]) for row in valid]
    outputs = [int((row.get("usage") or {}).get("output_tokens") or 0) for row in valid]
    retries = [int(row.get("pair_attempts") or 0) for row in valid]
    return {
        "rows": len(rows),
        "valid_rows": len(valid),
        "errors": len(rows) - len(valid),
        "mean_pair_latency_seconds": round(statistics.mean(latency), 3) if latency else None,
        "mean_output_tokens": round(statistics.mean(outputs), 1) if outputs else None,
        "total_pair_attempts": sum(retries),
        "quality": evaluate_jsonl(path),
    }


def score_model(
    *,
    model: str,
    lane: str,
    level: str,
    records: list[PairRecord],
    output_dir: Path,
    cache_dir: Path,
    client_id: str,
    orchestrator_url: str,
    max_output_tokens: int,
) -> tuple[str, dict[str, Any]]:
    stem = f"{model.lower()}-{level}"
    output = output_dir / f"{stem}.jsonl"
    cache = cache_dir / f"{model.lower()}-analysis-cache.jsonl"
    client = DualGPUClient(
        orchestrator_url=orchestrator_url,
        lane=lane,
        model=model,
        client_id=client_id,
        task_label=f"thinking-level-{level}-{model}",
        timeout=900,
        retries=1,
        max_tokens=max_output_tokens,
    )
    result = score_dataset_optimized(
        records,
        client=client,
        output_path=output,
        cache_path=cache,
        resume=True,
        # Benchmark every selected pair even if one long structured response is rejected.
        # Error rows remain resumable and are reported separately from valid-score accuracy.
        max_errors=10,
    )
    return model, {"output": str(output), "run": result, **summarize(output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output-dir", default="outputs/thinking-level-benchmark")
    parser.add_argument("--client-config", default=".candidate-fit-client.json")
    parser.add_argument("--orchestrator-url", default="http://127.0.0.1:8090")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "analysis-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = load_client_config(args.client_config)
    client_id = str(config.get("client_id") or "")
    if not client_id:
        raise SystemExit("No registered client ID; run candidate-fit register first.")
    records, selection = select_records(args.dataset)
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {"selection": selection, "conditions": {}, "levels": LEVELS}

    try:
        for level, budget in LEVELS.items():
            print(f"Deploying thinking level={level} budget={budget}", flush=True)
            deployment = deploy(args.orchestrator_url, budget, args.max_output_tokens)
            summary["conditions"][level] = {"deployment_id": deployment.get("deployment_id"), "models": {}}
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        score_model,
                        model=QWEN_CLIENT_MODEL,
                        lane="r9700",
                        level=level,
                        records=records,
                        output_dir=output_dir,
                        cache_dir=cache_dir,
                        client_id=client_id,
                        orchestrator_url=args.orchestrator_url,
                        max_output_tokens=args.max_output_tokens,
                    ),
                    pool.submit(
                        score_model,
                        model=GPT_MODEL,
                        lane="9070xt",
                        level=level,
                        records=records,
                        output_dir=output_dir,
                        cache_dir=cache_dir,
                        client_id=client_id,
                        orchestrator_url=args.orchestrator_url,
                        max_output_tokens=args.max_output_tokens,
                    ),
                ]
                for future in as_completed(futures):
                    model, result = future.result()
                    summary["conditions"][level]["models"][model] = result
                    print(f"Completed {level} {model}: {result['valid_rows']}/{result['rows']} valid", flush=True)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    finally:
        print("Restoring reasoning-off production profile", flush=True)
        deploy(args.orchestrator_url, 0, 6144)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
