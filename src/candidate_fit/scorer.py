from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Iterable

from candidate_fit.client import LocalLLMClient
from candidate_fit.dataset import PairRecord
from candidate_fit.prompt import ACTIVE_SYSTEM_PROMPT, build_user_prompt, normalize_prediction, parse_json_object


@dataclass(frozen=True, slots=True)
class ScoreOptions:
    max_resume_chars: int = 100_000
    max_jd_chars: int = 100_000
    workers: int = 1
    max_pending_multiplier: int = 2
    max_errors: int = 10


def _content_hash(record: PairRecord) -> str:
    digest = hashlib.blake2b(digest_size=12)
    digest.update(record.resume.encode("utf-8", errors="replace"))
    digest.update(b"\x00")
    digest.update(record.job_description.encode("utf-8", errors="replace"))
    return digest.hexdigest()


def _prompt_hash() -> str:
    return hashlib.blake2b(ACTIVE_SYSTEM_PROMPT.encode("utf-8"), digest_size=12).hexdigest()


def _scoring_fingerprint(client: LocalLLMClient, options: ScoreOptions) -> str:
    reproducibility = {
        "model": client.model,
        "prompt_hash": _prompt_hash(),
        "max_resume_chars": options.max_resume_chars,
        "max_jd_chars": options.max_jd_chars,
        "max_output_tokens": getattr(client, "max_tokens", None),
    }
    encoded = json.dumps(reproducibility, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=12).hexdigest()


def score_one(client: LocalLLMClient, record: PairRecord, options: ScoreOptions) -> dict[str, Any]:
    prompt = build_user_prompt(
        record.resume,
        record.job_description,
        max_resume_chars=options.max_resume_chars,
        max_jd_chars=options.max_jd_chars,
    )
    completion = client.complete(prompt, stage="single_pass")
    prediction = normalize_prediction(parse_json_object(completion.content))
    return {
        "pair_id": record.pair_id,
        "content_hash": _content_hash(record),
        "prompt_hash": _prompt_hash(),
        "scoring_fingerprint": _scoring_fingerprint(client, options),
        "input_resume_chars": len(record.resume),
        "input_jd_chars": len(record.job_description),
        "source_file": record.source_file,
        "row_index": record.row_index,
        "gold_label": record.gold_label,
        **record.metadata,
        **prediction,
        "status": "ok",
        "model": client.model,
        "endpoint": client.base_url,
        "raw_output": completion.content,
        "usage": completion.usage,
        "finish_reason": completion.finish_reason,
        "reasoning_tokens": completion.usage.get("thinking_tokens"),
        "context_window": completion.context_window,
        "dgpu_run_id": completion.dgpu_run_id,
        "timing": completion.timing,
        "resources": completion.resources,
        "client_delivery": completion.client_delivery,
        "latency_seconds": round(completion.latency_seconds, 4),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def completed_pair_ids(output_path: Path, scoring_fingerprint: str | None = None) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fingerprint_matches = scoring_fingerprint is None or row.get("scoring_fingerprint") == scoring_fingerprint
            if row.get("status") == "ok" and fingerprint_matches and isinstance(row.get("pair_id"), str):
                completed.add(row["pair_id"])
    return completed


def score_dataset(
    records: Iterable[PairRecord],
    *,
    client: LocalLLMClient,
    output_path: str | Path,
    options: ScoreOptions,
    resume: bool = True,
) -> dict[str, int]:
    if options.workers < 1:
        raise ValueError("workers must be positive")
    if options.max_errors < 1:
        raise ValueError("max_errors must be positive")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scoring_fingerprint = _scoring_fingerprint(client, options)
    completed = completed_pair_ids(path, scoring_fingerprint) if resume else set()
    counters = {"scored": 0, "errors": 0, "skipped": 0, "aborted": 0, "interrupted": 0}
    write_lock = threading.Lock()

    def run(record: PairRecord) -> dict[str, Any]:
        try:
            return score_one(client, record, options)
        except Exception as exc:  # Preserve progress; errors can be retried on the next run.
            return {
                "pair_id": record.pair_id,
                "content_hash": _content_hash(record),
                "prompt_hash": _prompt_hash(),
                "scoring_fingerprint": scoring_fingerprint,
                "input_resume_chars": len(record.resume),
                "input_jd_chars": len(record.job_description),
                "source_file": record.source_file,
                "row_index": record.row_index,
                "gold_label": record.gold_label,
                **record.metadata,
                "status": "error",
                "model": client.model,
                "endpoint": client.base_url,
                "error": f"{type(exc).__name__}: {exc}",
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }

    def write_result(handle, result: dict[str, Any]) -> None:
        with write_lock:
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
        counters["scored" if result["status"] == "ok" else "errors"] += 1

    max_pending = max(options.workers, options.workers * options.max_pending_multiplier)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        # The normal single-worker path stays on the main thread so Ctrl+C stops
        # immediately after the current HTTP call and every completed row is durable.
        if options.workers == 1:
            try:
                for record in records:
                    if record.pair_id in completed:
                        counters["skipped"] += 1
                        continue
                    write_result(handle, run(record))
                    if counters["errors"] >= options.max_errors:
                        counters["aborted"] = 1
                        break
            except KeyboardInterrupt:
                counters["interrupted"] = 1
            return counters

        with ThreadPoolExecutor(max_workers=options.workers, thread_name_prefix="fit-score") as pool:
            pending: set[Future[dict[str, Any]]] = set()
            for record in records:
                if record.pair_id in completed:
                    counters["skipped"] += 1
                    continue
                pending.add(pool.submit(run, record))
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        write_result(handle, future.result())
                    if counters["errors"] >= options.max_errors:
                        counters["aborted"] = 1
                        break
            for future in wait(pending).done if pending else ():
                write_result(handle, future.result())
    return counters

