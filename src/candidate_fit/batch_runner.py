"""Provider-independent batch runner for cloud models.

Flow for one task and one provider model:
  1. Results are keyed by item id and tagged with a fingerprint of everything that changes what a reply means
     (task, prompts, validator, output cap, reference date). Items with a valid reply under the current
     fingerprint are skipped; results made under another fingerprint are kept in the file but never reused.
  2. Pending items are grouped by attempt number and split into batches within the provider's limits. Requests
     are built one batch at a time, so memory stays flat for very large runs.
  3. Before each submission an intent is saved. If the process dies between the provider creating the batch and
     the batch id being saved, the next start asks the provider for it (find_submitted) instead of paying twice.
  4. Each batch is polled until it ends. Every reply is validated by the task (strict JSON, schema fields, and for
     pairs the calculator). Valid replies are final; invalid or failed ones are recorded and retried in a later
     attempt, up to max_attempts.

Files written to the run folder:
  .lock              held while a run is active, so two processes cannot work in the same folder
  batch_state.json   submitted batch ids and unconfirmed submission intents
  results.jsonl      one line per attempt per item; the source of truth
  results.csv        exported by export_csv()
"""
from __future__ import annotations

import csv
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from candidate_fit.cloud_batch import BatchApiError
from candidate_fit.tasks import ID_COLUMNS, Item, ValidationError, item_texts, result_columns, validate_reply

Log = Callable[[str], None]


class RunLockedError(RuntimeError):
    pass


class RunLock:
    """An OS-level lock on <folder>/.lock. It is released automatically if the process crashes."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._handle = None

    def __enter__(self) -> "RunLock":
        self.folder.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.folder / ".lock", "a+b")
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._handle.close()
            raise RunLockedError(f"another process is already running in {self.folder}") from exc
        return self

    def __exit__(self, *exc_info: Any) -> None:
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()


@dataclass(frozen=True, slots=True)
class RunPaths:
    folder: Path

    @property
    def state(self) -> Path:
        return self.folder / "batch_state.json"

    @property
    def results(self) -> Path:
        return self.folder / "results.jsonl"

    @property
    def csv(self) -> Path:
        return self.folder / "results.csv"


def load_state(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)  # atomic, so a crash never leaves a half-written state


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line after a crash; the item is simply retried
    return rows


def current_rows(rows: Iterable[dict[str, Any]], fingerprint: str) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Latest row and attempt count per item id, counting only rows made under the current fingerprint."""
    latest: dict[str, dict[str, Any]] = {}
    attempts: dict[str, int] = {}
    for row in rows:
        if row.get("fp", "") != fingerprint or "item_id" not in row:
            continue
        latest[row["item_id"]] = row
        attempts[row["item_id"]] = max(attempts.get(row["item_id"], 0), row.get("attempt", 0))
    return latest, attempts


def split_requests(requests: Iterable[dict[str, Any]], client: Any) -> Iterator[list[dict[str, Any]]]:
    """Group requests into batches that respect the provider's request-count and byte limits."""
    current: list[dict[str, Any]] = []
    size = 0
    for request in requests:
        encoded = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        if current and (len(current) >= client.max_requests or size + encoded > client.max_bytes):
            yield current
            current, size = [], 0
        current.append(request)
        size += encoded
    if current:
        yield current


def chunk(requests: list[dict[str, Any]], client: Any) -> list[list[dict[str, Any]]]:
    return list(split_requests(requests, client))


def iter_chunks(group: list[Item], attempt: int, client: Any) -> Iterator[list[dict[str, Any]]]:
    """Build requests lazily, one batch at a time."""
    requests = ({"custom_id": f"a{attempt}-{item.index}", "item_id": item.item_id, "system": item.system,
                 "user": item.user, "max_output_tokens": item.max_output_tokens} for item in group)
    return split_requests(requests, client)


def run_batches(
    task: str,
    items: list[Item],
    client: Any,
    folder: Path,
    *,
    fingerprint: str = "",
    max_attempts: int = 2,
    poll_seconds: float = 60.0,
    max_wait_seconds: float = 24 * 3600.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    log: Log = print,
) -> dict[str, Any]:
    paths = RunPaths(folder)
    with RunLock(folder):
        state = load_state(paths.state, {"task": task, "model": client.model, "provider": client.provider})
        if state.get("model") != client.model or state.get("task") != task:
            raise RuntimeError(f"{folder} holds a different run ({state.get('task')}, {state.get('model')}); use a new folder")
        state.setdefault("batches", [])
        state.setdefault("intents", [])
        by_id = {item.item_id: item for item in items}
        latest, attempts = current_rows(load_results(paths.results), fingerprint)

        def done(item_id: str) -> bool:
            return bool(latest.get(item_id, {}).get("ok"))

        def collect(batch: dict[str, Any]) -> None:
            """Validate and store every reply of one finished batch; items with no reply become errors."""
            stale = batch.get("fp", "") != fingerprint
            seen: set[str] = set()
            for reply in client.results(batch["batch_id"]):
                item_id = batch["custom_ids"].get(reply.get("custom_id"))
                if item_id is None or item_id in seen:
                    continue
                seen.add(item_id)
                row = {"item_id": item_id, "model": client.model, "attempt": batch["attempt"], "fp": batch.get("fp", ""),
                       "batch_id": batch["batch_id"], "finish_reason": reply.get("finish_reason"),
                       "input_tokens": reply.get("input_tokens"), "output_tokens": reply.get("output_tokens"),
                       "thinking_tokens": reply.get("thinking_tokens"), "output": reply.get("text", ""),
                       "error": reply.get("error"), "ok": False, "json_valid": False}
                if stale:
                    row["error"], row["stale"] = "stale: prompt or settings changed after this batch was submitted", True
                elif reply.get("ok"):
                    try:
                        row.update(validate_reply(task, by_id[item_id], reply["text"]))
                        row["ok"], row["error"] = True, None
                    except ValidationError as exc:
                        row["error"] = f"invalid reply: {exc}"
                append_jsonl(paths.results, row)
                if not stale:
                    latest[item_id] = row
                    attempts[item_id] = max(attempts.get(item_id, 0), batch["attempt"])
            for item_id in batch["custom_ids"].values():
                if item_id not in seen:
                    row = {"item_id": item_id, "model": client.model, "attempt": batch["attempt"], "fp": batch.get("fp", ""),
                           "batch_id": batch["batch_id"], "ok": False, "json_valid": False, "output": "",
                           "error": "no reply returned for this item"}
                    append_jsonl(paths.results, row)
                    if not stale:
                        latest[item_id] = row
                        attempts[item_id] = max(attempts.get(item_id, 0), batch["attempt"])
            batch["done"] = True
            save_state(paths.state, state)

        def wait(batch: dict[str, Any]) -> None:
            started = clock()
            while True:
                status = client.status(batch["batch_id"])
                if status.done:
                    log(f"batch {batch['batch_id']} {status.state}: {status.counts}")
                    collect(batch)
                    return
                if clock() - started > max_wait_seconds:
                    raise RuntimeError(f"batch {batch['batch_id']} still {status.state} after {max_wait_seconds:.0f}s; "
                                       "rerun the same command later to resume polling it")
                sleep(poll_seconds)

        # 1. Submissions that were started but whose batch id was never saved: ask the provider before paying again.
        known = {b["batch_id"] for b in state["batches"]}
        for intent in list(state["intents"]):
            found = client.find_submitted(intent["intent_id"], intent["created_epoch"], intent["expected"], known)
            if len(found) > 1:
                raise RuntimeError(f"an interrupted submission could match {len(found)} batches at the provider "
                                   f"({', '.join(found)}); check the provider console and remove the intent from {paths.state}")
            if found:
                log(f"recovered interrupted submission as batch {found[0]}")
                state["batches"].append({"batch_id": found[0], "attempt": intent["attempt"], "fp": intent["fp"],
                                         "custom_ids": intent["custom_ids"], "submitted_at": intent["created_epoch"],
                                         "done": False})
                known.add(found[0])
            else:
                log("an interrupted submission left no batch at the provider; it will be submitted again")
            state["intents"].remove(intent)
            save_state(paths.state, state)

        # 2. Unfinished batches from an earlier run: poll them, never resubmit them.
        for batch in state["batches"]:
            if batch.get("done"):
                continue
            if batch.get("fp", "") == fingerprint:
                outside = [i for i in batch["custom_ids"].values() if i not in by_id]
                if outside:
                    raise RuntimeError(f"unfinished batch {batch['batch_id']} covers {len(outside)} items outside this run's "
                                       "sample; rerun with the same --limit/--seed (or collect it with the original settings)")
            log(f"resuming batch {batch['batch_id']} (attempt {batch['attempt']})")
            wait(batch)

        # 3. New attempts for every item without a valid reply, until max_attempts is reached.
        while True:
            pending = [item for item in items if not done(item.item_id) and attempts.get(item.item_id, 0) < max_attempts]
            if not pending:
                break
            groups: dict[int, list[Item]] = {}
            for item in pending:
                groups.setdefault(attempts.get(item.item_id, 0) + 1, []).append(item)
            for attempt in sorted(groups):
                submitted = []
                for requests in iter_chunks(groups[attempt], attempt, client):
                    intent = {"intent_id": uuid.uuid4().hex, "attempt": attempt, "fp": fingerprint,
                              "created_epoch": time.time(), "expected": len(requests),
                              "custom_ids": {r["custom_id"]: r["item_id"] for r in requests}}
                    state["intents"].append(intent)
                    save_state(paths.state, state)
                    try:
                        batch_id = client.submit(requests, intent["intent_id"])
                    except BatchApiError as exc:
                        if exc.status is not None and exc.status < 500:  # the provider rejected it: nothing was created
                            state["intents"].remove(intent)
                            save_state(paths.state, state)
                        raise
                    state["intents"].remove(intent)
                    batch = {"batch_id": batch_id, "attempt": attempt, "fp": fingerprint, "custom_ids": intent["custom_ids"],
                             "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "done": False}
                    state["batches"].append(batch)
                    save_state(paths.state, state)
                    log(f"submitted {len(requests)} requests as batch {batch_id} (attempt {attempt})")
                    submitted.append(batch)
                for batch in submitted:
                    wait(batch)

        ok = sum(1 for item in items if done(item.item_id))
        failed = {item.item_id: (latest.get(item.item_id) or {}).get("error") for item in items if not done(item.item_id)}
        return {"items": len(items), "ok": ok, "failed": len(failed), "failed_examples": list(failed.items())[:5],
                "folder": str(folder)}


def export_csv(task: str, items: list[Item], folder: Path, model: str, fingerprint: str = "") -> Path:
    paths = RunPaths(folder)
    folder.mkdir(parents=True, exist_ok=True)
    latest, _ = current_rows(load_results(paths.results), fingerprint)
    columns = result_columns(task)
    id_column = ID_COLUMNS[task]  # results.jsonl keeps the generic key `item_id`; the CSV uses the dataset's name
    blank = lambda value: "" if value is None else value  # noqa: E731
    try:
        handle = paths.csv.open("w", encoding="utf-8-sig", newline="")
    except PermissionError as exc:  # on Windows, a file open in Excel cannot be overwritten
        raise RuntimeError(f"{paths.csv} is open in another program (for example Excel). "
                           "Close it and run the export again") from exc
    with handle:
        writer = csv.DictWriter(handle, fieldnames=columns, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            row = latest.get(item.item_id)
            texts = item_texts(task, item)  # the complete source text sits beside every result
            if row is None:
                writer.writerow({id_column: item.item_id, **texts, "model": model, "status": "not run"})
                continue
            writer.writerow({
                **row,
                id_column: item.item_id,
                **texts,
                "status": "ok" if row.get("ok") else "error",
                **{key: blank(row.get(key)) for key in ("input_tokens", "output_tokens", "thinking_tokens",
                                                        "tokens_per_sec", "prefill_tokens_per_sec", "latency_seconds")},
            })
    return paths.csv
