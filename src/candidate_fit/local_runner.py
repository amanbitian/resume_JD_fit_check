"""Local runner: one model on one dual_gpu_setup lane, sequential requests, results in the same format as batches.

It follows the dual_gpu_setup client contract (API_CONTRACT.md / CLIENT_API.md):
  - step 0: launch_service.py starts the service if it is not already running
  - step 2: the lane is deployed with the real token budget and the thinking budget
  - step 7: the lane is released afterwards, but only if this run loaded it
Another application's model on the lane is never replaced silently: the run refuses unless told to.

Reliability: a request that fails because the service is briefly unreachable or busy (connection reset, 429, 502-504)
is retried with backoff and does not use up one of the item's attempts. Three items in a row that cannot reach the
service stop the run with a clear message instead of grinding through the whole dataset.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from candidate_fit.batch_runner import RunLock, RunPaths, append_jsonl, current_rows, load_results, save_state
from candidate_fit.client import _LOCAL_OPENER, require_loopback
from candidate_fit.tasks import Item, ValidationError, validate_reply

ORCHESTRATOR_URL = os.getenv("DGPU_ORCHESTRATOR_URL", "http://127.0.0.1:8090")
DGPU_ROOT = Path(os.getenv("DGPU_ROOT", "F:/Projects/Dual_gpu_setup"))
DGPU_CONFIG = DGPU_ROOT / "example.dual_gpu.toml"
CLIENT_CONFIG = Path(os.getenv("CANDIDATE_FIT_CLIENT_CONFIG", ".candidate-fit-client.json"))
SAFETY_TOKENS = 512
RETRY_DELAYS = (2.0, 5.0, 10.0)
RETRY_HTTP = {429, 502, 503, 504}
MAX_UNREACHABLE_IN_A_ROW = 3
MAX_SERVICE_ERRORS_IN_A_ROW = 5  # the service answered, but with an error, this many requests in a row
PROGRESS_WINDOW = 30  # rows averaged for the speed and ETA shown in the progress line
# Characters per token used to estimate how much KV memory a request needs. Qwen measured 4.7 (lowest 4.68) on real JD requests,
# so 4.0 leaves about 15% margin. Used only to decide how many requests may run together.
CHARS_PER_TOKEN = 4.0

# Local models the wizard offers: key -> (lane, catalog model name, label)
LOCAL_MODELS = {
    "qwen3.8-27b": ("r9700", "Qwen3.8-27B-Q4_K_M", "Qwen3.8-27B on the R9700 (32 GB)"),
    "gemma-4-12b": ("9070xt", "gemma-4-12B-it-QAT-Q4_0", "gemma-4-12B on the 9070 XT (16 GB)"),
    "qwen3.5-4b": ("9070xt", "Qwen3.5-4B-Q4_K_M", "Qwen3.5-4B on the 9070 XT (16 GB)"),
    "gpt-oss-20b": ("9070xt", "gpt-oss-20b-GGUF", "gpt-oss-20B on the 9070 XT (16 GB)"),
    "gemma-4-26b-a4b": ("9070xt", "gemma-4-26B-A4B-it-QAT-Q4_0", "gemma-4-26B-A4B (QAT Q4_0) on the 9070 XT (16 GB)"),
    "qwen3.8-27b-iq3": ("9070xt", "Qwen3.8-27B-IQ3_M", "Qwen3.8-27B IQ3_M (3-bit) on the 9070 XT (16 GB)"),
    "qwen3.8-27b-unsloth-q3": ("9070xt", "Qwen3.8-27B-UD-Q3_K_XL", "Qwen3.8-27B UD-Q3_K_XL (Unsloth 3-bit) on the 9070 XT (16 GB)"),
}


class LocalRunError(RuntimeError):
    pass


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {rest // 60:02d}m"
    return f"{rest // 60}m {rest % 60:02d}s"


def api_post(path: str, body: dict[str, Any], timeout: float = 900.0) -> tuple[int, dict[str, Any]]:
    req = request.Request(require_loopback(ORCHESTRATOR_URL) + path, data=json.dumps(body).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with _LOCAL_OPENER.open(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8") or "{}")
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(text)
        except ValueError:
            payload = {"error": text[:500]}
        return exc.code, payload if isinstance(payload, dict) else {"error": text[:500]}


def get_status(attempts: int = 6) -> dict[str, Any]:
    """GET /api/status, riding out the connection resets a just-started service gives its first requests."""
    for attempt in range(attempts):
        try:
            with _LOCAL_OPENER.open(require_loopback(ORCHESTRATOR_URL) + "/api/status", timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError):
            if attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)
    raise LocalRunError("unreachable")


def ensure_service() -> None:
    """Start the dual_gpu_setup service if needed. launch_service.py is idempotent."""
    launcher = DGPU_ROOT / "launch_service.py"
    if not launcher.exists():
        raise LocalRunError(f"dual_gpu_setup not found at {DGPU_ROOT}; set DGPU_ROOT")
    result = subprocess.run([sys.executable, str(launcher), "--config", str(DGPU_CONFIG), "--no-browser"],
                            capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise LocalRunError(f"launch_service.py could not confirm the service is up: {result.stdout} {result.stderr}")


def client_id() -> str:
    if CLIENT_CONFIG.exists():
        return json.loads(CLIENT_CONFIG.read_text(encoding="utf-8"))["client_id"]
    value = os.getenv("CANDIDATE_FIT_CLIENT_ID", "")
    if not value:
        raise LocalRunError("no client registration: run `candidate-fit register` or set CANDIDATE_FIT_CLIENT_ID")
    return value


def context_for(items: list[Item], max_output: int) -> tuple[int, int]:
    """Deploy-time token budget: the largest request (about 3 characters per token, which is conservative)."""
    largest_input = max((math.ceil((len(i.system) + len(i.user)) / 3) for i in items), default=1024)
    input_tokens = max(largest_input, 1024)
    needed = input_tokens + max_output + SAFETY_TOKENS
    context = max(16384, int(math.ceil(needed / 4096) * 4096))
    return input_tokens, context


@dataclass(frozen=True, slots=True)
class LocalSettings:
    model_key: str
    reasoning_budget: int = 0
    replace_other_model: bool = False


class LocalRunner:
    def __init__(self, settings: LocalSettings, *, log: Callable[[str], None] = print) -> None:
        if settings.model_key not in LOCAL_MODELS:
            raise LocalRunError(f"unknown local model {settings.model_key!r}")
        self.settings = settings
        self.lane, self.model, _label = LOCAL_MODELS[settings.model_key]
        self.log = log
        self.loaded_here = False
        self._client_id: str | None = None

    def _lane_server(self) -> dict[str, Any] | None:
        for server in get_status().get("servers", []):
            if server.get("target") == self.lane:
                return server
        return None

    def load(self, items: list[Item], pool: int | None = None) -> None:
        ensure_service()
        max_output = max((i.max_output_tokens for i in items), default=4096)
        input_tokens, context = context_for(items, max_output)
        if pool:  # several requests share the server's KV memory, so it is loaded with the whole pool
            context = max(context, pool)
        current = self._lane_server()
        if current and current.get("alive") and current.get("model") == self.model:
            if current.get("reasoning_budget") != self.settings.reasoning_budget:
                raise LocalRunError(f"{self.lane} already runs {self.model} with thinking budget "
                                    f"{current.get('reasoning_budget')}; release it first to change the budget")
            self.log(f"{self.lane}: {self.model} already loaded; it stays loaded after the run")
            return
        if current and current.get("alive") and not self.settings.replace_other_model:
            raise LocalRunError(f"{self.lane} is serving {current.get('model')}; pass replace_other_model to replace it")
        status, payload = api_post("/api/clients/deploy_lane", {
            "model": self.model, "lane": self.lane, "context_window": context,
            "input_tokens": input_tokens, "max_output_tokens": max_output,
        }, timeout=1800)
        if status != 200:
            raise LocalRunError(f"deploy_lane {self.lane} {self.model}: HTTP {status}: {payload.get('error')}")
        self.loaded_here = True
        self.log(f"{self.lane}: loaded {self.model} (context {context}, budget in {input_tokens}, out {max_output})")
        loaded = self._lane_server() or {}
        if loaded.get("reasoning_budget") != self.settings.reasoning_budget:
            raise LocalRunError("the service did not apply the thinking budget")

    def release(self) -> bool:
        if not self.loaded_here:
            return False
        for _ in range(5):
            status, payload = api_post("/api/unload", {"lane": self.lane, "model": self.model}, timeout=120)
            if status != 409:
                self.loaded_here = False
                return status == 200 and bool(payload.get("released"))
            time.sleep(2)
        return False

    def chat(self, item: Item, user_text: str, *, temperature: float = 0, seed: int = 42) -> dict[str, Any]:
        """One request. Raises OSError if the service cannot be reached. The defaults are the settings of every first attempt."""
        if self._client_id is None:
            self._client_id = client_id()  # read once per run, not once per request
        body = {
            "client_id": self._client_id, "target": self.lane, "task_label": "candidate_fit_local",
            "workload_kind": "evaluation", "connection_type": "batch",
            "messages": [{"role": "system", "content": item.system}, {"role": "user", "content": user_text}],
            "temperature": temperature, "seed": seed, "max_tokens": item.max_output_tokens, "timeout_seconds": 900,
        }
        status, payload = api_post("/api/chat", body, timeout=1000)
        if status != 200:
            return {"status": "failed", "http_status": status, "output": "", "error": f"HTTP {status}: {payload.get('error')}"}
        result = (payload.get("results") or {}).get(self.lane) or {}
        return {"status": result.get("status"), "output": result.get("output_text") or "",
                "finish_reason": result.get("finish_reason"), "usage": result.get("usage") or {},
                "timing": result.get("timing") or {}, "error": result.get("error_text")}

    def chat_with_retries(self, item: Item, user_text: str, **decoding: Any) -> dict[str, Any]:
        """Retry transient failures (service unreachable or busy). The reply says `unreachable` if it never got through."""
        reply: dict[str, Any] = {}
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                reply = self.chat(item, user_text, **decoding)
            except OSError as exc:
                reply = {"status": "failed", "output": "", "unreachable": True, "error": f"service unreachable: {exc}"}
            transient = reply.get("unreachable") or reply.get("http_status") in RETRY_HTTP
            if not transient or attempt == len(RETRY_DELAYS):
                return reply
            time.sleep(RETRY_DELAYS[attempt])
        return reply

    def run(self, task: str, items: list[Item], folder: Path, *, fingerprint: str = "", max_attempts: int = 2,
            progress_every: int = 0, workers: int = 1, kv_pool: int | None = None) -> dict[str, Any]:
        """Run the items. Every attempt is appended to results.jsonl as it finishes, so a run can be stopped at any time
        and started again with the same command: finished items are skipped and the rest continue.

        Ctrl+C asks for a clean stop: the requests in flight finish, the model is released and the results are kept.
        A second Ctrl+C stops at once. With `progress_every` set, a progress line is logged after that many rows.

        By default one request runs at a time. With `workers` > 1, up to that many run together, but only while their token
        needs fit in `kv_pool`, the server's shared KV memory in tokens. A row reserves its estimated input, its output limit
        and a safety margin, and starts only when that still fits next to the rows already running. So two large rows never
        run together if they do not fit, and several small rows can. This needs a server with that many slots over one
        shared (unified) KV pool, which is set in the service's configuration, not here.
        """
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if workers > 1 and not kv_pool:
            raise LocalRunError("running several requests at once needs kv_pool, the size of the server's shared KV memory in tokens")

        def reserve(item: Item) -> int:
            """Tokens a request can occupy in the server's KV memory: its input, everything it may write, and a margin."""
            return math.ceil((len(item.system) + len(item.user)) / CHARS_PER_TOKEN) + item.max_output_tokens + SAFETY_TOKENS

        paths = RunPaths(folder)
        with RunLock(folder):
            state = {"task": task, "model": self.model, "provider": "local", "lane": self.lane}
            if paths.state.exists():
                previous = json.loads(paths.state.read_text(encoding="utf-8"))
                if previous.get("model") != self.model or previous.get("task") != task:
                    raise LocalRunError(f"{folder} holds a different run; use a new folder")
            save_state(paths.state, state)
            latest, attempts = current_rows(load_results(paths.results), fingerprint)
            finished = {item_id for item_id, row in latest.items() if row.get("ok")}
            pool_tokens = 0
            if workers > 1:  # refuse before loading anything if a single row could never fit
                largest = max((reserve(item) for item in items), default=0)
                if largest > kv_pool:
                    raise LocalRunError(f"the largest row needs {largest:,} tokens of KV memory (input + output limit + margin) "
                                        f"but the pool is {kv_pool:,}. Lower the output limit or raise the pool")
            self.load(items, pool=kv_pool if workers > 1 else None)
            if workers > 1:  # use what the server really has, which may be less than was asked for
                pool_tokens = min(kv_pool, (self._lane_server() or {}).get("context_window") or kv_pool)
                if pool_tokens < kv_pool:
                    self.log(f"the server has a {pool_tokens:,}-token pool, less than the {kv_pool:,} asked for; using {pool_tokens:,}")
                if largest > pool_tokens:
                    raise LocalRunError(f"the largest row needs {largest:,} tokens but the server's pool is {pool_tokens:,}")
            counts = {"ok": 0, "failed": 0, "skipped": 0}
            lock = threading.RLock()  # guards the counters, the results file and the in-flight bookkeeping
            wake = threading.Condition(lock)
            unreachable_in_a_row = 0
            service_errors_in_a_row = 0
            # Progress: how many rows are settled (usable, or out of attempts), and when recent rows finished.
            total = len(items)
            usable = sum(1 for item in items if item.item_id in finished)
            gave_up = sum(1 for item in items if item.item_id not in finished and attempts.get(item.item_id, 0) >= max_attempts)
            recent: deque[float] = deque([time.monotonic()], maxlen=PROGRESS_WINDOW + 1)  # row finish times, from the start
            processed = 0
            inflight = 0
            reserved = 0
            stopped = False
            stop_asked = False

            def progress_line() -> str:
                settled = usable + gave_up
                rows = len(recent) - 1
                average = (recent[-1] - recent[0]) / rows  # seconds per finished row, however many ran together
                left = total - settled
                finish = time.strftime("%a %d %b %H:%M", time.localtime(time.time() + left * average))
                running = f" | {inflight} running" if workers > 1 else ""
                return (f"progress {settled:,}/{total:,} ({settled / total:.2%}) | usable {usable:,}, failed {gave_up:,} | "
                        f"last {rows} rows {average:.0f} s/row{running} | {left:,} left, ETA {_duration(left * average)} (about {finish})")

            def ask_to_stop(signum: int, _frame: Any) -> None:
                nonlocal stop_asked
                if stop_asked:  # a second press stops at once
                    signal.signal(signum, signal.default_int_handler)
                    raise KeyboardInterrupt
                stop_asked = True
                self.log("stop requested: the requests in flight will finish, then the run stops and the model is released "
                         "(press Ctrl+C again to stop at once)")

            previous_handlers: dict[int, Any] = {}
            for name in ("SIGINT", "SIGBREAK"):  # SIGBREAK (Ctrl+Break) exists on Windows only
                if hasattr(signal, name):
                    try:
                        previous_handlers[getattr(signal, name)] = signal.signal(getattr(signal, name), ask_to_stop)
                    except ValueError:  # not the main thread: no clean-stop handler, the default Ctrl+C applies
                        break
            if progress_every:
                self.log(f"resuming: {usable + gave_up:,} of {total:,} already settled ({usable:,} usable, {gave_up:,} failed); "
                         f"{total - usable - gave_up:,} left. Ctrl+C stops cleanly; rerun the same command to resume")
            if workers > 1:
                needs = sorted(reserve(item) for item in items if item.item_id not in finished) or [0]
                smallest, typical, biggest = needs[0], needs[len(needs) // 2], needs[-1]
                at_once = lambda need: min(workers, pool_tokens // need) if need else workers  # noqa: E731
                self.log(f"parallel: up to {workers} requests at once in a {pool_tokens:,}-token pool. A row reserves its input + "
                         f"output limit + {SAFETY_TOKENS}: smallest {smallest:,}, typical {typical:,}, largest {biggest:,} tokens, "
                         f"so {at_once(smallest)} / {at_once(typical)} / {at_once(biggest)} rows at once")

            def run_item(item: Item) -> None:
                """All attempts for one row. Safe to call from several threads, one row per thread."""
                nonlocal unreachable_in_a_row, service_errors_in_a_row, usable, gave_up, processed
                ran_a_request = False
                # On a resume, an item that already failed once is retried with the reason it failed, as it would have been.
                prior = latest.get(item.item_id)
                last_error: str | None = prior.get("error") if prior and not prior.get("ok") and attempts.get(item.item_id, 0) else None
                while attempts.get(item.item_id, 0) < max_attempts:
                    if stop_asked:
                        break
                    attempt = attempts.get(item.item_id, 0) + 1
                    user_text = item.user
                    if attempt > 1 and last_error:
                        user_text += (f"\n\nYour previous reply was rejected: {last_error}. "
                                      "Return one complete JSON object only.")
                    # Only an item that asks for it (a JD prompt version with a retry temperature) changes the decoding
                    # on a retry: the same greedy request would give the same reply again. All others retry as before.
                    decoding: dict[str, Any] = {}
                    if attempt > 1 and item.retry_temperature is not None:
                        decoding = {"temperature": item.retry_temperature, "seed": 42 + attempt - 1}
                    started = time.perf_counter()
                    reply = self.chat_with_retries(item, user_text, **decoding)
                    seconds = time.perf_counter() - started
                    if reply.get("unreachable"):
                        # Not the model's fault: leave the item for the next run and do not use up an attempt.
                        with lock:
                            unreachable_in_a_row += 1
                            counts["failed"] += 1
                            too_many = unreachable_in_a_row >= MAX_UNREACHABLE_IN_A_ROW
                        if too_many:
                            raise LocalRunError(f"the dual GPU service was unreachable for {MAX_UNREACHABLE_IN_A_ROW} items "
                                                "in a row; rerun the same command to resume")
                        break
                    with lock:
                        unreachable_in_a_row = 0
                    ran_a_request = True
                    attempts[item.item_id] = attempt
                    usage, timing = reply.get("usage") or {}, reply.get("timing") or {}
                    row = {"item_id": item.item_id, "model": self.model, "attempt": attempt, "fp": fingerprint,
                           "finish_reason": reply.get("finish_reason"), "ok": False, "json_valid": False,
                           "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                           "thinking_tokens": usage.get("thinking_tokens"),
                           "tokens_per_sec": timing.get("tokens_per_second"),
                           "prefill_tokens_per_sec": timing.get("prefill_tokens_per_second"),
                           "latency_seconds": round(seconds, 3), "output": reply.get("output", ""),
                           "error": reply.get("error")}
                    if decoding:
                        row["retry_decoding"] = decoding
                    last_error = None
                    if reply.get("status") != "completed":
                        last_error = row["error"] = row["error"] or f"status {reply.get('status')}"
                    elif reply.get("finish_reason") == "length":
                        last_error = row["error"] = "output token limit reached"
                    else:
                        try:
                            row.update(validate_reply(task, item, reply["output"]))
                            row["ok"], row["error"] = True, None
                        except ValidationError as exc:
                            last_error = row["error"] = f"invalid reply: {exc}"
                    with lock:
                        append_jsonl(paths.results, row)
                        # A service that keeps answering with errors would use up two attempts on every remaining item in
                        # seconds, and a resume would not retry them. Stop after a few in a row instead.
                        if reply.get("status") != "completed":
                            service_errors_in_a_row += 1
                            broken = service_errors_in_a_row >= MAX_SERVICE_ERRORS_IN_A_ROW
                            broken_count = service_errors_in_a_row
                        else:
                            service_errors_in_a_row = 0
                            broken = False
                        if row["ok"]:
                            finished.add(item.item_id)
                            counts["ok"] += 1
                    if broken:
                        raise LocalRunError(
                            f"the model service returned an error for {broken_count} requests in a row "
                            f"(last: {str(row['error'])[:150]}). Stopped so the rest of the items are not marked failed. "
                            "Fix the service, then rerun the same command to resume")
                    if row["ok"]:
                        break
                else:
                    with lock:
                        counts["failed"] += 1
                with lock:
                    if ran_a_request and (item.item_id in finished or attempts.get(item.item_id, 0) >= max_attempts):
                        usable += item.item_id in finished
                        gave_up += item.item_id not in finished
                        processed += 1
                        recent.append(time.monotonic())
                        if progress_every and processed % progress_every == 0:
                            self.log(progress_line())

            def run_in_parallel() -> bool:
                """Start rows in dataset order, each once it fits next to the rows already running. True if a stop was asked."""
                nonlocal inflight, reserved
                failures: list[BaseException] = []

                def work(item: Item, need: int) -> None:
                    nonlocal inflight, reserved
                    try:
                        run_item(item)
                    except BaseException as exc:  # noqa: BLE001 - handed to the main thread, which stops and re-raises it
                        with wake:
                            failures.append(exc)
                    finally:
                        with wake:
                            inflight -= 1
                            reserved -= need
                            wake.notify_all()

                executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="row")
                try:
                    for item in items:
                        if item.item_id in finished:
                            counts["skipped"] += 1
                            continue
                        need = reserve(item)
                        with wake:
                            while not failures and not stop_asked and (inflight >= workers or reserved + need > pool_tokens):
                                wake.wait(timeout=0.5)  # wakes at least twice a second, so Ctrl+C is answered quickly
                            if failures or stop_asked:
                                break
                            inflight += 1
                            reserved += need
                        executor.submit(work, item, need)
                    executor.shutdown(wait=True)  # rows already started always finish
                except BaseException:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                if failures:
                    raise failures[0]
                return stop_asked

            try:
                if workers > 1:
                    stopped = run_in_parallel()
                else:
                    for item in items:
                        if stop_asked:
                            stopped = True
                            break
                        if item.item_id in finished:
                            counts["skipped"] += 1
                            continue
                        run_item(item)
                if progress_every and processed % progress_every:
                    self.log(progress_line())  # the last, shorter group of rows
                if stopped:
                    self.log(f"stopped on request after {processed:,} rows in this session; rerun the same command to resume")
            finally:
                for number, handler in previous_handlers.items():
                    signal.signal(number, handler)
                released = self.release()
                self.log(f"{self.lane}: released {self.model}: {released}" if released else f"{self.lane}: left loaded")
            return {"folder": str(folder), **counts, **({"stopped": True} if stopped else {})}
