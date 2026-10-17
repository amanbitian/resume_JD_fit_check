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
import subprocess
import sys
import time
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

# Local models the wizard offers: key -> (lane, catalog model name, label)
LOCAL_MODELS = {
    "qwen3.8-27b": ("r9700", "Qwen3.8-27B-Q4_K_M", "Qwen3.8-27B on the R9700 (32 GB)"),
    "gemma-4-12b": ("9070xt", "gemma-4-12B-it-QAT-Q4_0", "gemma-4-12B on the 9070 XT (16 GB)"),
    "qwen3.5-4b": ("9070xt", "Qwen3.5-4B-Q4_K_M", "Qwen3.5-4B on the 9070 XT (16 GB)"),
    "gpt-oss-20b": ("9070xt", "gpt-oss-20b-GGUF", "gpt-oss-20B on the 9070 XT (16 GB)"),
}


class LocalRunError(RuntimeError):
    pass


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

    def load(self, items: list[Item]) -> None:
        ensure_service()
        max_output = max((i.max_output_tokens for i in items), default=4096)
        input_tokens, context = context_for(items, max_output)
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

    def chat(self, item: Item, user_text: str) -> dict[str, Any]:
        """One request. Raises OSError if the service cannot be reached."""
        if self._client_id is None:
            self._client_id = client_id()  # read once per run, not once per request
        body = {
            "client_id": self._client_id, "target": self.lane, "task_label": "candidate_fit_local",
            "workload_kind": "evaluation", "connection_type": "batch",
            "messages": [{"role": "system", "content": item.system}, {"role": "user", "content": user_text}],
            "temperature": 0, "seed": 42, "max_tokens": item.max_output_tokens, "timeout_seconds": 900,
        }
        status, payload = api_post("/api/chat", body, timeout=1000)
        if status != 200:
            return {"status": "failed", "http_status": status, "output": "", "error": f"HTTP {status}: {payload.get('error')}"}
        result = (payload.get("results") or {}).get(self.lane) or {}
        return {"status": result.get("status"), "output": result.get("output_text") or "",
                "finish_reason": result.get("finish_reason"), "usage": result.get("usage") or {},
                "timing": result.get("timing") or {}, "error": result.get("error_text")}

    def chat_with_retries(self, item: Item, user_text: str) -> dict[str, Any]:
        """Retry transient failures (service unreachable or busy). The reply says `unreachable` if it never got through."""
        reply: dict[str, Any] = {}
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                reply = self.chat(item, user_text)
            except OSError as exc:
                reply = {"status": "failed", "output": "", "unreachable": True, "error": f"service unreachable: {exc}"}
            transient = reply.get("unreachable") or reply.get("http_status") in RETRY_HTTP
            if not transient or attempt == len(RETRY_DELAYS):
                return reply
            time.sleep(RETRY_DELAYS[attempt])
        return reply

    def run(self, task: str, items: list[Item], folder: Path, *, fingerprint: str = "", max_attempts: int = 2) -> dict[str, Any]:
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
            self.load(items)
            counts = {"ok": 0, "failed": 0, "skipped": 0}
            unreachable_in_a_row = 0
            try:
                for item in items:
                    if item.item_id in finished:
                        counts["skipped"] += 1
                        continue
                    last_error: str | None = None
                    while attempts.get(item.item_id, 0) < max_attempts:
                        attempt = attempts.get(item.item_id, 0) + 1
                        user_text = item.user
                        if attempt > 1 and last_error:
                            user_text += (f"\n\nYour previous reply was rejected: {last_error}. "
                                          "Return one complete JSON object only.")
                        started = time.perf_counter()
                        reply = self.chat_with_retries(item, user_text)
                        seconds = time.perf_counter() - started
                        if reply.get("unreachable"):
                            # Not the model's fault: leave the item for the next run and do not use up an attempt.
                            unreachable_in_a_row += 1
                            counts["failed"] += 1
                            if unreachable_in_a_row >= MAX_UNREACHABLE_IN_A_ROW:
                                raise LocalRunError(f"the dual GPU service was unreachable for {MAX_UNREACHABLE_IN_A_ROW} items "
                                                    "in a row; rerun the same command to resume")
                            break
                        unreachable_in_a_row = 0
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
                        append_jsonl(paths.results, row)
                        if row["ok"]:
                            finished.add(item.item_id)
                            counts["ok"] += 1
                            break
                    else:
                        counts["failed"] += 1
            finally:
                released = self.release()
                self.log(f"{self.lane}: released {self.model}: {released}" if released else f"{self.lane}: left loaded")
            return {"folder": str(folder), **counts}
