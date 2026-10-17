"""Cloud batch clients: Anthropic Message Batches and OpenAI Batch. Standard library only.

Every client takes requests shaped as {"custom_id", "system", "user", "max_output_tokens"} and returns
normalised results shaped as {"custom_id", "ok", "text", "finish_reason", "input_tokens", "output_tokens",
"thinking_tokens", "error"}. Batch processing is the only way cloud models are used by this package.

API details follow the providers' public batch documentation. They are verified only by the unit tests here,
which use a fake transport. Run one small live batch (a few items) before a large run.

Reliability:
  - GET requests are retried with exponential backoff on network errors and on 408/425/429/5xx/529.
  - POST requests are retried only on those status codes, never on a network error: a lost reply to a POST may
    mean the batch was created, so a repeat could be billed twice. Instead the caller records the intent first and
    uses find_submitted() on the next start to adopt a batch that was created after all.

API keys come from environment variables and are never written to disk or to output files.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable

Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, bytes]]

# USD per million tokens at standard (synchronous) rates. Batch rates are half of these for both providers.
# Sources: see docs/cloud_pricing.md. Verify against the provider pricing pages before a billed run.
PRICES_PER_MILLION = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "source": "Haiku 5.5 secondary price summaries, 2026"},
    "gpt-5.6-luna": {"input": 0.20, "output": 1.20, "source": "AnotherWrapper GPT-5.6 Luna page, Aug 2026"},
}
BATCH_DISCOUNT = 0.5
ANTHROPIC_MAX_REQUESTS = 100_000
ANTHROPIC_MAX_BYTES = 256 * 1024 * 1024
OPENAI_MAX_REQUESTS = 50_000
OPENAI_MAX_BYTES = 200 * 1024 * 1024
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 529}
REASONING_EFFORTS = ("minimal", "low", "medium", "high")


class BatchApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def default_transport(timeout: float = 120.0) -> Transport:
    def send(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    return send


def retrying(send: Transport, *, sleep: Callable[[float], None] = time.sleep, tries: int = 5,
             base: float = 2.0, factor: float = 2.0) -> Transport:
    """Wrap a transport with backoff. See the module docstring for why POST is not retried on network errors."""

    def wrapped(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
        delay = base
        for attempt in range(1, tries + 1):
            try:
                status, data = send(method, url, headers, body)
            except OSError as exc:  # URLError, timeouts and resets are OSErrors
                if method != "GET" or attempt == tries:
                    raise BatchApiError(f"network error on {method} {url}: {exc}") from exc
            else:
                if status not in RETRY_STATUS or attempt == tries:
                    return status, data
            sleep(delay)
            delay *= factor
        raise AssertionError("unreachable")

    return wrapped


def _json(status: int, data: bytes, what: str) -> Any:
    if status >= 400:
        raise BatchApiError(f"{what} failed: HTTP {status}: {data[:500].decode('utf-8', 'replace')}", status)
    try:
        return json.loads(data or b"{}")
    except json.JSONDecodeError as exc:
        raise BatchApiError(f"{what} returned invalid JSON") from exc


def _jsonl(data: bytes) -> Iterable[dict[str, Any]]:
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


@dataclass(frozen=True, slots=True)
class BatchStatus:
    done: bool
    state: str
    counts: dict[str, int]


def _blank_result(custom_id: Any) -> dict[str, Any]:
    return {"custom_id": custom_id, "ok": False, "text": "", "finish_reason": None, "input_tokens": None,
            "output_tokens": None, "thinking_tokens": None, "error": None}


class AnthropicBatchClient:
    provider = "anthropic"
    max_requests = ANTHROPIC_MAX_REQUESTS
    max_bytes = ANTHROPIC_MAX_BYTES

    def __init__(self, model: str, api_key: str, *, base_url: str = "https://api.anthropic.com",
                 transport: Transport | None = None, api_version: str = "2023-06-01",
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self._api_version = api_version
        self._send = retrying(transport or default_transport(), sleep=sleep)

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._key, "anthropic-version": self._api_version, "content-type": "application/json"}

    def submit(self, requests: list[dict[str, Any]], intent_id: str | None = None) -> str:
        body = {
            "requests": [
                {
                    "custom_id": r["custom_id"],
                    "params": {
                        "model": self.model,
                        "max_tokens": r["max_output_tokens"],
                        "temperature": 0,
                        "system": r["system"],
                        "messages": [{"role": "user", "content": r["user"]}],
                    },
                }
                for r in requests
            ]
        }
        status, data = self._send("POST", f"{self.base_url}/v1/messages/batches", self._headers(),
                                  json.dumps(body, ensure_ascii=False).encode("utf-8"))
        return _json(status, data, "Anthropic batch submit")["id"]

    def find_submitted(self, intent_id: str, since_epoch: float, expected_count: int, exclude: set[str]) -> list[str]:
        """Batches created since `since_epoch` with exactly `expected_count` requests. Anthropic batches carry no
        metadata, so size and creation time are the only available match."""
        status, data = self._send("GET", f"{self.base_url}/v1/messages/batches?limit=100", self._headers(), None)
        found = []
        for info in _json(status, data, "Anthropic batch list").get("data", []):
            created = _parse_iso(info.get("created_at"))
            total = sum((info.get("request_counts") or {}).values())
            if info["id"] not in exclude and created is not None and created >= since_epoch - 120 and total == expected_count:
                found.append(info["id"])
        return found

    def status(self, batch_id: str) -> BatchStatus:
        status, data = self._send("GET", f"{self.base_url}/v1/messages/batches/{batch_id}", self._headers(), None)
        info = _json(status, data, "Anthropic batch status")
        state = info.get("processing_status", "")
        return BatchStatus(done=state == "ended", state=state, counts=info.get("request_counts") or {})

    def results(self, batch_id: str) -> list[dict[str, Any]]:
        status, data = self._send("GET", f"{self.base_url}/v1/messages/batches/{batch_id}", self._headers(), None)
        info = _json(status, data, "Anthropic batch status")
        url = info.get("results_url") or f"{self.base_url}/v1/messages/batches/{batch_id}/results"
        status, data = self._send("GET", url, self._headers(), None)
        if status >= 400:
            raise BatchApiError(f"Anthropic batch results failed: HTTP {status}", status)
        return [normalise_anthropic(line) for line in _jsonl(data)]


def _parse_iso(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def normalise_anthropic(line: dict[str, Any]) -> dict[str, Any]:
    result = line.get("result") or {}
    base = _blank_result(line.get("custom_id"))
    if result.get("type") != "succeeded":
        base["error"] = f"{result.get('type')}: {(result.get('error') or {}).get('message', '')}".strip(": ")
        return base
    message = result.get("message") or {}
    usage = message.get("usage") or {}
    stop = message.get("stop_reason")
    text = "".join(block.get("text", "") for block in message.get("content") or [] if block.get("type") == "text")
    base.update(text=text, finish_reason="length" if stop == "max_tokens" else "stop",
                input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
    if stop == "max_tokens":
        base["error"] = "output token limit reached"
    elif stop == "refusal":
        base["error"] = "the model refused to answer"
    elif not text.strip():
        base["error"] = "empty reply"
    else:
        base["ok"] = True
    return base


class OpenAIBatchClient:
    provider = "openai"
    max_requests = OPENAI_MAX_REQUESTS
    max_bytes = OPENAI_MAX_BYTES

    def __init__(self, model: str, api_key: str, *, base_url: str = "https://api.openai.com/v1",
                 transport: Transport | None = None, reasoning_effort: str | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self._key = api_key
        self._send = retrying(transport or default_transport(), sleep=sleep)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"}

    def submit(self, requests: list[dict[str, Any]], intent_id: str | None = None) -> str:
        def body_for(r: dict[str, Any]) -> dict[str, Any]:
            # Temperature is not sent: GPT-5 family models accept only their default.
            body: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                "max_completion_tokens": r["max_output_tokens"],
            }
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
            return body

        lines = [json.dumps({"custom_id": r["custom_id"], "method": "POST", "url": "/v1/chat/completions",
                             "body": body_for(r)}, ensure_ascii=False) for r in requests]
        file_id = self._upload(("\n".join(lines) + "\n").encode("utf-8"))
        body: dict[str, Any] = {"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"}
        if intent_id:
            body["metadata"] = {"candidate_fit_intent": intent_id}
        status, data = self._send("POST", f"{self.base_url}/batches",
                                  {**self._headers(), "content-type": "application/json"},
                                  json.dumps(body).encode("utf-8"))
        return _json(status, data, "OpenAI batch create")["id"]

    def find_submitted(self, intent_id: str, since_epoch: float, expected_count: int, exclude: set[str]) -> list[str]:
        """Batches tagged with this intent id (OpenAI keeps the metadata we attached at creation)."""
        status, data = self._send("GET", f"{self.base_url}/batches?limit=100", self._headers(), None)
        return [info["id"] for info in _json(status, data, "OpenAI batch list").get("data", [])
                if info["id"] not in exclude and (info.get("metadata") or {}).get("candidate_fit_intent") == intent_id]

    def _upload(self, payload: bytes) -> str:
        boundary = f"candidatefit{uuid.uuid4().hex}"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"batch.jsonl\"\r\n"
            "Content-Type: application/jsonl\r\n\r\n"
        ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")
        headers = {**self._headers(), "content-type": f"multipart/form-data; boundary={boundary}"}
        status, data = self._send("POST", f"{self.base_url}/files", headers, body)
        return _json(status, data, "OpenAI file upload")["id"]

    def status(self, batch_id: str) -> BatchStatus:
        status, data = self._send("GET", f"{self.base_url}/batches/{batch_id}", self._headers(), None)
        info = _json(status, data, "OpenAI batch status")
        state = info.get("status", "")
        return BatchStatus(done=state in {"completed", "failed", "expired", "cancelled"}, state=state,
                           counts=info.get("request_counts") or {})

    def results(self, batch_id: str) -> list[dict[str, Any]]:
        status, data = self._send("GET", f"{self.base_url}/batches/{batch_id}", self._headers(), None)
        info = _json(status, data, "OpenAI batch status")
        out: list[dict[str, Any]] = []
        for key, normalise in (("output_file_id", normalise_openai), ("error_file_id", normalise_openai_error)):
            file_id = info.get(key)
            if not file_id:
                continue
            status, data = self._send("GET", f"{self.base_url}/files/{file_id}/content", self._headers(), None)
            if status >= 400:
                raise BatchApiError(f"OpenAI batch file {file_id} failed: HTTP {status}", status)
            out.extend(normalise(line) for line in _jsonl(data))
        return out


def normalise_openai(line: dict[str, Any]) -> dict[str, Any]:
    response = line.get("response") or {}
    body = response.get("body") or {}
    base = _blank_result(line.get("custom_id"))
    if response.get("status_code") != 200 or not body.get("choices"):
        base["error"] = f"HTTP {response.get('status_code')}: {(body.get('error') or line.get('error') or {})}"
        return base
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    finish = choice.get("finish_reason")
    text = (choice.get("message") or {}).get("content") or ""
    base.update(text=text, finish_reason="length" if finish == "length" else "stop",
                input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                thinking_tokens=reasoning)
    if finish == "length":
        used = usage.get("completion_tokens") or 0
        if reasoning and used and reasoning / used >= 0.5:
            base["error"] = (f"output token limit reached; {reasoning} of {used} tokens were hidden reasoning. "
                             "Raise the output cap or lower the reasoning effort")
        else:
            base["error"] = "output token limit reached"
    elif not text.strip():
        base["error"] = "empty reply"
    else:
        base["ok"] = True
    return base


def normalise_openai_error(line: dict[str, Any]) -> dict[str, Any]:
    error = line.get("error") or {}
    result = _blank_result(line.get("custom_id"))
    result["error"] = f"{error.get('code', 'error')}: {error.get('message', '')}"
    return result


def load_api_key(env_name: str, environ: dict[str, str] | None = None) -> str:
    key = (environ if environ is not None else os.environ).get(env_name, "").strip()
    if not key:
        raise BatchApiError(f"{env_name} is not set; export the key in this shell (it is never stored by this tool)")
    return key


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    """Batch cost in USD. Returns the price source so the estimate can be checked against it."""
    price = PRICES_PER_MILLION.get(model)
    if price is None:
        return {"model": model, "known_price": False}
    usd = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000 * BATCH_DISCOUNT
    return {"model": model, "known_price": True, "usd": round(usd, 2), "usd_exact": usd,
            "price_source": price["source"], "input_tokens": input_tokens, "output_tokens": output_tokens}
