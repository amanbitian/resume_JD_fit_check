from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
import time
from typing import Any
from urllib import error, parse, request

from candidate_fit.prompt import ACTIVE_SYSTEM_PROMPT, COMPACT_OUTPUT_SCHEMA


class OutputTokenLimitError(RuntimeError):
    """The backend stopped before completing the structured response."""


# dual_gpu_setup has no authentication (API_CONTRACT.md: "local-trust-boundary service"), so the
# only protection for resume/JD text is that the connection never leaves this machine.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
# An opener with an empty ProxyHandler ignores HTTP(S)_PROXY, so local traffic is never proxied.
_LOCAL_OPENER = request.build_opener(request.ProxyHandler({}))


def require_loopback(url: str) -> str:
    host = parse.urlsplit(url).hostname
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"Refusing non-loopback orchestrator URL {url!r}: dual_gpu_setup is unauthenticated "
            "and must only be reached on 127.0.0.1/localhost (pass allow_remote=True to override)"
        )
    return url


@dataclass(frozen=True, slots=True)
class Completion:
    content: str
    usage: dict[str, Any]
    latency_seconds: float
    finish_reason: str | None = None
    reasoning_text: str | None = None
    context_window: int | None = None
    dgpu_run_id: str | None = None
    timing: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None
    client_delivery: dict[str, Any] | None = None


class LocalLLMClient:
    """Small dependency-free client for llama.cpp/LM Studio OpenAI endpoints."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
        retries: int = 2,
        use_json_schema: bool = True,
        max_tokens: int = 6_144,
        client_id: str | None = None,
        task_label: str = "candidate_fit_scoring",
        connection_type: str = "batch",
        no_history: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("DGPU_BASE_URL") or "http://127.0.0.1:8081").rstrip("/")
        self.model = model or os.getenv("DGPU_MODEL_NAME") or "local-model"
        self.client_id = client_id or os.getenv("DGPU_CLIENT_ID")
        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY") or self.client_id or "not-needed-for-local"
        self.timeout = timeout
        self.retries = retries
        self.use_json_schema = use_json_schema
        self.max_tokens = max_tokens
        self.task_label = task_label
        self.connection_type = connection_type
        self.no_history = no_history
        self.extra_body = dict(extra_body or {})
        self.generation_config = {
            "temperature": 0,
            "seed": 42,
            "use_json_schema": self.use_json_schema,
            "extra_body": self.extra_body,
        }

    @property
    def completions_url(self) -> str:
        if self.base_url.endswith("/v1/chat/completions") or self.base_url.endswith("/chat/completions"):
            return self.base_url
        path = self.base_url.split("://", 1)[-1].partition("/")[2].rstrip("/")
        if path == "v1" or path.startswith("v1/"):
            return self.base_url + "/chat/completions"
        return self.base_url + "/v1/chat/completions"

    def complete(
        self,
        user_prompt: str,
        *,
        system_prompt: str = ACTIVE_SYSTEM_PROMPT,
        response_schema: dict[str, Any] | None = COMPACT_OUTPUT_SCHEMA,
        stage: str | None = None,  # accepted for interface parity; direct endpoints have no per-stage tagging
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "seed": 42,
            "max_tokens": self.max_tokens,
            "stream": False,
            # llama.cpp extension: reuse the stable system/rubric prefix.
            "cache_prompt": True,
        }
        body.update(self.extra_body)
        if self.use_json_schema and response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "candidate_fit",
                    "strict": True,
                    "schema": response_schema,
                },
            }

        schema_fallback_used = False
        attempt = 0
        last_error: Exception = RuntimeError("No request was attempted")
        while attempt <= self.retries:
            started = time.perf_counter()
            try:
                payload = self._post(body)
                latency = time.perf_counter() - started
                content = payload["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Local model returned an empty completion")
                finish_reason = payload["choices"][0].get("finish_reason")
                if finish_reason in {"length", "max_tokens"}:
                    raise OutputTokenLimitError(
                        f"Model output hit its token limit ({finish_reason}); refusing to save a partial score"
                    )
                return Completion(
                    content=content,
                    usage=payload.get("usage", {}),
                    latency_seconds=latency,
                    finish_reason=finish_reason,
                    dgpu_run_id=payload.get("dgpu_run_id"),
                )
            except OutputTokenLimitError:
                raise
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                # Older llama.cpp/LM Studio versions may not accept json_schema.
                if exc.code in (400, 422) and "response_format" in body and not schema_fallback_used:
                    body.pop("response_format", None)
                    schema_fallback_used = True
                    continue
                last_error = RuntimeError(f"LLM HTTP {exc.code}: {detail}")
            except (error.URLError, TimeoutError, OSError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep((0.6 * (2**attempt)) + random.random() * 0.2)
            attempt += 1
        raise RuntimeError(f"Local LLM request failed after {self.retries + 1} attempts: {last_error}")

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-DGPU-Task-Label": self.task_label,
            "X-DGPU-Connection-Type": self.connection_type,
            "X-DGPU-Agent-Role": "evaluator",
        }
        if self.no_history:
            headers["X-DGPU-No-History"] = "true"
        req = request.Request(
            self.completions_url,
            data=encoded,
            headers=headers,
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("LLM endpoint returned a non-object JSON response")
        return value


class DualGPUClient:
    """Client for dual_gpu_setup's attributed evaluation API."""

    def __init__(
        self,
        *,
        orchestrator_url: str,
        lane: str,
        model: str,
        client_id: str,
        task_label: str = "candidate_fit_scoring",
        timeout: float = 900.0,
        retries: int = 2,
        max_tokens: int = 6_144,
        allow_remote: bool = False,
        connection_type: str = "batch",
        store_history: bool = True,
    ) -> None:
        self.base_url = (orchestrator_url if allow_remote else require_loopback(orchestrator_url)).rstrip("/")
        self.lane = lane
        self.model = model
        self.client_id = client_id
        self.task_label = task_label
        self.connection_type = connection_type
        # True archives the full conversation in dual_gpu_setup's chat history, if the service enables it.
        self.store_history = store_history
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self.context_window: int | None = None
        self.generation_config = {
            "temperature": 0,
            "seed": 42,
            "use_json_schema": False,
            "extra_body": {},
        }

    def complete(
        self,
        user_prompt: str,
        *,
        system_prompt: str = ACTIVE_SYSTEM_PROMPT,
        response_schema: dict[str, Any] | None = COMPACT_OUTPUT_SCHEMA,
        stage: str | None = None,
    ) -> Completion:
        if self.context_window is None:
            self._load_context_window()
        body = {
            "client_id": self.client_id,
            "target": self.lane,
            # One label per stage (e.g. "candidate_fit_scoring.jd_extraction") so the dashboard's
            # Analytics tab can break latency and tokens down by prompt.
            "task_label": f"{self.task_label}.{stage}" if stage else self.task_label,
            "workload_kind": "evaluation",
            # app.py reads connection_type from the body on /api/chat (the header only applies to the
            # OpenAI gateway); it keys the per-project usage rollup. Not listed in API_CONTRACT.md.
            "connection_type": self.connection_type,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "seed": 42,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout,
            "store_history": self.store_history,
        }
        attempt = 0
        last_error: Exception = RuntimeError("No request was attempted")
        while attempt <= self.retries:
            started = time.perf_counter()
            try:
                payload = self._post(body)
                result = payload.get("results", {}).get(self.lane)
                if not isinstance(result, dict):
                    raise RuntimeError(f"Missing result for lane {self.lane}")
                if result.get("status") != "completed":
                    raise RuntimeError(str(result.get("error_text") or "dual_gpu_setup run failed"))
                content = result.get("output_text")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Local model returned an empty completion")
                finish_reason = result.get("finish_reason")
                if finish_reason in {"length", "max_tokens"}:
                    raise OutputTokenLimitError(
                        f"Model output hit its token limit ({finish_reason}); refusing to save a partial score"
                    )
                timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
                latency = float(timing.get("end_to_end_duration_ms") or 0) / 1000
                if latency <= 0:
                    latency = time.perf_counter() - started
                return Completion(
                    content=content,
                    usage=result.get("usage") or {},
                    latency_seconds=latency,
                    finish_reason=finish_reason,
                    reasoning_text=result.get("reasoning_text"),
                    context_window=self.context_window,
                    dgpu_run_id=result.get("run_id"),
                    timing=timing,
                    resources=result.get("resources") or {},
                    client_delivery=result.get("client_delivery"),
                )
            except OutputTokenLimitError:
                raise
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                last_error = RuntimeError(f"dual_gpu_setup HTTP {exc.code}: {detail}")
            except (error.URLError, TimeoutError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep((0.6 * (2**attempt)) + random.random() * 0.2)
            attempt += 1
        raise RuntimeError(f"dual_gpu_setup request failed after {self.retries + 1} attempts: {last_error}")

    def _load_context_window(self) -> None:
        try:
            with _LOCAL_OPENER.open(self.base_url + "/api/status", timeout=10) as response:
                status = json.loads(response.read().decode("utf-8"))
            for server in status.get("servers", []):
                if server.get("target") == self.lane and server.get("context_window") is not None:
                    self.context_window = int(server["context_window"])
                    return
        except (error.URLError, TimeoutError, OSError, ValueError, TypeError, KeyError):
            # Status metadata is useful for auditing but should not block an otherwise valid run.
            return

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        req = request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _LOCAL_OPENER.open(req, timeout=self.timeout + 10) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("dual_gpu_setup returned a non-object JSON response")
        return value

