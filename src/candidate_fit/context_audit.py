from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from itertools import islice
from typing import Any, Iterable
from urllib import request

from candidate_fit.dataset import PairRecord
from candidate_fit.prompt import ACTIVE_SYSTEM_PROMPT, build_user_prompt


@dataclass(frozen=True, slots=True)
class ChatTemplate:
    before_system: str
    between_system_user: str
    after_user: str

    def render(self, system: str, user: str) -> str:
        return self.before_system + system + self.between_system_user + user + self.after_user


def _post_json(url: str, body: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object from {url}")
    return value


def load_chat_template(backend_url: str) -> ChatTemplate:
    system_marker = "__CANDIDATE_FIT_SYSTEM_MARKER_7E2C__"
    user_marker = "__CANDIDATE_FIT_USER_MARKER_9A41__"
    value = _post_json(
        backend_url.rstrip("/") + "/apply-template",
        {
            "messages": [
                {"role": "system", "content": system_marker},
                {"role": "user", "content": user_marker},
            ]
        },
    )
    rendered = value.get("prompt")
    if not isinstance(rendered, str) or rendered.count(system_marker) != 1 or rendered.count(user_marker) != 1:
        raise RuntimeError("Backend chat template did not preserve the audit markers exactly once")
    before_system, remainder = rendered.split(system_marker, 1)
    between_system_user, after_user = remainder.split(user_marker, 1)
    return ChatTemplate(before_system, between_system_user, after_user)


def audit_context(
    records: Iterable[PairRecord],
    *,
    backend_url: str,
    context_window: int,
    max_output_tokens: int = 6_144,
    safety_tokens: int = 512,
    max_resume_chars: int = 100_000,
    max_jd_chars: int = 100_000,
    workers: int = 8,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    input_budget = context_window - max_output_tokens - safety_tokens
    if input_budget <= 0:
        raise ValueError("context_window must exceed max_output_tokens + safety_tokens")

    backend_url = backend_url.rstrip("/")
    template = load_chat_template(backend_url)

    def count(record: PairRecord) -> dict[str, Any]:
        user_prompt = build_user_prompt(
            record.resume,
            record.job_description,
            max_resume_chars=max_resume_chars,
            max_jd_chars=max_jd_chars,
        )
        rendered = template.render(ACTIVE_SYSTEM_PROMPT, user_prompt)
        response = _post_json(
            backend_url + "/tokenize",
            {"content": rendered, "add_special": False},
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError("Backend /tokenize response did not contain a token list")
        return {
            "pair_id": record.pair_id,
            "row_index": record.row_index,
            "input_tokens": len(tokens),
            "raw_resume_chars": len(record.resume),
            "raw_jd_chars": len(record.job_description),
        }

    checked = 0
    over_budget = 0
    top: list[dict[str, Any]] = []
    iterator = iter(records)
    batch_size = workers * 4
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="context-audit") as pool:
        while batch := list(islice(iterator, batch_size)):
            for result in pool.map(count, batch):
                checked += 1
                if result["input_tokens"] > input_budget:
                    over_budget += 1
                top.append(result)
                top.sort(key=lambda item: item["input_tokens"], reverse=True)
                del top[5:]

    maximum = top[0]["input_tokens"] if top else 0
    return {
        "backend_url": backend_url,
        "context_window": context_window,
        "max_output_tokens": max_output_tokens,
        "safety_tokens": safety_tokens,
        "input_token_budget": input_budget,
        "rows_checked": checked,
        "rows_over_budget": over_budget,
        "maximum_input_tokens": maximum,
        "minimum_headroom_after_reserved_output": context_window - maximum - max_output_tokens,
        "safe_for_all_checked_rows": checked > 0 and over_budget == 0,
        "largest_inputs": top,
    }
