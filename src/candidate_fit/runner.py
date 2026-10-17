"""Turn one run plan into work: build the items, run them on the chosen backend, export the CSV.

The same plan comes from the terminal wizard (candidate-fit wizard) or from flags (candidate-fit run).
Runs are resumable: the output folder is fixed by (task, backend, model), and results are keyed by item id and tagged
with a fingerprint of the prompts and settings, so rerunning the same plan continues where it stopped, a different
sample reuses what it can, and a changed prompt never reuses stale results.

JD extraction has more than one prompt version. Each version has its own folder (see tasks.output_task_dir), so running a
new prompt never writes into the results of an older one.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from candidate_fit.batch_runner import export_csv, run_batches
from candidate_fit.cloud_batch import (
    REASONING_EFFORTS,
    AnthropicBatchClient,
    OpenAIBatchClient,
    estimate_cost,
    load_api_key,
)
from candidate_fit.local_runner import LOCAL_MODELS, LocalRunner, LocalSettings
from candidate_fit.tasks import (
    DEFAULT_REFERENCE_DATE,
    TASK_NAMES,
    build_items,
    default_jd_prompt_version,
    jd_prompt,
    output_task_dir,
    task_fingerprint,
)

OUTPUT_ROOT = Path(os.getenv("CANDIDATE_FIT_OUTPUT", "outputs/production"))
LOCAL_LARGE_RUN = 500  # items above which the summary warns that a local run is sequential
CLOUD_PROVIDERS = {
    "anthropic": {"label": "Claude Haiku 5.5 (claude-haiku-5-5)", "model": "claude-haiku-5-5",
                  "key_env": "ANTHROPIC_API_KEY"},
    "openai": {"label": "GPT-5.6 Luna (set the exact model id)", "model": "gpt-5.6-luna",
               "key_env": "OPENAI_API_KEY", "model_env": "OPENAI_LUNA_MODEL"},
}


def timestamped(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True, slots=True)
class RunPlan:
    task: str
    backend: str  # "local" or "cloud"
    local_model: str | None = None  # key in LOCAL_MODELS
    cloud_provider: str | None = None  # key in CLOUD_PROVIDERS
    cloud_model: str | None = None  # exact model id sent to the provider
    limit: int | None = None
    seed: int | None = 42
    max_output_tokens: int | None = None
    reference_date: str = DEFAULT_REFERENCE_DATE
    max_attempts: int = 2
    poll_seconds: float = 60.0
    replace_other_model: bool = False
    reasoning_budget: int = 0  # local thinking budget
    reasoning_effort: str | None = None  # OpenAI only; leave unset for models that do not reason
    max_cost_usd: float | None = None  # refuse a cloud run whose worst-case estimate is higher
    item_ids: tuple[str, ...] | None = None  # run exactly these items (in dataset order); cannot be combined with limit
    output_root: Path = OUTPUT_ROOT
    jd_prompt_version: str | None = None  # JD extraction only; None means the current default (tasks.DEFAULT_JD_PROMPT_VERSION)

    def validate(self) -> None:
        if self.task not in TASK_NAMES:
            raise ValueError(f"unknown task {self.task!r}")
        if self.jd_prompt_version is not None:
            if self.task != "jd_extraction":
                raise ValueError("a JD prompt version applies to the jd_extraction task only")
            jd_prompt(self.jd_prompt_version)  # raises for an unknown version
        written_for = jd_prompt(self.effective_jd_prompt_version).models if self.task == "jd_extraction" else None
        if written_for and not (self.backend == "local" and self.local_model in written_for):
            raise ValueError(f"JD prompt {self.effective_jd_prompt_version} is written for {', '.join(written_for)} only")
        if self.backend == "local":
            if self.local_model not in LOCAL_MODELS:
                raise ValueError(f"choose a local model from {', '.join(LOCAL_MODELS)}")
        elif self.backend == "cloud":
            if self.cloud_provider not in CLOUD_PROVIDERS:
                raise ValueError(f"choose a cloud provider from {', '.join(CLOUD_PROVIDERS)}")
            if not self.cloud_model:
                raise ValueError("a cloud model id is required")
            if self.reasoning_effort and self.reasoning_effort not in REASONING_EFFORTS:
                raise ValueError(f"reasoning effort must be one of {', '.join(REASONING_EFFORTS)}")
            if self.reasoning_effort and self.cloud_provider != "openai":
                raise ValueError("reasoning effort applies to OpenAI models only")
        else:
            raise ValueError("backend must be local or cloud")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be a positive number")
        if self.item_ids is not None and self.limit:
            raise ValueError("give either a list of item ids or a limit, not both")

    @property
    def effective_jd_prompt_version(self) -> str | None:
        """The version asked for, else the one written for this local model (gpt-oss-20b has its own), else None for the default."""
        if self.jd_prompt_version is not None:
            return self.jd_prompt_version
        return default_jd_prompt_version(self.local_model) if self.backend == "local" else None

    def folder(self) -> Path:
        model = self.local_model if self.backend == "local" else f"{self.cloud_provider}-{self.cloud_model}"
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in model)
        return self.output_root / output_task_dir(self.task, self.effective_jd_prompt_version) / f"{self.backend}-{safe}"


def plan_summary(plan: RunPlan, item_count: int, input_tokens: int, max_output: int) -> tuple[list[str], float | None]:
    """Lines to show before confirming, and the worst-case USD estimate for cloud runs (None when unknown)."""
    lines = [f"Task: {plan.task}", f"Items: {item_count:,}" + (" (the listed ids)" if plan.item_ids is not None else " (all)" if not plan.limit else f" (sample of {plan.limit})")]
    if plan.task == "jd_extraction":
        lines.insert(1, f"JD prompt: {jd_prompt(plan.effective_jd_prompt_version).version}")
    usd: float | None = None
    if plan.backend == "local":
        lines.append(f"Backend: local, {LOCAL_MODELS[plan.local_model][2]}")
        if item_count > LOCAL_LARGE_RUN:
            lines.append(f"Note: local runs handle one item at a time, so {item_count:,} items can take many hours or days. "
                         "A cloud batch or a smaller sample is faster.")
    else:
        provider = CLOUD_PROVIDERS[plan.cloud_provider]
        lines.append(f"Backend: cloud, {provider['label']}, model id {plan.cloud_model}, BATCH ONLY")
        lines.append(f"Data: the resume and JD text of every item is sent to {plan.cloud_provider} servers.")
        cost = estimate_cost(plan.cloud_model, input_tokens, item_count * max_output)
        if cost["known_price"]:
            usd = cost["usd_exact"]  # the ceiling check uses the unrounded figure
            shown = f"${usd:,.2f}" if usd >= 0.01 else "under $0.01"
            lines.append(f"Estimated batch cost: about {shown} (input {input_tokens:,} tokens; "
                         f"output at the cap of {max_output:,} tokens per item, the worst case)")
            lines.append(f"Price source: {cost['price_source']}. Verify before a billed run.")
        else:
            lines.append("No price in the table for this model id; cost not estimated.")
        if plan.reasoning_effort:
            lines.append(f"Reasoning effort: {plan.reasoning_effort} (hidden reasoning counts against the output cap)")
    lines.append(f"Output folder: {plan.folder()}")
    return lines, usd


def make_cloud_client(plan: RunPlan, environ: dict[str, str] | None = None):
    provider = CLOUD_PROVIDERS[plan.cloud_provider]
    key = load_api_key(provider["key_env"], environ)
    if plan.cloud_provider == "anthropic":
        return AnthropicBatchClient(plan.cloud_model, key)
    return OpenAIBatchClient(plan.cloud_model, key, reasoning_effort=plan.reasoning_effort)


def execute(plan: RunPlan, *, confirm: Callable[[list[str]], bool], log: Callable[[str], None] = timestamped,
            sleep=None, clock=None, client=None, export_only: bool = False) -> dict:
    plan.validate()
    items = build_items(plan.task, limit=plan.limit, seed=plan.seed, reference_date=plan.reference_date,
                        max_output_tokens=plan.max_output_tokens, item_ids=plan.item_ids,
                        jd_prompt_version=plan.effective_jd_prompt_version)
    if not items:
        raise RuntimeError("no items to run")
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise RuntimeError("the dataset has duplicate item ids; results are keyed by id, so every item needs a unique one")
    fingerprint = task_fingerprint(plan.task, reference_date=plan.reference_date, max_output_tokens=plan.max_output_tokens,
                                   jd_prompt_version=plan.effective_jd_prompt_version)
    model_name = plan.cloud_model if plan.backend == "cloud" else LOCAL_MODELS[plan.local_model][1]
    if export_only:  # rebuild results.csv from saved results; nothing is loaded, confirmed or sent
        return {"csv": str(export_csv(plan.task, items, plan.folder(), model_name, fingerprint)), "exported_only": True}
    cloud_client = None
    if plan.backend == "cloud":
        cloud_client = client or make_cloud_client(plan)  # checks the API key now, before anything is confirmed
    max_output = items[0].max_output_tokens
    input_tokens = sum(max(1, (len(i.system) + len(i.user)) // 4) for i in items)  # about 4 characters per token
    summary, usd = plan_summary(plan, len(items), input_tokens, max_output)
    if plan.max_cost_usd is not None and usd is not None and usd > plan.max_cost_usd:
        raise RuntimeError(f"estimated worst-case cost ${usd:,.2f} is above the limit of ${plan.max_cost_usd:,.2f}; "
                           "lower --limit or raise --max-cost-usd")
    if not confirm(summary):
        return {"cancelled": True}
    folder = plan.folder()
    if plan.backend == "cloud":
        kwargs = {"fingerprint": fingerprint, "max_attempts": plan.max_attempts, "poll_seconds": plan.poll_seconds, "log": log}
        if sleep:
            kwargs["sleep"] = sleep
        if clock:
            kwargs["clock"] = clock
        result = run_batches(plan.task, items, cloud_client, folder, **kwargs)
    else:
        runner = LocalRunner(LocalSettings(plan.local_model, plan.reasoning_budget, plan.replace_other_model), log=log)
        result = runner.run(plan.task, items, folder, fingerprint=fingerprint, max_attempts=plan.max_attempts)
    result["csv"] = str(export_csv(plan.task, items, folder, model_name, fingerprint))
    return result
