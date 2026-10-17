"""Terminal wizard: choose a task, then a local or cloud model, review the plan, and run it.

Cloud runs always use the provider's batch API. The wizard never offers a synchronous cloud call.
Nothing is sent to a cloud provider until the plan has been shown, the data sharing acknowledged, and the
submission confirmed.
"""
from __future__ import annotations

import os
from typing import Callable, Sequence

from candidate_fit.local_runner import LOCAL_MODELS
from candidate_fit.runner import CLOUD_PROVIDERS, RunPlan, execute, timestamped
from candidate_fit.tasks import TASK_LABELS, TASK_NAMES

Reader = Callable[[str], str]


def choose(reader: Reader, out: Callable[[str], None], title: str, options: Sequence[tuple[str, str]]) -> str:
    out(title)
    for number, (_key, label) in enumerate(options, 1):
        out(f"  {number}) {label}")
    while True:
        answer = reader("Enter a number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        out("  Please enter one of the numbers shown.")


def ask_int(reader: Reader, out: Callable[[str], None], question: str, default: int | None) -> int | None:
    while True:
        shown = "all" if default is None else str(default)
        answer = reader(f"{question} [{shown}]: ").strip()
        if not answer:
            return default
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        out("  Enter a positive whole number, or press Enter for the default.")


def ask_yes_no(reader: Reader, out: Callable[[str], None], question: str, default: bool) -> bool:
    shown = "Y/n" if default else "y/N"
    while True:
        answer = reader(f"{question} [{shown}]: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        out("  Answer y or n.")


def ask_text(reader: Reader, question: str, default: str) -> str:
    answer = reader(f"{question} [{default}]: ").strip()
    return answer or default


def build_plan(reader: Reader = input, out: Callable[[str], None] = print,
               environ: dict[str, str] | None = None) -> RunPlan | None:
    """Ask the questions and return a plan, or None if a required key is missing."""
    env = environ if environ is not None else os.environ
    out("Candidate-fit production run")
    task = choose(reader, out, "What do you want to run?", [(t, TASK_LABELS[t]) for t in TASK_NAMES])
    backend = choose(reader, out, "Where should the model run?", [
        ("local", "Local model on the dual GPU service"),
        ("cloud", "Cloud model, batch API only"),
    ])
    local_model = cloud_provider = cloud_model = reasoning_effort = None
    replace = False
    if backend == "local":
        local_model = choose(reader, out, "Which local model?",
                             [(key, label) for key, (_lane, _model, label) in LOCAL_MODELS.items()])
        replace = ask_yes_no(reader, out, "If its GPU already serves another model, replace that model?", default=False)
    else:
        cloud_provider = choose(reader, out, "Which cloud model?",
                                [(key, value["label"]) for key, value in CLOUD_PROVIDERS.items()])
        provider = CLOUD_PROVIDERS[cloud_provider]
        default_model = env.get(provider.get("model_env", ""), "") or provider["model"]
        cloud_model = ask_text(reader, "Exact model id to send to the provider", default_model)
        if not env.get(provider["key_env"]):
            out(f"  {provider['key_env']} is not set in this shell. Export it, then start the wizard again.")
            return None
        if cloud_provider == "openai":
            effort = choose(reader, out, "Reasoning effort (hidden reasoning counts against the output cap)", [
                ("", "Do not set it (models that do not reason)"), ("minimal", "minimal"), ("low", "low"),
                ("medium", "medium"), ("high", "high"),
            ])
            reasoning_effort = effort or None
    limit = ask_int(reader, out, "How many items to run (Enter for all)", None)
    seed = 42
    if limit:
        seed = ask_int(reader, out, "Random seed for the sample", 42) or 42
    return RunPlan(task=task, backend=backend, local_model=local_model, cloud_provider=cloud_provider,
                   cloud_model=cloud_model, limit=limit, seed=seed, replace_other_model=replace,
                   reasoning_effort=reasoning_effort)


def run_wizard(reader: Reader = input, out: Callable[[str], None] = print, environ: dict[str, str] | None = None,
               sleep=None, clock=None) -> dict | None:
    plan = build_plan(reader, out, environ)
    if plan is None:
        return None

    def confirm(summary: list[str]) -> bool:
        out("")
        for line in summary:
            out(line)
        out("")
        if plan.backend == "cloud":
            if not ask_yes_no(reader, out, "I am allowed to send this resume and JD text to a third-party provider", default=False):
                return False
            return ask_yes_no(reader, out, "Submit this batch to the cloud provider now?", default=False)
        return ask_yes_no(reader, out, "Start this local run now?", default=True)

    log = timestamped if out is print else out
    result = execute(plan, confirm=confirm, log=log, sleep=sleep, clock=clock)
    if result.get("cancelled"):
        out("Cancelled. Nothing was sent.")
        return result
    out("")
    for key, value in result.items():
        out(f"{key}: {value}")
    return result
