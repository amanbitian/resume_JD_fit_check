from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from candidate_fit.client import DualGPUClient, LocalLLMClient
from candidate_fit.context_audit import audit_context
from candidate_fit.dataset import dataset_summary, iter_pairs
from candidate_fit.dual_gpu import load_client_config, register_client, save_client_config, upsert_project
from candidate_fit.local_runner import LOCAL_MODELS
from candidate_fit.metrics import evaluate_jsonl
from candidate_fit.optimized import (
    JD_ANALYSIS_SCHEMA,
    JD_ANALYSIS_PROMPT,
    PAIR_SCORING_SCHEMA,
    PAIR_SCORING_PROMPT,
    RESUME_ANALYSIS_SCHEMA,
    RESUME_ANALYSIS_PROMPT,
    score_dataset_optimized,
)
from candidate_fit.prompt import ACTIVE_SYSTEM_PROMPT, build_user_prompt
from candidate_fit.scorer import ScoreOptions, score_dataset
from candidate_fit.tasks import DEFAULT_JD_PROMPT_VERSION, JD_PROMPTS


DEFAULT_DATASET = (
    r"F:\SLM\training_data\external_data_1010\resume_jd_hugging Face"
    r"\train-00000-of-00002.parquet"
)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="candidate-fit", description="Score resume/JD pairs with a local LLM")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="Show parquet files, row count, and schema")
    inspect_parser.add_argument("dataset", nargs="?", default=DEFAULT_DATASET)

    register = sub.add_parser("register", help="Register this repo with dual_gpu_setup for attributed metrics")
    register.add_argument("--orchestrator-url", default="http://127.0.0.1:8090")
    register.add_argument("--project-name", default="candidate_scoring_using_local_LLM")
    register.add_argument("--metrics-output", default="metrics/dual_gpu_runs.jsonl")
    register.add_argument("--github-repo")
    register.add_argument("--repo-path", default=".", help="Local repo folder shown on the dual_gpu_setup project")
    register.add_argument("--git-remote", help="Git remote URL shown on the dual_gpu_setup project")
    register.add_argument("--config-out", default=".candidate-fit-client.json")

    preview = sub.add_parser("preview", help="Export the exact prompt and resume/JD inputs for sample rows")
    preview.add_argument("dataset", nargs="?", default=DEFAULT_DATASET)
    preview.add_argument("--rows", type=int, default=3)
    preview.add_argument("--offset", type=int, default=0)
    preview.add_argument("--output")
    preview.add_argument(
        "--optimized",
        action="store_true",
        help="Show the three-stage cached pipeline prompts and raw JD/resume inputs",
    )

    audit = sub.add_parser("audit-context", help="Tokenize dataset prompts and prove they fit a model context")
    audit.add_argument("dataset", nargs="?", default=DEFAULT_DATASET)
    audit.add_argument("--backend-url", required=True, help="Loaded llama-server URL, for example http://127.0.0.1:8081")
    audit.add_argument("--context-window", type=int, default=16_384)
    audit.add_argument("--max-output-tokens", type=int, default=6_144)
    audit.add_argument("--safety-tokens", type=int, default=512)
    audit.add_argument("--max-resume-chars", type=int, default=100_000)
    audit.add_argument("--max-jd-chars", type=int, default=100_000)
    audit.add_argument("--workers", type=int, default=8)
    audit.add_argument("--limit", type=int)
    audit.add_argument("--offset", type=int, default=0)
    audit.add_argument("--output")

    def add_score_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("dataset", nargs="?", default=DEFAULT_DATASET)
        command.add_argument("--output", required=True, help="Append-only JSONL prediction file")
        command.add_argument("--base-url", default=os.getenv("DGPU_BASE_URL", "http://127.0.0.1:8081"))
        command.add_argument("--model", default=os.getenv("DGPU_MODEL_NAME", "local-model"))
        command.add_argument(
            "--workers",
            type=int,
            default=1,
            help="Concurrent requests; match the endpoint's supported parallelism",
        )
        command.add_argument("--limit", type=int)
        command.add_argument("--offset", type=int, default=0)
        command.add_argument("--shard-index", type=int, default=0)
        command.add_argument("--num-shards", type=int, default=1)
        command.add_argument("--batch-size", type=int, default=128)
        command.add_argument("--resume-col", default="resume")
        command.add_argument("--jd-col", default="jd")
        command.add_argument("--label-col", default="label")
        command.add_argument("--max-resume-chars", type=int, default=100_000)
        command.add_argument("--max-jd-chars", type=int, default=100_000)
        command.add_argument("--max-output-tokens", type=int, default=6_144)
        command.add_argument("--timeout", type=float, default=900.0)
        command.add_argument("--retries", type=int, default=2)
        command.add_argument("--max-errors", type=int, default=10, help="Stop after this many failed pairs")
        command.add_argument("--no-json-schema", action="store_true")
        command.add_argument("--no-resume", action="store_true", help="Do not skip successful IDs already in output")
        command.add_argument("--client-config", default=".candidate-fit-client.json")
        command.add_argument("--client-id", default=os.getenv("DGPU_CLIENT_ID"))
        command.add_argument("--task-label", default="candidate_fit_scoring")
        command.add_argument("--connection-type", default="batch")
        command.add_argument(
            "--no-store-history",
            action="store_true",
            help="Ask dual_gpu_setup not to archive the conversation (default archives it, when the service has "
            "project.store_chat_history enabled)",
        )
        command.add_argument(
            "--api-key-env",
            default="LOCAL_LLM_API_KEY",
            help="Environment variable containing the API key (the secret is never written to output)",
        )
        command.add_argument(
            "--disable-thinking",
            action="store_true",
            help="Send enable_thinking=false to compatible Qwen endpoints to reduce latency and tokens",
        )
        command.add_argument("--orchestrator-url", help="Use the attributed dual_gpu_setup /api/chat endpoint")
        command.add_argument("--lane", default=os.getenv("DGPU_LANE_KEY"), help="Loaded lane: r9700 or 9070xt")

    score = sub.add_parser("score", help="Single-pass full analysis for each pair")
    add_score_arguments(score)
    optimized = sub.add_parser("score-optimized", help="Cache repeated JD/resume analysis and run compact pair matching")
    add_score_arguments(optimized)
    optimized.add_argument("--analysis-cache", default="outputs/analysis-cache.jsonl")

    evaluate = sub.add_parser("evaluate", help="Evaluate predictions against parquet gold labels copied into JSONL")
    evaluate.add_argument("predictions")
    evaluate.add_argument("--output", help="Optional metrics JSON path")

    sub.add_parser("wizard", help="Interactive wizard: choose a task, a local or cloud model, then run it")

    run = sub.add_parser("run", help="Run one production task without the wizard (same plan as the wizard)")
    run.add_argument("--task", required=True, choices=["resume_extraction", "jd_extraction", "resume_jd_pair"])
    run.add_argument("--backend", required=True, choices=["local", "cloud"])
    run.add_argument("--local-model", choices=list(LOCAL_MODELS))
    run.add_argument("--cloud-provider", choices=["anthropic", "openai"])
    run.add_argument("--cloud-model", help="Exact model id, for example claude-haiku-5-5")
    run.add_argument("--limit", type=int, help="Run a seeded random sample of this many items")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-output-tokens", type=int)
    run.add_argument("--reference-date", default="2026-10-17")
    run.add_argument("--max-attempts", type=int, default=2)
    run.add_argument("--poll-seconds", type=float, default=60.0)
    run.add_argument("--reasoning-budget", type=int, default=0, help="Local thinking budget; 0 turns thinking off")
    run.add_argument("--replace-other-model", action="store_true", help="Replace another model on the lane")
    run.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"],
                     help="OpenAI only; hidden reasoning counts against the output cap")
    run.add_argument("--jd-prompt-version", choices=list(JD_PROMPTS),
                     help=f"JD extraction only: which prompt version to run (default {DEFAULT_JD_PROMPT_VERSION}; gpt-oss-20b "
                     "uses v2.3-gptoss, which only it can run). Each version writes to its own output folder, so a new "
                     "prompt never overwrites older results")
    run.add_argument("--max-cost-usd", type=float, help="Refuse a cloud run whose worst-case estimate is higher")
    run.add_argument("--ids-file", help="Run exactly these items: a JSON list of ids (or objects with an id field) "
                     "or a text file with one id per line. Cannot be combined with --limit")
    run.add_argument("--export-only", action="store_true",
                     help="Rebuild results.csv from the saved results of this plan; no model is loaded or called")
    run.add_argument("--yes", action="store_true", help="Skip the confirmation prompt (for scripts)")
    return parser


def _stopped() -> None:
    print("\nStopped. Rerun the same command to resume where it left off.", file=sys.stderr)
    sys.exit(130)


def _run_wizard() -> None:
    from candidate_fit.wizard import run_wizard

    try:
        result = run_wizard()
    except (KeyboardInterrupt, EOFError):
        _stopped()
    except (RuntimeError, ValueError) as exc:  # BatchApiError, LocalRunError and RunLockedError are RuntimeErrors
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if result is not None and not result.get("cancelled") else 1)


def _run_production(args: argparse.Namespace) -> None:
    from candidate_fit.runner import RunPlan, execute
    from candidate_fit.tasks import load_item_ids

    if args.backend == "local" and not args.local_model:
        raise SystemExit("--local-model is required for --backend local")
    if args.backend == "cloud" and not (args.cloud_provider and args.cloud_model):
        raise SystemExit("--cloud-provider and --cloud-model are required for --backend cloud")
    plan = RunPlan(task=args.task, backend=args.backend, local_model=args.local_model,
                   cloud_provider=args.cloud_provider, cloud_model=args.cloud_model, limit=args.limit,
                   seed=args.seed, max_output_tokens=args.max_output_tokens, reference_date=args.reference_date,
                   max_attempts=args.max_attempts, poll_seconds=args.poll_seconds,
                   replace_other_model=args.replace_other_model, reasoning_budget=args.reasoning_budget,
                   reasoning_effort=args.reasoning_effort, max_cost_usd=args.max_cost_usd,
                   item_ids=load_item_ids(Path(args.ids_file)) if args.ids_file else None,
                   jd_prompt_version=args.jd_prompt_version)

    def confirm(summary: list[str]) -> bool:
        print("\n".join(summary))
        if args.yes:
            return True
        return input("Proceed? [y/N]: ").strip().lower() in ("y", "yes")

    try:
        result = execute(plan, confirm=confirm, export_only=args.export_only)
    except (KeyboardInterrupt, EOFError):
        _stopped()
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if not result.get("cancelled") else 1)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if args.command == "wizard":
        _run_wizard()
    if args.command == "run":
        _run_production(args)
    if args.command == "inspect":
        print(json.dumps(dataset_summary(args.dataset), ensure_ascii=False, indent=2))
        return
    if args.command == "register":
        registration = register_client(
            orchestrator_url=args.orchestrator_url,
            project_name=args.project_name,
            output_path=args.metrics_output,
            github_repo=args.github_repo,
        )
        save_client_config(args.config_out, registration, args.orchestrator_url)
        project = upsert_project(
            orchestrator_url=args.orchestrator_url,
            name=args.project_name,
            repo_path=args.repo_path,
            git_remote=args.git_remote or args.github_repo,
        )
        print(json.dumps({**registration, "project": project, "config_file": str(Path(args.config_out).resolve())}, indent=2))
        return
    if args.command == "preview":
        rows = []
        for record in iter_pairs(args.dataset, offset=args.offset, limit=args.rows):
            row = {
                "pair_id_for_reference_only": record.pair_id,
                "model_input": {
                    "job_description": record.job_description,
                    "resume": record.resume,
                },
            }
            if args.optimized:
                row["jd_extraction_user_prompt"] = (
                    f"<JOB_DESCRIPTION>\n{record.job_description}\n</JOB_DESCRIPTION>"
                )
                row["resume_extraction_user_prompt"] = f"<RESUME>\n{record.resume}\n</RESUME>"
            else:
                row["rendered_user_prompt"] = build_user_prompt(record.resume, record.job_description)
            rows.append(row)
        if args.optimized:
            payload = {
                "note": (
                    "Only these prompts and JD/resume-derived evidence are sent. Labels, row IDs, "
                    "source, and dataset domain metadata are excluded. The pair prompt receives the "
                    "two cached analyses, indexed exactly as described by its system prompt."
                ),
                "jd_analysis_system_prompt": JD_ANALYSIS_PROMPT,
                "jd_analysis_schema": JD_ANALYSIS_SCHEMA,
                "resume_analysis_system_prompt": RESUME_ANALYSIS_PROMPT,
                "resume_analysis_schema": RESUME_ANALYSIS_SCHEMA,
                "pair_scoring_system_prompt": PAIR_SCORING_PROMPT,
                "pair_scoring_schema": PAIR_SCORING_SCHEMA,
                "rows": rows,
            }
        else:
            payload = {
                "note": "Only system_prompt and rendered_user_prompt are sent to the model; labels and metadata are excluded.",
                "system_prompt": ACTIVE_SYSTEM_PROMPT,
                "rows": rows,
            }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        return
    if args.command == "audit-context":
        result = audit_context(
            iter_pairs(args.dataset, limit=args.limit, offset=args.offset),
            backend_url=args.backend_url,
            context_window=args.context_window,
            max_output_tokens=args.max_output_tokens,
            safety_tokens=args.safety_tokens,
            max_resume_chars=args.max_resume_chars,
            max_jd_chars=args.max_jd_chars,
            workers=args.workers,
        )
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        if not result["safe_for_all_checked_rows"]:
            sys.exit(3)
        return
    if args.command == "evaluate":
        result = evaluate_jsonl(args.predictions)
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        return

    client_config = load_client_config(args.client_config)
    client_id = args.client_id or client_config.get("client_id")
    orchestrator_url = args.orchestrator_url or client_config.get("orchestrator_url")
    if orchestrator_url:
        if not client_id:
            raise SystemExit("The dual_gpu_setup API requires registration. Run: candidate-fit register")
        if not args.lane:
            raise SystemExit("--lane is required when using the dual_gpu_setup API")
        client = DualGPUClient(
            orchestrator_url=orchestrator_url,
            lane=args.lane,
            model=args.model,
            client_id=client_id,
            task_label=args.task_label,
            timeout=args.timeout,
            retries=args.retries,
            max_tokens=args.max_output_tokens,
            connection_type=args.connection_type,
            store_history=not args.no_store_history,
        )
    else:
        client = LocalLLMClient(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            use_json_schema=not args.no_json_schema,
            max_tokens=args.max_output_tokens,
            api_key=os.getenv(args.api_key_env),
            client_id=client_id,
            task_label=args.task_label,
            connection_type=args.connection_type,
            extra_body={"enable_thinking": False} if args.disable_thinking else None,
        )
    records = iter_pairs(
        args.dataset,
        resume_col=args.resume_col,
        jd_col=args.jd_col,
        label_col=args.label_col,
        batch_size=args.batch_size,
        limit=args.limit,
        offset=args.offset,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.command == "score-optimized":
        result = score_dataset_optimized(
            records,
            client=client,
            output_path=args.output,
            cache_path=args.analysis_cache,
            resume=not args.no_resume,
            max_errors=args.max_errors,
            workers=args.workers,
        )
    else:
        result = score_dataset(
            records,
            client=client,
            output_path=args.output,
            options=ScoreOptions(
                max_resume_chars=args.max_resume_chars,
                max_jd_chars=args.max_jd_chars,
                workers=args.workers,
                max_errors=args.max_errors,
            ),
            resume=not args.no_resume,
        )
    print(json.dumps(result, indent=2))
    if result["interrupted"]:
        sys.exit(130)
    if result["errors"] or result["aborted"]:
        sys.exit(2)


if __name__ == "__main__":
    main()

