# Candidate Fit Scoring with Local LLMs

This repository scores **each resume/job-description pair** with either a local or hosted
OpenAI-compatible model. It streams parquet instead of loading the corpus into memory, writes one
crash-safe JSONL record per completed pair, and resumes without repeating successful requests.

The supplied dataset has this schema:

| column | meaning |
|---|---|
| `resume` | resume text |
| `jd` | job-description text |
| `label` | `No Fit`, `Potential Fit`, or `Good Fit` |
| `source` | data source |
| `resume_domain`, `jd_domain` | domain metadata |

Dataset size and class balance can change as files are refreshed. Run `candidate-fit inspect` and
the analysis notebook against the exact input files before each production run. Model selection
should emphasize macro-F1 and per-class recall, not accuracy alone.

## Recommended model placement on this machine

Use `dual_gpu_setup`'s `/api/clients/check_fit` as the final authority because context size and
KV cache also consume VRAM.

| Local GGUF already present | File size | Recommended lane |
|---|---:|---|
| `Qwen3.8-27B-Q4_K_M` | 15.66 GiB | R9700 32 GB; fails the 9070's 14.40 GiB safe budget |
| `gpt-oss-20b-GGUF` (MXFP4) | 12.11 GB | 9070 at the validated 16,384-token context |

The CLI accepts any compatible model ID. Run candidate models on the same stratified evaluation
sample and compare macro-F1, Good-Fit recall, calibration, latency, and cost before the full run.

## Setup

Python 3.11+ is required.

```powershell
cd F:\SLM\candidate_scoring_using_local_LLM
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

Inspect the dataset without loading it into memory:

```powershell
.\.venv\Scripts\candidate-fit.exe inspect `
  "F:\SLM\training_data\external_data_1010\resume_jd_hugging Face\train-00000-of-00002.parquet"
```

## Connect through `dual_gpu_setup`

1. Copy the task block in [`config/dual_gpu.task.toml`](config/dual_gpu.task.toml) into
   `F:/Projects/Dual_gpu_setup/example.dual_gpu.toml`.
2. Start the service:

```powershell
cd F:\Projects\Dual_gpu_setup
py launch_service.py --config example.dual_gpu.toml
```

3. Register this repository once. The generated client file is local and gitignored:

```powershell
cd F:\SLM\candidate_scoring_using_local_LLM
.\.venv\Scripts\candidate-fit.exe register `
  --orchestrator-url http://127.0.0.1:8090 `
  --metrics-output metrics\dual_gpu_runs.jsonl
```

4. Deploy Qwen to `r9700` and GPT-OSS to `9070xt`, then run the `candidate-fit` task. Inference
   uses the attributed `/api/chat` evaluation endpoint. `dual_gpu_setup` receives token, timing,
   GPU, memory, and run-attribution data and writes a copy into `metrics/dual_gpu_runs.jsonl`.

### What is sent to `dual_gpu_setup`, and what it keeps

Every request carries `client_id` (which resolves to the project name registered in step 3),
`target` (the lane), `workload_kind: evaluation`, `connection_type` (`--connection-type`, default
`batch`), `store_history`, `temperature: 0`, `seed: 42`, `max_tokens`, and a `task_label` per
prompt stage: `<task-label>.single_pass`, `.resume_extraction`, `.jd_extraction`, and
`.relevancy_scoring` (default base label `candidate_fit_scoring`). `register` also sets the
project's repo path and git remote. Labels, row IDs and dataset metadata are never sent.

**`store_history` is true by default**, so the full conversation (prompts, resume and JD text, and
model output) is archived in `dual_gpu_setup`'s chat history. Pass `--no-store-history` to opt out
of the archive. Three things to know:

- The archive only exists when the service enables it. It is off by default; set
  `store_chat_history = true` under `[project]` in `example.dual_gpu.toml` and restart the service
  (which unloads the models). `GET /api/chat-history` reports `enabled: false` until then.
- `store_history=false` never kept resume and JD text out of the service. On `/api/chat` the
  metrics database stores every run's full request and output regardless of this setting, so
  treat `dual_gpu_setup`'s database files as containing candidate data either way.
- The service has no authentication. The client refuses non-loopback orchestrator URLs and ignores
  proxy environment variables; keep the service bound to `127.0.0.1`.

> **Rule: this repo never starts a model.** Loading, unloading and inference all go through
> `dual_gpu_setup` on `http://127.0.0.1:8090`; never run `llama-server` yourself or call ports
> 8081/8082 for inference. See [AGENTS.md](AGENTS.md).

For a smoke run against a model already loaded by `dual_gpu_setup` (this also goes through the
orchestrator at port 8090):

```powershell
cd F:\SLM\candidate_scoring_using_local_LLM
.\scripts\smoke_score.ps1 -Lane r9700 -Model Qwen3.8-27B-Q4_K_M -Limit 20
```

For production, use the cached three-stage scorer. It analyzes each unique JD and resume once,
then performs a smaller indexed comparison for every pair:

```powershell
.\.venv\Scripts\candidate-fit.exe score-optimized `
  "F:\SLM\training_data\external_data_1010\resume_jd_hugging Face\train-00000-of-00002.parquet" `
  --output outputs\qwen38-27b-optimized.jsonl `
  --analysis-cache outputs\qwen38-27b-analysis-cache.jsonl `
  --orchestrator-url http://127.0.0.1:8090 `
  --lane r9700 `
  --model Qwen3.8-27B-Q4_K_M `
  --limit 1000
```

To score both training shards, pass the quoted glob
`F:\SLM\training_data\external_data_1010\resume_jd_hugging Face\train-*.parquet`.
Do not pass the whole directory for training because it also contains the held-out test split.

The default `--workers 1` matches the orchestrator's `parallel = 1`. Increase it only when the
server was launched with matching parallel slots; otherwise concurrency increases latency and
VRAM without increasing throughput.

For a hosted OpenAI-compatible endpoint, put the secret in an environment variable and use a
separate output and cache per model/configuration. Qwen endpoints that support the option can use
`--disable-thinking` to reduce evaluation latency and output-token spend:

```powershell
$env:MODEL_API_KEY = "..."
.\.venv\Scripts\candidate-fit.exe score-optimized `
  "F:\path\to\dataset" `
  --base-url "https://provider.example/v1" `
  --model "provider-model-id" `
  --api-key-env MODEL_API_KEY `
  --disable-thinking `
  --workers 8 `
  --output outputs\provider-model-v1.jsonl `
  --analysis-cache outputs\provider-model-v1-cache.jsonl
```

Keep a separate analysis-cache file per model. Both the cache and final output are append-only and
flushed after every successful record. Re-running the same command resumes completed work; a prompt
change automatically uses a new cache fingerprint without deleting old evidence.

Before a full run, prove that every capped prompt fits the exact tokenizer and chat template of
the loaded model. This is a fast tokenizer-only audit; it performs no generation:

```powershell
.\.venv\Scripts\candidate-fit.exe audit-context `
  "F:\SLM\training_data\external_data_1010\resume_jd_hugging Face\train-00000-of-00002.parquet" `
  --backend-url http://127.0.0.1:8081 `
  --context-window 16384 `
  --max-output-tokens 6144 `
  --safety-tokens 512 `
  --output outputs\qwen-context-audit.json
```

Run the same audit against `http://127.0.0.1:8082` for GPT-OSS because tokenizers differ. The
command exits with code 3 if any row exceeds the safe input budget.

## Output and evaluation

Each JSONL row includes a stable pair ID, gold label, application-derived 0-10 score and class,
confidence, full JD analysis, full candidate analysis, role/domain/subdomain alignment, domains
for every work experience, every JD skill and its resume evidence, job-critical skills, seven
per-dimension scores, one check per JD requirement, derived plus points and gaps, human-review
routing, short rationale, token/timing/resource metrics, and the shared dual-GPU run ID. See
[`docs/output-fields.md`](docs/output-fields.md) for the complete contract.
Resume and JD text are deliberately not duplicated in prediction or metrics output; the stable ID,
source path, row index, and content hash provide traceability back to the parquet.

Export the exact full prompt and three raw model inputs (JD + resume only) with:

```powershell
.\.venv\Scripts\candidate-fit.exe preview `
  "F:\SLM\training_data\external_data_1010\resume_jd_hugging Face\train-00000-of-00002.parquet" `
  --rows 3 `
  --output outputs\three-model-inputs-and-full-prompt.json
```

For the recommended cached scorer, export all three exact system prompts plus three raw JD/resume
inputs with:

```powershell
.\.venv\Scripts\candidate-fit.exe preview `
  "F:\SLM\training_data\external_data_1010\resume_jd_hugging Face\train-00000-of-00002.parquet" `
  --rows 3 `
  --optimized `
  --output outputs\three-optimized-model-inputs-and-full-prompts.json
```

No label, dataset domain, source, or row metadata is sent to the model. Extraction receives only
the raw JD or raw resume. Pair scoring receives only the two derived, cached analyses. See
[`docs/optimization-log.md`](docs/optimization-log.md) for measured settings, rejected experiments,
and the requested five-pair benchmark.

```powershell
.\.venv\Scripts\candidate-fit.exe evaluate outputs\qwen38-27b.jsonl `
  --output outputs\qwen38-27b.metrics.json
```

Useful controls:

- `--limit N` for smoke tests.
- `--offset N` to start later in the corpus.
- `--num-shards K --shard-index I` for stable disjoint partitions.
- press Ctrl+C at any time; the active call stops and every previously completed line remains
  flushed to disk.
- rerun the same command and output path to skip successful pair IDs with the same model/prompt
  fingerprint; unfinished, error, or differently configured rows are retried.
- the run stops after 10 failed pairs by default (`--max-errors`) instead of silently burning
  through the dataset when an endpoint or output format is broken.
- a backend `length`/`max_tokens` finish is rejected as an error rather than saved as a score.
- `--no-json-schema` for an older server that rejects structured output.

## Design choices

- The scoring bands are fixed: 0-3 `No Fit`, 4-6 `Potential Fit`, 7-10 `Good Fit`.
- The optimized pipeline derives scores from fixed, versioned dimension weights and enforces
  mandatory-gap caps; labels are then derived from scores. The model cannot emit a contradictory
  overall score or label.
- The supplied shard is passed without truncation (100k-character safety limits per field). Exact
  tokenizer audits show every full pair fits both 16K contexts with a 6,144-token output reserve
  and 512-token safety allowance.
- The model receives only the system rubric plus `<JOB_DESCRIPTION>` and `<RESUME>` text. Gold
  labels, source, domains, row IDs, and dataset metadata are never sent to the model.
- Functional domain means the work performed; industry domain means the sector where it was
  performed. The prompt uses controlled taxonomies and requires evidence from responsibilities,
  not title or employer-name guesses.
- The prompt explicitly treats input text as data and excludes protected traits.
- This is decision support, not an autonomous hiring decision. Validate subgroup behavior and
  require human review before operational use.

