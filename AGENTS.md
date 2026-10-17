# Agent rules for this repository

## Never start a model from this repo. Always go through `dual_gpu_setup`.

All local-model work (loading, unloading, inference, metrics) is owned by the **`dual_gpu_setup`**
service at `F:\Projects\Dual_gpu_setup`, listening on `http://127.0.0.1:8090`. This repo is only a
client of that service.

**Do not:**

- start, stop or restart `llama-server`, `llama.cpp`, Ollama, vLLM, LM Studio or any other model
  server, from a shell, a script, `subprocess` or `Popen`;
- load a GGUF directly or add code that does;
- point inference at the backend ports `8081` / `8082`. They bypass project attribution and metrics;
- kill, replace or reconfigure a model server process on your own;
- invent a new launcher, wrapper or "quick local run" path for a model.

**Do:**

1. Make sure the service is up. It is idempotent and safe to call before any session:
   `py F:\Projects\Dual_gpu_setup\launch_service.py --config F:\Projects\Dual_gpu_setup\example.dual_gpu.toml`
2. Check what is loaded: `GET http://127.0.0.1:8090/api/status`. If the model you need is already on
   a lane, use it as is.
3. If a model must be loaded or changed, use the orchestrator's deploy API (`POST /api/deploy`, or
   `POST /api/clients/deploy_parallel`, see `F:\Projects\Dual_gpu_setup\CLIENT_API.md`), or ask the
   user to use the dashboard. Deploying through the API is allowed. It reloads models, which stops
   whatever is running on those lanes, so confirm with the user first if a run may be in progress.
4. Run inference only through the orchestrator, with the registered client from
   `.candidate-fit-client.json` (`orchestrator_url`, `client_id`) and an explicit `--lane` and `--model`:
   `candidate-fit score` / `score-optimized` with `--orchestrator-url http://127.0.0.1:8090`
   (see [`config/dual_gpu.task.toml`](config/dual_gpu.task.toml) and [`scripts/smoke_score.ps1`](scripts/smoke_score.ps1)),
   or `DualGPUClient` in Python (see [`scripts/run_eval_extremes.py`](scripts/run_eval_extremes.py)).
5. If the service is not reachable or the model is not loaded, **stop and tell the user**. Do not
   work around it by starting a model yourself.

Lanes on this machine: `r9700` (32 GB, runs Qwen3.8-27B-Q4_K_M) and `9070xt` (runs the smaller
model). `/api/clients/check_fit` on the service is the authority on whether a model fits a lane.

### Known exceptions (read-only, do not extend)

- `candidate-fit audit-context --backend-url http://127.0.0.1:8081` calls only `/tokenize` and
  `/apply-template` on a model that `dual_gpu_setup` has already loaded. It performs no generation.
  Never use it to start a model.
- The `--base-url` / `LocalLLMClient` path in [`src/candidate_fit/client.py`](src/candidate_fit/client.py)
  exists for hosted OpenAI-compatible APIs. Do not use it for local models; use `DualGPUClient`.

## Output token limit

The default `--max-output-tokens` is 6144. Qwen hit it on pair `150030` in the optimized pipeline
(see `outputs/eval-extremes-8/`). Client code rejects any `length` finish rather than saving a
partial score, so a raised limit must stay inside the deployed context window, which is 24576 on both lanes.
