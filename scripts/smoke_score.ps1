param(
    [string]$OrchestratorUrl = "http://127.0.0.1:8090",
    [string]$Lane = $(if ($env:DGPU_LANE_KEY) { $env:DGPU_LANE_KEY } else { "r9700" }),
    [string]$Model = $(if ($env:DGPU_MODEL_NAME) { $env:DGPU_MODEL_NAME } else { "Qwen3.8-27B-Q4_K_M" }),
    [int]$Limit = 20
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv. Follow the setup commands in README.md first."
}
& $python -m candidate_fit score `
    "F:\SLM\training_data\external_data_1010\resume_jd_hugging Face\train-00000-of-00002.parquet" `
    --output (Join-Path $repo "outputs\smoke-$Model.jsonl") `
    --orchestrator-url $OrchestratorUrl `
    --lane $Lane `
    --model $Model `
    --limit $Limit

