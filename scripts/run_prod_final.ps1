# Full production run: both models as two tabs in ONE Windows Terminal window. Progress shows live; every attempt is saved to
# results.jsonl as it finishes (the checkpoint), and each tab's log is in run.log. Rerunning resumes where each run stopped.
# Refuses to start if a production run is already going, so a second launch cannot duplicate the work.
param([switch]$DryRun)
$repo = Split-Path -Parent $PSScriptRoot
$one = Join-Path $PSScriptRoot "run_one_model.ps1"
$runs = @(
  @{ name = "r9700";  model = "qwen3.8-27b";            ids = "config\full_r9700_ids.txt";  workers = 8; pool = 151552 },
  @{ name = "9070xt"; model = "qwen3.8-27b-unsloth-q3"; ids = "config\full_9070xt_ids.txt"; workers = 2; pool = 49152 }
)
$args = @()
foreach ($r in $runs) {
  if ($args.Count -gt 0) { $args += ";" }
  $args += @("new-tab", "--title", $r.name, "powershell", "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $one,
             "-Name", $r.name, "-Model", $r.model, "-Ids", $r.ids, "-Workers", $r.workers, "-Pool", $r.pool)
}
$args = @("-w", "prod_run_final") + $args
if ($DryRun) { "wt.exe " + ($args -join " "); exit 0 }
$running = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'candidate-fit.exe' -and $_.CommandLine -match 'prod_run_final' }
if ($running) { Write-Host "A production run is already going ($(@($running).Count) process(es)). Not starting another." -ForegroundColor Red; exit 1 }
if (-not (Get-Command wt.exe -ErrorAction SilentlyContinue)) { Write-Host "Windows Terminal (wt.exe) not found; install it or run scripts\run_one_model.ps1 per model." -ForegroundColor Red; exit 1 }
Start-Process wt.exe -ArgumentList $args -WorkingDirectory $repo
Write-Host "Started both models as tabs in the window 'prod_run_final'."
