# One model's production run, shown in its own tab. Called by run_prod_final.ps1; you can also run it directly.
param([Parameter(Mandatory)][string]$Name, [Parameter(Mandatory)][string]$Model, [Parameter(Mandatory)][string]$Ids,
      [Parameter(Mandatory)][int]$Workers, [Parameter(Mandatory)][int]$Pool)
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\candidate-fit.exe"
$out = Join-Path $repo "outputs\prod_run_final\$Name"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $repo
Write-Host "=== ${Name}: $Model, $Workers at once, pool $Pool. Log: $out\run.log" -ForegroundColor Cyan
& $py run --task jd_extraction --backend local --local-model $Model --ids-file $Ids --order length `
  --output-root $out --workers $Workers --kv-pool $Pool --progress-every 25 --yes 2>&1 |
  Tee-Object -FilePath "$out\run.log" -Append
Write-Host "=== $Name finished or stopped. Rerun scripts\run_prod_final.ps1 to resume." -ForegroundColor Yellow
