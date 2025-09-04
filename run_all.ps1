\
param(
  [Parameter(Mandatory=$true)][string]$Image,
  [int]$ApiPort = 8000,
  [int]$UiPort  = 8501,
  [switch]$SkipTrain,
  [switch]$NoApi,
  [switch]$NoUI
)

$ErrorActionPreference = "Stop"

# -- Resolve repo root and paths --
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$Proc = Join-Path $RepoRoot "data\processed"
$Arts = Join-Path $RepoRoot "artifacts"
$PatchedStreamlit = Join-Path $RepoRoot "src\ai_tweets\streamlit_app.py"

# Ensure directories
New-Item -ItemType Directory -Force -Path $Arts | Out-Null

Write-Host "[1/4] Preparing data..." -ForegroundColor Cyan
$trainCsv = Join-Path $Proc "train.csv"
$valCsv   = Join-Path $Proc "val.csv"
if (Test-Path $trainCsv -and Test-Path $valCsv) {
  Write-Host "  Skipping: found processed/train.csv and processed/val.csv"
} else {
  throw "Missing data. Expected: `n  $trainCsv `n  $valCsv"
}

# Helper: stop a container if running
function Stop-ContainerIfRunning($name) {
  try { docker rm -f $name | Out-Null } catch { }
}

# Helper: quick port check
function Test-PortBusy([int]$port) {
  try {
    $tcp = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction Stop
    return $true
  } catch {
    return $false
  }
}

# Train
if (-not $SkipTrain) {
  Write-Host "[2/4] Training model..." -ForegroundColor Cyan
  $cmd = @"
python -m pip install -q --upgrade 'accelerate>=1.2.1' || true
export PYTHONPATH=/app/src
python -u -m ai_tweets.cli train \
  --config configs/gpu.yaml \
  --train-csv data/train.csv \
  --eval-csv  data/val.csv
"@

  docker run --rm --gpus all `
    -e TRANSFORMERS_VERBOSITY=info `
    -e HF_HUB_DISABLE_TELEMETRY=1 `
    -v "${Proc}:/app/data" `
    -v "${Arts}:/app/artifacts" `
    $Image sh -lc $cmd

  if ($LASTEXITCODE -ne 0) { throw "Training failed (exit $LASTEXITCODE)" }
}

# API
if (-not $NoApi) {
  Write-Host "[3/4] Starting API on http://localhost:$ApiPort ..." -ForegroundColor Cyan
  if (Test-PortBusy $ApiPort) {
    throw "Port $ApiPort is already in use. Pick a different -ApiPort or stop the conflicting process."
  }
  Stop-ContainerIfRunning "crisis-api"
  docker run -d --rm --name crisis-api --gpus all `
    -p ${ApiPort}:8000 `
    -v "${Arts}:/app/artifacts" `
    -e MODEL_DIR="/app/artifacts/checkpoints/final" `
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 `
    $Image uvicorn ai_tweets.api:app --host 0.0.0.0 --port 8000 --log-level info

  Start-Sleep -Seconds 5
  Write-Host "  Try: http://localhost:$ApiPort/healthz  and  http://localhost:$ApiPort/docs"
}

# UI
if (-not $NoUI) {
  Write-Host "[4/4] Launching Streamlit on http://localhost:$UiPort ..." -ForegroundColor Cyan
  if (Test-PortBusy $UiPort) {
    throw "Port $UiPort is already in use. Pick a different -UiPort or stop the conflicting process."
  }
  Stop-ContainerIfRunning "crisis-ui"
  docker run -d --rm --name crisis-ui --gpus all `
    -p ${UiPort}:8501 `
    -v "${Arts}:/app/artifacts" `
    -v "${PatchedStreamlit}:/app/src/ai_tweets/streamlit_app.py:ro" `
    -e MODEL_DIR="/app/artifacts/checkpoints/final" `
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 `
    $Image streamlit run src/ai_tweets/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false

  Start-Sleep -Seconds 3
  Write-Host "Done. Visit:" -ForegroundColor Green
  Write-Host "  - API docs:     http://localhost:$ApiPort/docs" -ForegroundColor Green
  Write-Host "  - Streamlit UI: http://localhost:$UiPort" -ForegroundColor Green
  Write-Host "Artifacts in $Arts (checkpoints, metrics, confusion_matrix.png)" -ForegroundColor Green
}
