param(
  [Parameter(Mandatory = $true)]
  [string]$Image,

  [string]$Proc    = "$PSScriptRoot\data\processed",
  [string]$Arts    = "$PSScriptRoot\artifacts",
  [int]$ApiPort    = 8000,
  [int]$UiPort     = 8501,
  [switch]$StartApi = $true,
  [switch]$StartUI  = $true
)

# 1) Data check
Write-Host "[1/4] Preparing data..."
New-Item -ItemType Directory -Path $Arts -Force | Out-Null

$trainCsv = Join-Path $Proc "train.csv"
$valCsv   = Join-Path $Proc "val.csv"

if ((Test-Path -LiteralPath $trainCsv) -and (Test-Path -LiteralPath $valCsv)) {
  Write-Host "  Skipping: found processed/train.csv and processed/val.csv"
} else {
  Write-Host "  ERROR: Missing processed csvs at $Proc" -ForegroundColor Yellow
  Write-Host "  Expected:"
  Write-Host "   $trainCsv"
  Write-Host "   $valCsv"
  throw "Place your pre-split train/val CSVs in $Proc, then rerun."
}

# 2) Train
Write-Host "[2/4] Training model..."

$trainCmd = @'
export PYTHONPATH=/app/src;
python -u -m ai_tweets.cli train \
  --config configs/gpu.yaml \
  --train-csv data/train.csv \
  --eval-csv  data/val.csv
'@

docker run --rm --gpus all `
  -e TRANSFORMERS_VERBOSITY=info `
  -e HF_HUB_DISABLE_TELEMETRY=1 `
  -v "${Proc}:/app/data" `
  -v "${Arts}:/app/artifacts" `
  $Image bash -lc $trainCmd

if ($LASTEXITCODE -ne 0) {
  throw "Training failed (exit $LASTEXITCODE)"
}

$ModelDir = "/app/artifacts/checkpoints/final"

# 3) API
if ($StartApi) {
  Write-Host "[3/4] Starting API on http://localhost:$ApiPort ..."
  docker rm -f crisis-api 2>$null | Out-Null

  docker run -d --rm --name crisis-api --gpus all `
    -p ${ApiPort}:8000 `
    -v "${Arts}:/app/artifacts" `
    -e MODEL_DIR="$ModelDir" `
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 `
    $Image uvicorn ai_tweets.api:app --host 0.0.0.0 --port 8000 --log-level info | Out-Null

  Start-Sleep -Seconds 3
  try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:${ApiPort}/healthz" -TimeoutSec 5 -ErrorAction Stop
    if ($resp.ok -ne $true) { throw "unhealthy" }
    Write-Host "  API healthy." -ForegroundColor Green
  } catch {
    Write-Host "  (health check skipped/failed — try http://localhost:${ApiPort}/docs)" -ForegroundColor Yellow
  }
}

# 4) UI
if ($StartUI) {
  Write-Host "[4/4] Launching Streamlit at http://localhost:$UiPort ..."
  docker rm -f crisis-ui 2>$null | Out-Null

  docker run -d --rm --name crisis-ui --gpus all `
    -p ${UiPort}:8501 `
    -v "${Arts}:/app/artifacts" `
    -e MODEL_DIR="$ModelDir" `
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 `
    $Image streamlit run src/ai_tweets/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false | Out-Null

  Write-Host "Done. Visit:" -ForegroundColor Green
  Write-Host "  - API docs:     http://localhost:${ApiPort}/docs"
  Write-Host "  - Streamlit UI: http://localhost:${UiPort}"
  Write-Host "Artifacts in $Arts (model, metrics, confusion_matrix.png)"
}
