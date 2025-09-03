param(
  [string]$Image = "sandeep_pandey/crisis-llm:gpu-latest",
  [switch]$StartApi = $true,
  [switch]$StartUI = $true
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Get-Location).Path
$Proc  = Join-Path $RepoRoot "data\processed"
$Arts  = Join-Path $RepoRoot "artifacts"

Write-Host "[1/4] Preparing data..."
# Ensure artifacts dir exists
New-Item -ItemType Directory -Force -Path $Arts | Out-Null

$trainCsv = Join-Path $Proc "train.csv"
$valCsv   = Join-Path $Proc "val.csv"
if ( (Test-Path $trainCsv) -and (Test-Path $valCsv) ) {
  Write-Host "  Skipping: found processed/train.csv and processed/val.csv"
} else {
  Write-Warning "  Expected CSVs not found in data\processed. Put train.csv and val.csv there."
}

Write-Host "[2/4] Training model..."
# Train with your local CSVs mounted into the container
# Also upgrades accelerate to avoid keep_torch_compile error seen earlier
$trainCmd = @"
python -m pip install -q --upgrade 'accelerate>=1.2.1' || true
export PYTHONPATH=/app/src
python -u -m ai_tweets.cli train \
  --config configs/gpu.yaml \
  --train-csv data/train.csv \
  --eval-csv  data/val.csv
"@

docker run --rm --gpus all `
  -e HF_HUB_DISABLE_TELEMETRY=1 `
  -v "${Proc}:/app/data" `
  -v "${Arts}:/app/artifacts" `
  $Image sh -lc "$trainCmd"

if ($LASTEXITCODE -ne 0) {
  throw "docker run (training) failed with exit code $LASTEXITCODE"
}

if ($StartApi) {
  Write-Host "[3/4] Starting API on http://localhost:8000 ..."
  # Stop an old container if running
  try { docker stop crisis-api | Out-Null } catch { }

  docker run -d --rm --name crisis-api --gpus all `
    -p 8000:8000 `
    -v "${Arts}:/app/artifacts" `
    $Image sh -lc "export PYTHONPATH=/app/src; uvicorn ai_tweets.api:app --host 0.0.0.0 --port 8000"

  Start-Sleep -Seconds 3
  try {
    $resp = Invoke-WebRequest -UseBasicParsing http://localhost:8000/healthz -TimeoutSec 5
    if ($resp.StatusCode -eq 200) {
      Write-Host "  API is healthy." -ForegroundColor Green
    } else {
      Write-Host "  API returned status $($resp.StatusCode). Check logs: docker logs crisis-api" -ForegroundColor Yellow
    }
  } catch {
    Write-Host "  API health check failed (you can still visit /docs). Check logs: docker logs crisis-api" -ForegroundColor Yellow
  }
}

if ($StartUI) {
  Write-Host "[4/4] Launching Streamlit at http://localhost:8501 ..."
  try { docker stop crisis-ui | Out-Null } catch { }

  docker run -d --rm --name crisis-ui `
    -p 8501:8501 `
    -v "${Arts}:/app/artifacts" `
    $Image sh -lc "export PYTHONPATH=/app/src; streamlit run apps/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"

  Write-Host "Done. Visit:"
  Write-Host "  - API docs:     http://localhost:8000/docs"
  Write-Host "  - Streamlit UI: http://localhost:8501"
  Write-Host "Artifacts in $Arts (model, metrics, confusion_matrix.png)"
}
