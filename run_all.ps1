param([string]$Image = "sandeep_pandey/crisis-llm:gpu-latest")
$ErrorActionPreference = "Stop"

# --- Paths ---
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Raw  = Join-Path $RepoRoot "data\raw"
$Proc = Join-Path $RepoRoot "data\processed"
$Arts = Join-Path $RepoRoot "artifacts"

New-Item -ItemType Directory -Force -Path $Raw,$Proc,$Arts | Out-Null

Write-Host "[1/5] Preparing data..."
$trainCsv = Join-Path $Proc "train.csv"
$valCsv   = Join-Path $Proc "val.csv"
if ((Test-Path $trainCsv) -and (Test-Path $valCsv)) {
  Write-Host "  Skipping: found processed/train.csv and processed/val.csv"
} else {
  docker run --rm `
    -v "${Raw}:/app/data/raw" `
    -v "${Proc}:/app/data/processed" `
    -v "${RepoRoot}:/app" `
    $Image `
    sh -lc "export PYTHONPATH=/app/src; python -m ai_tweets.cli prepare --raw-dir /app/data/raw --out-dir /app/data/processed"
}

Write-Host "[2/5] Training model..."
docker run --rm --gpus all `
  -v "${Proc}:/app/data" `
  -v "${Arts}:/app/artifacts" `
  -v "${RepoRoot}:/app" `
  $Image `
  sh -lc "export PYTHONPATH=/app/src; python -m ai_tweets.cli train --config configs/gpu.yaml --train-csv data/train.csv --eval-csv data/val.csv"

Write-Host "[3/5] Stamping label maps..."
docker run --rm `
  -v "${Arts}:/app/artifacts" `
  $Image `
  python -c "import json,os; p='/app/artifacts/label_map.json'; json.dump({'id2label':{0:'non_disaster',1:'disaster'}, 'label2id':{'non_disaster':0,'disaster':1}}, open(p,'w')); print('Wrote', p)"

Write-Host "[4/5] Starting API on http://localhost:8000 ..."
# stop any old container
$null = docker rm -f crisis-api 2>$null
docker run -d --rm --gpus all --name crisis-api `
  -p 8000:8000 `
  -v "${Arts}:/app/artifacts" `
  -v "${RepoRoot}:/app" `
  $Image `
  sh -lc "export PYTHONPATH=/app/src; uvicorn ai_tweets.api:app --host 0.0.0.0 --port 8000"

Start-Sleep -Seconds 2
try { (Invoke-WebRequest -UseBasicParsing http://localhost:8000/healthz).Content | Write-Host } catch { Write-Host "Health check failed (API may still be starting)"; }

Write-Host "[5/5] Launching Streamlit on http://localhost:8501 ..."
$null = docker rm -f crisis-ui 2>$null
docker run -d --rm --name crisis-ui `
  -p 8501:8501 `
  -v "${Arts}:/app/artifacts" `
  -v "${RepoRoot}:/app" `
  $Image `
  sh -lc "export PYTHONPATH=/app/src; streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501"

Write-Host ""
Write-Host "Done. Visit:"
Write-Host "  - API docs:     http://localhost:8000/docs"
Write-Host "  - Streamlit UI: http://localhost:8501"
Write-Host "Artifacts in $Arts (model, metrics, confusion_matrix.png)"
