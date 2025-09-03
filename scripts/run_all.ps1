param(
  [string]$Image = "sandeep_pandey/crisis-llm:gpu-latest"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path ".").Path
$Raw  = Join-Path $RepoRoot "data\raw"
$Proc = Join-Path $RepoRoot "data\processed"
$Arts = Join-Path $RepoRoot "artifacts"

New-Item -ItemType Directory -Force -Path $Raw,$Proc,$Arts | Out-Null

Write-Host "Image: $Image"
Write-Host ""

Write-Host "[1/5] Preparing data..."
docker run --rm --gpus all `
  -v "${Raw}:/app/data/raw" `
  -v "${Proc}:/app/data/processed" `
  $Image `
  sh -lc "export PYTHONPATH=/app/src; python -m ai_tweets.cli prepare --raw-dir /app/data/raw --out-dir /app/data/processed"


Write-Host "[2/5] Training..."
docker run --rm --gpus all `
  -v "${Proc}:/app/data" `
  -v "${Arts}:/app/artifacts" `
  $Image `
  sh -lc "export PYTHONPATH=/app/src; python -m ai_tweets.cli train --config configs/gpu.yaml --train-csv /app/data/train.csv --eval-csv /app/data/val.csv"


Write-Host "[3/5] Stamping labels..."
docker run --rm `
  -v "${Arts}:/app/artifacts" `
  $Image `
  sh -lc "python -c \"from transformers import AutoConfig; p='/app/artifacts/checkpoints/final'; cfg=AutoConfig.from_pretrained(p); cfg.id2label={0:'non_disaster',1:'disaster'}; cfg.label2id={'non_disaster':0,'disaster':1}; cfg.save_pretrained(p); print('labels saved')\""

Write-Host ""
Write-Host "[4/5] Starting API on http://localhost:8000 ..."
docker rm -f crisis_api 2>$null | Out-Null
docker run -d --gpus all --name crisis_api -p 8000:8000 `
  -v "${Arts}:/app/artifacts" `
  $Image `
  sh -lc "python -m pip install --no-cache-dir -q fastapi uvicorn[standard] tiktoken sentencepiece prometheus-client; export PYTHONPATH=/app/src; uvicorn api.app:app --host 0.0.0.0 --port 8000"

Start-Sleep -Seconds 2
Write-Host "Healthz:"
try { Invoke-RestMethod http://localhost:8000/healthz | ConvertTo-Json } catch { Write-Host "Health check failed" }

Write-Host ""
Write-Host "[5/5] Launching Streamlit on http://localhost:8501 ..."
docker rm -f crisis_ui 2>$null | Out-Null
docker run -d --name crisis_ui -p 8501:8501 `
  -e API_URL="http://host.docker.internal:8000" `
  $Image `
  sh -lc "python -m pip install -q streamlit requests; streamlit run ui/streamlit_app.py --server.address 0.0.0.0 --server.port 8501"

Write-Host ""
Write-Host "Done. Visit:"
Write-Host "  - API docs:     http://localhost:8000/docs"
Write-Host "  - Streamlit UI: http://localhost:8501"
Write-Host "Artifacts in $Arts"
