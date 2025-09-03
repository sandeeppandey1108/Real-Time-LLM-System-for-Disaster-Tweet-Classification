param(
  [string]$Image = "sandeep_pandey/crisis-llm:gpu-latest",
  [double]$ValSplit = 0.2
)

Write-Host "Image: $Image`n" -ForegroundColor Cyan

# Ensure folders
$pwdPath = (Get-Location).Path
New-Item -ItemType Directory -Force -Path "$pwdPath\data" | Out-Null
New-Item -ItemType Directory -Force -Path "$pwdPath\artifacts" | Out-Null

# 1) Prepare data (merge + clean + split)
Write-Host "[1/5] Preparing data..." -ForegroundColor Yellow
docker run --rm `
  -e TEST_SIZE=$ValSplit `
  -v "${pwdPath}\data:/app/data" `
  -v "${pwdPath}\artifacts:/app/artifacts" `
  $Image `
  python tools/prep_data.py

if ($LASTEXITCODE -ne 0) { Write-Host "Data prep failed." -ForegroundColor Red; exit 1 }

# 2) Train
Write-Host "`n[2/5] Training..." -ForegroundColor Yellow
docker run --rm --gpus all `
  -v "${pwdPath}\data:/app/data" `
  -v "${pwdPath}\artifacts:/app/artifacts" `
  $Image `
  python src/train.py --config configs/gpu.yaml --train-csv data/processed/train.csv --eval-csv data/processed/val.csv

if ($LASTEXITCODE -ne 0) { Write-Host "Training failed." -ForegroundColor Red; exit 2 }

# 3) Stamp human-readable labels
Write-Host "`n[3/5] Stamping labels..." -ForegroundColor Yellow
docker run --rm `
  -v "${pwdPath}\artifacts:/app/artifacts" `
  $Image `
  python - <<'PY'
from transformers import AutoConfig
p='/app/artifacts/checkpoints/final'
cfg=AutoConfig.from_pretrained(p)
cfg.id2label={0:'non_disaster',1:'disaster'}
cfg.label2id={'non_disaster':0,'disaster':1}
cfg.save_pretrained(p)
print('labels saved')
PY

# 4) Start API
Write-Host "`n[4/5] Starting API on http://localhost:8000 ..." -ForegroundColor Yellow
docker rm -f crisis_api 2>$null | Out-Null
docker run -d --gpus all --name crisis_api `
  -p 8000:8000 `
  -e HF_HOME=/root/.cache/huggingface `
  -v "${pwdPath}\artifacts:/app/artifacts" `
  $Image `
  sh -lc "python -m uvicorn api.app:app --host 0.0.0.0 --port 8000" | Out-Null

Start-Sleep -Seconds 3
Write-Host "Healthz:" -NoNewline
try {
  $resp = Invoke-RestMethod http://localhost:8000/healthz -TimeoutSec 5
  Write-Host ""
  $resp | ConvertTo-Json
} catch {
  Write-Host "`nAPI not ready yet. Check logs with: docker logs -f crisis_api" -ForegroundColor DarkYellow
}

# 5) Start Streamlit UI
Write-Host "`n[5/5] Launching Streamlit on http://localhost:8501 ..." -ForegroundColor Yellow
docker rm -f crisis_ui 2>$null | Out-Null
docker run -d --name crisis_ui `
  -p 8501:8501 `
  -e API_BASE="http://host.docker.internal:8000" `
  -v "${pwdPath}\artifacts:/app/artifacts" `
  $Image `
  streamlit run ui/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 | Out-Null

Write-Host "`nDone. Visit:" -ForegroundColor Green
Write-Host "  - API docs:     http://localhost:8000/docs"
Write-Host "  - Streamlit UI: http://localhost:8501"
Write-Host "Artifacts in .\artifacts (model, metrics, confusion_matrix.png)`n"
