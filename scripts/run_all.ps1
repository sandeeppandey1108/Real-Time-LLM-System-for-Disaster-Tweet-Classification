
param([string]$Image = "yourname/crisis-llm:gpu-latest")
Write-Host "Image: $Image" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path ".\data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path ".\data\processed" | Out-Null
New-Item -ItemType Directory -Force -Path ".\artifacts" | Out-Null
Write-Host "`n[1/5] Preparing data..." -ForegroundColor Yellow
docker run --rm -v "${PWD}\data:/app/data" $Image ai-tweets prepare --raw-dir data/raw --out-dir data/processed --val-size 0.2
if ($LASTEXITCODE -ne 0) { Write-Error "Prepare failed."; exit 1 }
Write-Host "`n[2/5] Training..." -ForegroundColor Yellow
docker run --rm --gpus all -v "${PWD}\data:/app/data" -v "${PWD}\artifacts:/app/artifacts" $Image ai-tweets train --config configs/gpu.yaml --train-csv data/processed/train.csv --eval-csv data/processed/val.csv
if ($LASTEXITCODE -ne 0) { Write-Error "Training failed."; exit 1 }
Write-Host "`n[3/5] Stamping labels..." -ForegroundColor Yellow
docker run --rm -v "${PWD}\artifacts:/app/artifacts" $Image python -c "from transformers import AutoConfig; p=r'/app/artifacts/checkpoints/final'; cfg=AutoConfig.from_pretrained(p); cfg.id2label={0:'non_disaster',1:'disaster'}; cfg.label2id={'non_disaster':0,'disaster':1}; cfg.save_pretrained(p); print('labels stamped')"
Write-Host "`n[4/5] Starting API on http://localhost:8000 ..." -ForegroundColor Yellow
docker rm -f crisis_api 2>$null | Out-Null
docker run -d --gpus all --name crisis_api -p 8000:8000 -v "${PWD}\artifacts:/app/artifacts" -e MODEL_DIR="/app/artifacts/checkpoints/final" $Image ai-tweets serve --model-dir /app/artifacts/checkpoints/final --host 0.0.0.0 --port 8000 | Out-Null
Start-Sleep -Seconds 2
Write-Host "Healthz:"; try { Invoke-RestMethod http://localhost:8000/healthz | ConvertTo-Json } catch { Write-Host "   (API not up yet, check: docker logs crisis_api)" -ForegroundColor DarkYellow }
Write-Host "`n[5/5] Launching Streamlit on http://localhost:8501 ..." -ForegroundColor Yellow
docker rm -f crisis_ui 2>$null | Out-Null
docker run -d --gpus all --name crisis_ui -p 8501:8501 -v "${PWD}\artifacts:/app/artifacts" $Image sh -lc "streamlit run src/ai_tweets/streamlit_app.py --server.address=0.0.0.0 --server.port=8501" | Out-Null
Write-Host "`nDone. Visit:" -ForegroundColor Green
Write-Host "  - API docs:     http://localhost:8000/docs"
Write-Host "  - Streamlit UI: http://localhost:8501"
Write-Host "Artifacts in .\artifacts (model, metrics, confusion_matrix.png)"
