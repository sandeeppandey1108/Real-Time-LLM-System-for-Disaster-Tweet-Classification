
# Real-Time LLM System for Disaster Tweet Classification

End-to-end setup for fine-tuning a lightweight transformer to classify disaster-related tweets, serving a FastAPI endpoint and a Streamlit demo.
Works fully **offline** after training (loads the local checkpoint).

## Table of Contents

* [Prerequisites](#prerequisites)
* [Folder Layout](#folder-layout)
* [Quick Start (Windows PowerShell)](#quick-start-windows-powershell)
* [Quick Start (Linux/macOS Bash)](#quick-start-linuxmacos-bash)
* [API Usage](#api-usage)
* [Streamlit UI](#streamlit-ui)
* [Retraining / Custom Data](#retraining--custom-data)
* [Troubleshooting](#troubleshooting)
* [FAQ](#faq)

---

## Prerequisites

* **Docker Desktop** installed and running

  * For GPU: NVIDIA driver + NVIDIA Container Toolkit (Docker Desktop → Settings → Resources → GPU enabled).
* **Internet access** only for the first run (to pull the Docker image and base model).
  After training, the model is used completely **offline**.

---

## Folder Layout

```
Real-Time-LLM-System-for-Disaster-Tweet-Classification/
├─ data/
│  └─ processed/
│     ├─ train.csv         # training data (text,labels)
│     └─ val.csv           # validation data (text,labels)
├─ artifacts/              # created automatically
│  └─ checkpoints/
│     └─ final/            # trained checkpoint (config.json, model.safetensors, tokenizer.json, ...)
├─ src/
│  └─ ai_tweets/
│     ├─ api.py            # FastAPI app
│     ├─ cli.py            # Typer CLI to train/evaluate
│     └─ streamlit_app.py  # Streamlit demo (loads local checkpoint)
├─ configs/
│  └─ gpu.yaml             # training hyperparameters
└─ run_all.ps1             # one-shot helper (optional)
```

---

## Quick Start (Windows PowerShell)

> Open **PowerShell** and run the commands **from the repo root**:
> `C:\Users\sande\Music\disaster-tweets-llm-pro\Real-Time-LLM-System-for-Disaster-Tweet-Classification`

### 0) Variables

```powershell
$Image   = 'sandeep_pandey/crisis-llm:gpu-latest'
$Proc    = "$PWD\data\processed"      # train.csv / val.csv live here
$Arts    = "$PWD\artifacts"           # model artifacts live here
$ApiPort = 8010
$UiPort  = 8510

mkdir $Arts -Force | Out-Null
```

### 1) Train (or re-train)

```powershell
docker run --rm --gpus all `
  -e TRANSFORMERS_VERBOSITY=info `
  -e HF_HUB_DISABLE_TELEMETRY=1 `
  -v "${Proc}:/app/data" `
  -v "${Arts}:/app/artifacts" `
  $Image sh -lc "export PYTHONPATH=/app/src; python -u -m ai_tweets.cli train --config configs/gpu.yaml --train-csv data/train.csv --eval-csv data/val.csv"
```

Verify the checkpoint:

```powershell
docker run --rm -v "${Arts}:/app/artifacts" $Image sh -lc "ls -1 /app/artifacts/checkpoints/final/{config.json,model.safetensors,tokenizer.json,special_tokens_map.json,tokenizer_config.json,sentencepiece.bpe.model}"
```

### 2) Serve the API (GPU)

```powershell
docker rm -f crisis-api crisis-ui 2>$null | Out-Null

docker run -d --rm --name crisis-api --gpus all `
  -p ${ApiPort}:8000 `
  -v "${Arts}:/app/artifacts" `
  -e MODEL_DIR="/app/artifacts/checkpoints/final" `
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 `
  $Image uvicorn ai_tweets.api:app --host 0.0.0.0 --port 8000 --log-level info
```

Wait for health:

```powershell
$deadline = (Get-Date).AddMinutes(2)
$healthy = $false
while(-not $healthy -and (Get-Date) -lt $deadline){
  try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/healthz" -TimeoutSec 2
    if($r.ok){ $healthy = $true; break }
  } catch { Start-Sleep -Milliseconds 500 }
}
if(-not $healthy){ Write-Warning "API didn't become healthy. See logs below."; docker logs crisis-api --tail 200 }
```

Open the docs:

```powershell
Start-Process "http://localhost:$ApiPort/docs"
```

### 3) Serve the Streamlit UI (GPU)

```powershell
docker run -d --rm --name crisis-ui --gpus all `
  -p ${UiPort}:8501 `
  -v "${Arts}:/app/artifacts" `
  -e MODEL_DIR="/app/artifacts/checkpoints/final" `
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 `
  $Image streamlit run src/ai_tweets/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
```

Open the UI:

```powershell
Start-Process "http://localhost:$UiPort"
```

### 4) Stop

```powershell
docker rm -f crisis-api crisis-ui
```

---

## Quick Start (Linux/macOS Bash)

```bash
cd /path/to/Real-Time-LLM-System-for-Disaster-Tweet-Classification

IMAGE='sandeep_pandey/crisis-llm:gpu-latest'
PROC="$PWD/data/processed"
ARTS="$PWD/artifacts"
APIPORT=8010
UIPORT=8510
mkdir -p "$ARTS"

# Train
docker run --rm --gpus all \
  -e TRANSFORMERS_VERBOSITY=info \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  -v "$PROC:/app/data" \
  -v "$ARTS:/app/artifacts" \
  "$IMAGE" sh -lc "export PYTHONPATH=/app/src; python -u -m ai_tweets.cli train --config configs/gpu.yaml --train-csv data/train.csv --eval-csv data/val.csv"

# API
docker rm -f crisis-api crisis-ui >/dev/null 2>&1 || true
docker run -d --rm --name crisis-api --gpus all \
  -p ${APIPORT}:8000 \
  -v "$ARTS:/app/artifacts" \
  -e MODEL_DIR="/app/artifacts/checkpoints/final" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "$IMAGE" uvicorn ai_tweets.api:app --host 0.0.0.0 --port 8000 --log-level info

# Health
curl -fsS "http://127.0.0.1:${APIPORT}/healthz" || docker logs crisis-api --tail 200

# Streamlit
docker run -d --rm --name crisis-ui --gpus all \
  -p ${UIPORT}:8501 \
  -v "$ARTS:/app/artifacts" \
  -e MODEL_DIR="/app/artifacts/checkpoints/final" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "$IMAGE" streamlit run src/ai_tweets/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
```

---

## API Usage

* Docs: `http://localhost:<APIPORT>/docs` (FastAPI Swagger)
* Health: `GET /healthz` → `{ "ok": true }` when ready
* Predict: `POST /predict`
  Body:

  ```json
  { "text": "Wildfire near the highway, evacuations underway" }
  ```

  Example (Windows PowerShell):

  ```powershell
  Invoke-RestMethod `
    -Uri "http://127.0.0.1:$ApiPort/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body (@{ text = "Wildfire near the highway, evacuations underway" } | ConvertTo-Json)
  ```

---

## Streamlit UI

* Open `http://localhost:<UIPORT>`
* Paste a tweet and click **Predict**.
  The app loads the checkpoint from `artifacts/checkpoints/final` (offline).

---

## Retraining / Custom Data

* Put your CSVs in `data/processed/train.csv` and `data/processed/val.csv`

  * Required columns: `text`, `labels` (0 = non\_disaster, 1 = disaster)
* Re-run the **Train** command.
  The final checkpoint is written to `artifacts/checkpoints/final`.

---

## Troubleshooting

**“Empty reply from server” on `/healthz`**
The API is still loading the model. Use the wait loop above and/or check:

```powershell
docker logs crisis-api --tail 200
```

**HFValidationError about repo id in Streamlit**
Your `streamlit_app.py` is already updated to load from a local folder. Make sure the container has:

```
-e MODEL_DIR="/app/artifacts/checkpoints/final"
-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
-v "<repo>\artifacts:/app/artifacts"
```

**Port is already allocated (8000/8501)**
Something is already listening. Stop leftovers:

```powershell
docker rm -f crisis-api crisis-ui
```

Or use different host ports (e.g., 8010/8510).

**PowerShell script not signed**
Run with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image 'sandeep_pandey/crisis-llm:gpu-latest'
```

**GPU not used**
Ensure Docker Desktop → Settings → Resources → GPU is enabled, and your NVIDIA drivers are installed.

---

## FAQ

**Q: Can I run without GPU?**
A: Yes—remove `--gpus all` from the `docker run` commands. It will be slower.

**Q: Where is the model saved?**
A: `artifacts/checkpoints/final` (mounted inside the container at `/app/artifacts/checkpoints/final`).

**Q: How do I change batch size/epochs?**
A: Edit `configs/gpu.yaml`, then re-run training.

**Q: One-shot helper script?**
A: You can use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image 'sandeep_pandey/crisis-llm:gpu-latest'
```

Optional switches: `-StartApi:$false` and/or `-StartUI:$false`.

---

**That’s it!**
Train once, then serve the API and UI with the commands above.
