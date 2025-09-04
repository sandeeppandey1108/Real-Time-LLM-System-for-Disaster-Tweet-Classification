
# Real-Time LLM System for Disaster Tweet Classification

A reproducible, Dockerized system that **fine-tunes a multilingual transformer** to classify tweets as **disaster** or **non-disaster**, and serves the model via **FastAPI** and a **Streamlit** UI.

> **Windows-first:** One PowerShell script (`run_all.ps1`) trains and launches everything. Works on GPU (preferred) or CPU.

---

## 🔎 Highlights
- **Model:** `microsoft/Multilingual-MiniLM-L12-H384` fine-tuned with Hugging Face Transformers
- **Data:** `data/processed/train.csv` & `data/processed/val.csv` (two columns: `text`, `labels`)
- **Artifacts:** Saved to `artifacts/checkpoints/final` (model, tokenizer, metrics, confusion matrix)
- **Serving:** FastAPI (`/predict`, `/healthz`, Swagger at `/docs`), Streamlit demo UI
- **Dockerized:** Single image for training + inference, GPU-ready (falls back to CPU)
- **Offline-first:** UI/API load **local** checkpoint from `MODEL_DIR` (no Hub dependency)

---

## 🧭 System Architecture

```
User Browser
   │
   │  HTTP (localhost:8501)
   ▼
+---------------------+
|  Streamlit (UI)     |  ← container: $Image
|  Port: 8501         |
+----------+----------+
           │  REST call (/predict)
           │  HTTP (localhost:8000)
           ▼
+---------------------+
|  FastAPI (API)      |  ← container: $Image
|  Endpoints:         |      /healthz, /predict, /docs
|  Port: 8000         |
+----------+----------+
           │  reads
           ▼
+-------------------------------+
|  Model Artifacts (Volume)     |
|  host: artifacts/checkpoints/ |
|  container: /app/artifacts/   |
+-------------------------------+

Train Job (same image) writes model → artifacts/checkpoints/final
```

**Containers** (all from the same Docker image):
- **Trainer**: runs once to fine-tune and **write** artifacts to the host volume (bind mount)
- **API (FastAPI)**: loads local model from volume, exposes `/predict` & `/healthz`
- **UI (Streamlit)**: loads local model from volume for manual testing

**Volumes**:
- `artifacts/` on host ↔ `/app/artifacts/` in container (bidirectional)

**Default Ports**:
- API: 8000 (configurable via `-ApiPort`)
- UI:  8501 (configurable via `-UiPort`)

---

## 🚀 Quick Start (Windows)

> Prereqs: **Docker Desktop** (WSL2 backend recommended). GPU optional.

### A) Train + Launch (one command)
From the repo root (the folder with `artifacts/` and `data/processed/`):
```powershell
$Image = 'sandeep_pandey/crisis-llm:gpu-latest'
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image
```
Open:
- API docs → http://localhost:8000/docs
- UI → http://localhost:8501

> To change ports:
> ```powershell
> powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image -ApiPort 8010 -UiPort 8510
> ```

### B) Start Only (if you already have a trained model)
```powershell
$Image   = 'sandeep_pandey/crisis-llm:gpu-latest'
$Arts    = "$PWD\artifacts"
$ApiPort = 8010
$UiPort  = 8510

docker rm -f crisis-api crisis-ui 2>$null | Out-Null

# API
docker run -d --rm --name crisis-api --gpus all `
  -p ${ApiPort}:8000 `
  -v "${Arts}:/app/artifacts" `
  -e MODEL_DIR="/app/artifacts/checkpoints/final" `
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 `
  $Image uvicorn ai_tweets.api:app --host 0.0.0.0 --port 8000 --log-level info

# UI
docker run -d --rm --name crisis-ui --gpus all `
  -p ${UiPort}:8501 `
  -v "${Arts}:/app/artifacts" `
  -e MODEL_DIR="/app/artifacts/checkpoints/final" `
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 `
  $Image streamlit run src/ai_tweets/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
```

---

## 📡 API Reference

- **`GET /healthz`** → `{ "ok": true }` when ready
- **`POST /predict`** (JSON):
  ```json
  { "text": "Wildfire near the highway, evacuations underway" }
  ```
  **Response** (example):
  ```json
  {
    "label": "disaster",
    "score": 0.94,
    "probs": [
      {"label": "non_disaster", "score": 0.06},
      {"label": "disaster", "score": 0.94}
    ]
  }
  ```

**PowerShell-safe test:**
```powershell
$ApiPort = 8000
$body = @{ text = "Wildfire near the highway, evacuations underway" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:${ApiPort}/predict" -ContentType "application/json" -Body $body
```

---

## 🧪 Training & Artifacts

- Input CSVs (two columns): `data/processed/train.csv`, `data/processed/val.csv`
- Output artifacts: `artifacts/checkpoints/final/`
  - `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `sentencepiece.bpe.model`
  - `metrics.json`, `confusion_matrix.png`

Typical validation results:
- **Accuracy**: ~88–89%
- **F1**: ~0.85–0.86

---

## ⚙️ Configuration (env vars)
| Variable | Purpose | Default |
|---|---|---|
| `MODEL_DIR` | Where UI/API read the local model | `/app/artifacts/checkpoints/final` |
| `HF_HUB_OFFLINE` | Force offline mode | `1` |
| `TRANSFORMERS_OFFLINE` | Disable hub lookups | `1` |

---

## 🛠️ Troubleshooting (real-world fixes)

### 1) **Port already allocated**
**Symptom:** Docker says `Bind for 0.0.0.0:8000 (or 8501) failed`.  
**Fix:** Stop old containers or use different ports:
```powershell
docker rm -f crisis-api crisis-ui
# or run with -ApiPort/-UiPort
```

### 2) **Empty reply from server** on `/healthz`
**Symptom:** `curl: (52) Empty reply from server` right after starting the API.  
**Fix:** Wait a few seconds (model loading), then:
```powershell
docker logs crisis-api --tail 200
```

### 3) **HFValidationError** in Streamlit
**Symptom:** Streamlit tries to treat a folder path as a Hub repo id.  
**Fix:** Ensure `MODEL_DIR` points to the artifacts **directory** and set offline flags:
```powershell
-e MODEL_DIR="/app/artifacts/checkpoints/final" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
```

### 4) **PowerShell execution policy blocked**
**Symptom:** `run_all.ps1 is not digitally signed`.  
**Fix:** run with `-ExecutionPolicy Bypass` as shown in Quick Start.

### 5) **Accelerate/Trainer mismatch**
**Symptom:** `Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'`.  
**Fix:** The container’s training step upgrades `accelerate` internally to a compatible version before training.

### 6) **Dataset not found / wrong columns**
**Symptom:** Train step exits early or crashes.  
**Fix:** Put CSVs under `data/processed/` with **columns**: `text`, `labels` (0/1).

### 7) **GPU not available**
The image runs on CPU automatically (slower). Remove `--gpus all` if you don’t have NVIDIA runtime.

---

## 🔐 Security & Privacy
- Everything runs **locally** by default. No tweets or predictions are sent to third parties.
- Offline model loading avoids incidental calls to external model hubs.

---

## 📈 Next Steps
- Expand/refresh training data across event types and languages
- Add drift monitoring & scheduled re-training
- Quantization/distillation for CPU-only latency
- CI/CD for container builds and smoke tests on `/healthz` & `/predict`

---

## 🙌 Acknowledgments
- Hugging Face Transformers & Datasets
- FastAPI, Uvicorn
- Streamlit
- Docker

---

**Author:** Sandeep Pandey  
**Repo:** https://github.com/sandeeppandey1108/Real-Time-LLM-System-for-Disaster-Tweet-Classification
