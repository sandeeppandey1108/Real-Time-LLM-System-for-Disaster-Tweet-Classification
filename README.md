# Real-Time LLM System for Disaster Tweet Classification — Windows Quick Start

This patch contains:
- **run_all.ps1** — one command to train and launch the API + Streamlit UI
- **src/ai_tweets/streamlit_app.py** — patched to **load your local checkpoint** (no HF Hub lookups)

Your model artifacts are expected at:
`artifacts/checkpoints/final/` (mapped inside the container to `/app/artifacts/checkpoints/final`).

> If you already trained and see files like `model.safetensors`, `config.json`, and `tokenizer.json` in that folder,
> you can skip the training step by running the start-only commands below.

---

## Prerequisites

- **Docker Desktop** (Windows, WSL2 backend recommended)
- **NVIDIA GPU** + drivers (optional but recommended) and **NVIDIA Container Toolkit** for Docker GPU support
- **PowerShell**

> On first run, Windows may block scripts. Use `-ExecutionPolicy Bypass` as shown below.

---

## A) Train + Launch (recommended)

Open **PowerShell** in your repo root (the folder that contains `artifacts/` and `data/processed/`).

```powershell
# 1) Set your image (GPU build)
$Image = 'sandeep_pandey/crisis-llm:gpu-latest'

# 2) Run everything (train + start API & UI)
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image
```

When it finishes, open:
- API docs: **http://localhost:8000/docs**
- Streamlit UI: **http://localhost:8501**

> To use other ports:
> `powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image -ApiPort 8010 -UiPort 8510`

---

## B) Start Only (if you already have a trained checkpoint)

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

Open:
- **http://localhost:${ApiPort}/docs**
- **http://localhost:${UiPort}**

---

## C) Quick API test (PowerShell-safe JSON)

```powershell
$ApiPort = 8010
$body = @{ text = "Wildfire near the highway, evacuations underway" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:${ApiPort}/predict" -ContentType "application/json" -Body $body
```

---

## D) Stop

```powershell
docker rm -f crisis-api crisis-ui
```

---

## Troubleshooting

- **Port already allocated**: choose different ports (e.g., `-ApiPort 8010 -UiPort 8510`) or stop old containers.
- **Empty reply from server** on `/healthz`: wait a few seconds after starting the API, then try again, or check logs:
  ```powershell
  docker logs crisis-api --tail 200
  docker logs crisis-ui  --tail 200
  ```
- **HFValidationError** in Streamlit: this patch’s `streamlit_app.py` forces **offline/local** loading from `MODEL_DIR`.
- **Execution policy blocks script**: use `-ExecutionPolicy Bypass` as shown above.
- **Dataset paths**: put `train.csv` and `val.csv` in `data/processed/`.
