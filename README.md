# 🚨 Real‑Time LLM System for Disaster Tweet Classification

[![Docker](https://img.shields.io/badge/Containerized-Docker-informational?logo=docker)](https://www.docker.com/)
[![Transformers](https://img.shields.io/badge/NLP-Transformers-blue?logo=huggingface)](https://huggingface.co/docs/transformers)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![GPU Ready](https://img.shields.io/badge/Accelerated-GPU%20Ready-6C8EBF?logo=nvidia)](https://developer.nvidia.com/)

> Fast, reproducible pipeline to **fine‑tune** a multilingual transformer and **serve** it with **FastAPI** (inference endpoint) and **Streamlit** (demo UI). Built for Windows with Docker; GPU‑first but works on CPU.

---

## ✨ Highlights

- 🔁 **End‑to‑end script:** `run_all.ps1` trains + launches API & UI
- ⚡ **Efficient model:** `microsoft/Multilingual-MiniLM-L12-H384`
- 📦 **Containerized:** single image for train / serve / UI
- 🧰 **Offline‑first:** loads **local** checkpoint (`artifacts/checkpoints/final`) — no Hub calls
- 📊 **Artifacts:** metrics + `confusion_matrix.png` + saved tokenizer/model
- 🧪 **Swagger docs:** `GET /docs` for quick API testing

---

## 🗺️ Architecture (at a glance)

```
+---------------------------+         +-------------------+
|        Training Job       |         |    Streamlit UI   |
|  (inside Docker)          |         |  :8510 (default)  |
|   - Reads data/processed  | <-----> |  mounts artifacts |
|   - Saves artifacts       |         +-------------------+
|     /app/artifacts        |                 ↑
+-------------+-------------+                 |
              |                               
              v                                |
      HOST: artifacts/ (bind mount)            |
              ^                                |
+-------------+-------------+                 |
|         FastAPI API       |                 |
|   :8010 (default)         |  <--------------+
|   loads MODEL_DIR==/app/artifacts/checkpoints/final
+---------------------------+
```

**Containers**
- **Training:** runs via `ai_tweets.cli train` inside the image
- **API:** `uvicorn ai_tweets.api:app` → `/predict`, `/healthz`, Swagger at `/docs`
- **UI:** `streamlit_app.py` → manual testing; loads local **MODEL_DIR**

**Volumes & Ports**
- `artifacts/  ->  /app/artifacts` (bind mount)
- API: `:8010->8000` (configurable) — UI: `:8510->8501` (configurable)

---

## 🚀 Quick Start (Windows, PowerShell)

> Ensure **Docker Desktop** is running. GPU is used if available.

```powershell
# 1) Choose image
$Image = 'sandeep_pandey/crisis-llm:gpu-latest'

# 2) Train + launch everything
powershell -NoProfile -ExecutionPolicy Bypass -File .
un_all.ps1 -Image $Image

# Open:
# - API docs:     http://localhost:8000/docs
# - Streamlit UI: http://localhost:8501
```

Use different ports:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .
un_all.ps1 -Image $Image -ApiPort 8010 -UiPort 8510
```

### Start Only (if artifacts already exist)

```powershell
$Image   = 'sandeep_pandey/crisis-llm:gpu-latest'
$Arts    = "$PWDrtifacts"
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

**API test (PowerShell‑safe JSON):**
```powershell
$ApiPort = 8010
$body = @{ text = "Wildfire near the highway, evacuations underway" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:${ApiPort}/predict" -ContentType "application/json" -Body $body
```

---

## 📁 Project Layout

```
.
├─ data/
│  └─ processed/
│     ├─ train.csv
│     └─ val.csv
├─ artifacts/
│  └─ checkpoints/
│     └─ final/
│        ├─ model.safetensors
│        ├─ config.json
│        ├─ tokenizer.json
│        ├─ tokenizer_config.json
│        ├─ special_tokens_map.json
│        └─ sentencepiece.bpe.model
├─ src/ai_tweets/
│  ├─ api.py
│  ├─ cli.py
│  ├─ streamlit_app.py
│  └─ train.py
├─ run_all.ps1
└─ README.md
```

---

## 🧪 Model & Training

- Base: **microsoft/Multilingual-MiniLM-L12-H384**
- Typical validation: **Acc 88–89%**, **F1 0.85–0.86**
- Config knobs: epochs, batch size, max length, learning rate (see `configs/gpu.yaml`)

**Re‑training command used inside the container**
```bash
python -m ai_tweets.cli train   --config configs/gpu.yaml   --train-csv data/train.csv   --eval-csv  data/val.csv
```

---

## 🛠️ Troubleshooting (curated from real errors)

- **Port already allocated**  
  Another process is using 8000/8501.
  - Fix: stop old containers `docker rm -f crisis-api crisis-ui`
  - Or pick other ports: `-ApiPort 8010 -UiPort 8510`

- **Windows blocks scripts (UnauthorizedAccess/PSSecurityException)**  
  Use:
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File .
un_all.ps1 -Image $Image
  ```

- **`HFValidationError: Repo id must be in the form ... '/app/artifacts/checkpoints/final'`**  
  Ensure **offline/local** loading by setting:
  ```powershell
  -e MODEL_DIR="/app/artifacts/checkpoints/final"
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  ```
  (Already included in commands above.)

- **`Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'`**  
  Caused by an old `accelerate`. We upgrade before training inside the container:
  ```bash
  python -m pip install --upgrade "accelerate>=1.2.1" || true
  ```

- **PowerShell quoting / curl errors**  
  Prefer `Invoke-RestMethod` (shown above) to avoid quoting pitfalls.

- **`Empty reply from server`**  
  The API may still be starting up. Re‑try after 5–10s or check logs:
  ```powershell
  docker logs crisis-api --tail 200
  docker logs crisis-ui  --tail 200
  ```

- **Dataset not found**  
  Put `train.csv` and `val.csv` in `data/processed/` **before** training.

---

## 🔧 Environment Variables

| Variable                | Purpose                                  | Default in examples |
|-------------------------|------------------------------------------|---------------------|
| `MODEL_DIR`             | Path to local checkpoint in container    | `/app/artifacts/checkpoints/final` |
| `HF_HUB_OFFLINE`        | Force offline mode                       | `1`                 |
| `TRANSFORMERS_OFFLINE`  | Force offline transformers/hub           | `1`                 |

---

## 📈 Emoji Changelog (high‑level)

- 🚀 **v1.0**: First end‑to‑end training + serving; offline local model loading
- 🧪 **v1.1**: Improved Windows Quick Start; PowerShell‑safe API examples
- 🛡️ **v1.2**: Hardened error handling & troubleshooting guide
- ⚡ **v1.3**: GPU/CPU auto‑detect; faster training defaults

---

## 📝 License & Acknowledgments

- Built with ❤️ on top of **Hugging Face Transformers**, **FastAPI**, **Streamlit**.
- Model: `microsoft/Multilingual-MiniLM-L12-H384`.
- See `LICENSE` for details.
