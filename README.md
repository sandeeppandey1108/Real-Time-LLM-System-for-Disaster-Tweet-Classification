# 🚨 Real‑Time LLM System for Disaster Tweet Classification (Fake vs Real)

[![Docker](https://img.shields.io/badge/Containerized-Docker-informational?logo=docker)](https://www.docker.com/)
[![Transformers](https://img.shields.io/badge/NLP-Transformers-blue?logo=huggingface)](https://huggingface.co/docs/transformers)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![GPU Ready](https://img.shields.io/badge/Accelerated-GPU%20Ready-6C8EBF?logo=nvidia)](https://developer.nvidia.com/)

A fast, reproducible pipeline to **fine‑tune** a multilingual transformer on the Kaggle *Disaster Tweets* dataset and **serve** it with **FastAPI** (inference endpoint) and **Streamlit** (demo UI).  
Built and tested on **Windows + PowerShell + Docker**. Works on **GPU** (recommended) and CPU.

> This repo uses the **binary** labels **fake (0)** vs **real (1)**.  
> We intentionally **do not** use an “unrelated” class.

---

## ✨ What you get

- 🔁 **End‑to‑end script**: one PowerShell to *prepare data → train → launch API & UI*
- ⚡ **Efficient model**: `microsoft/Multilingual-MiniLM-L12-H384`
- 📦 **Single Docker image** for training + serving + UI
- 🧰 **Offline‑friendly loading** from `artifacts/checkpoints/final`
- 📊 **Artifacts**: metrics JSON + `confusion_matrix_bin.png`
- 🧪 **Swagger** docs at `/docs` and a Streamlit UI

---

## 🗺️ Architecture

```
+----------------------------+        +-------------------+
|        Training Job        |        |    Streamlit UI   |
|  docker run ... train.py   |<------>|  :8501            |
|  reads data/processed      |        |  mounts artifacts |
|  writes artifacts/         |        +-------------------+
|   checkpoints/final        |                 ^
+--------------+-------------+                 |
               |                               
               v                                |
      HOST: artifacts/ (bind mount)             |
               ^                                |
+--------------+-------------+                 |
|          FastAPI API       |                 |
|  uvicorn ai_bin.api:app    |<----------------+
|  :8000 (/predict, /docs)   |
+----------------------------+
```

---

## 📦 Project layout

```
.
├─ configs/
│  ├─ bin.yaml        # 3-epoch quick config
│  └─ bin_es.yaml     # 10-epoch recommended config (no early stopping)
├─ data/
│  ├─ raw/            # put Kaggle train.csv here (optional)
│  └─ processed/
│     ├─ train.csv
│     └─ val.csv
├─ artifacts/         # created by training
│  ├─ confusion_matrix_bin.png
│  └─ checkpoints/final/  # model + tokenizer
├─ src/ai_bin/
│  ├─ cli.py          # scripts: prepare_bin
│  ├─ train.py        # binary training
│  ├─ api.py          # FastAPI server
│  └─ streamlit_app.py
└─ scripts/
   └─ run_all_bin_mounted.ps1
```

---

## ✅ Prerequisites

1. **Docker Desktop** (Windows)
2. Optional **NVIDIA GPU** + drivers + CUDA runtime. Verify from inside containers with:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
3. Dataset (Kaggle: *nlp-getting-started*). You need **train.csv**; we’ll create a val split.

---

## 🚀 Quick Start (Windows PowerShell)

> These commands assume you’re in the repo root.

### 0) Choose the image
```powershell
$Image = 'sandeep_pandey/crisis-llm:gpu-latest'
```

### 1) Prepare data (binary: fake vs real)

If you already have `data/processed/train.csv` and `val.csv`, skip this.

```powershell
# Put Kaggle train.csv in data/raw/ OR point --train to your CSV.
docker run --rm `
  -e PYTHONPATH="/app/src" `
  -v "$PWD\data:/app/data" `
  -v "$PWD\src\ai_bin:/app/src/ai_bin" `
  -v "$PWD\configs:/app/configs" `
  $Image bash -lc `
  "python -m ai_bin.cli prepare_bin --train data/raw/train.csv `
     --out-train data/processed/train.csv --out-val data/processed/val.csv `
     --use-keyword-location --lowercase"
```

### 2) Train (recommended config: `configs/bin_es.yaml` → 10 epochs, lr=3e‑5)

```powershell
docker run --rm --gpus all `
  -e PYTHONPATH="/app/src" `
  -v "$PWD\artifacts:/app/artifacts" -v "$PWD\data:/app/data" `
  -v "$PWD\src\ai_bin:/app/src/ai_bin" -v "$PWD\configs:/app/configs" `
  $Image bash -lc `
  "python -m ai_bin.train --config configs/bin_es.yaml `
     --train-csv data/processed/train.csv `
     --eval-csv  data/processed/val.csv `
     --out-dir   artifacts"
```

**What you should see at the end**
```json
{
  "metrics": {
    "eval_accuracy": ~0.76,
    "eval_precision": ~0.68,
    "eval_recall": ~0.83,
    "eval_f1": ~0.75
  },
  "confusion_matrix": "artifacts/confusion_matrix_bin.png",
  "model_dir": "artifacts/checkpoints/final",
  "num_labels": 2
}
```

### 3) Serve the model (API + UI)

```powershell
docker network create crisis-net 2>$null | Out-Null
docker rm -f crisis-api crisis-ui 2>$null | Out-Null

# API (FastAPI)
docker run -d --name crisis-api --network crisis-net --gpus all `
  -e PYTHONPATH="/app/src" `
  -p 8000:8000 `
  -v "$PWD\artifacts:/app/artifacts" -v "$PWD\src\ai_bin:/app/src/ai_bin" `
  -e MODEL_DIR="/app/artifacts/checkpoints/final" `
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 `
  $Image uvicorn ai_bin.api:app --host 0.0.0.0 --port 8000 --log-level info

# UI (Streamlit)
docker run -d --name crisis-ui --network crisis-net `
  -e PYTHONPATH="/app/src" `
  -p 8501:8501 `
  -v "$PWD\artifacts:/app/artifacts" -v "$PWD\src\ai_bin:/app/src/ai_bin" `
  -e MODEL_DIR="/app/artifacts/checkpoints/final" -e API_URL="http://crisis-api:8000" `
  $Image streamlit run src/ai_bin/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
```

Open:
- **API docs**: http://localhost:8000/docs  
- **UI**: http://localhost:8501

### 4) Test the API
```powershell
$body = @{ text = "Wildfire near the highway, evacuations underway" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body
```

Example response:
```json
{"label":"real","score":0.92}
```

---

## ⚙️ Configs (recommended)

`configs/bin_es.yaml` (used in your successful run):
```yaml
seed: 42
model_name: microsoft/Multilingual-MiniLM-L12-H384
max_length: 128
batch_size: 16
epochs: 10
learning_rate: 3.0e-5
weight_decay: 0.01
warmup_ratio: 0.06
monitor_metric: f1
greater_is_better: true
# early_stopping: intentionally disabled (keeps it simple + consistent)
```

**Why 10 epochs / 3e‑5?**  
Stable fine‑tuning for MiniLM, matches your throughput (~7.6 steps/s) and yields good F1 on validation without overfitting.

---

## 🧪 Ground rules & labels

- **Binary** classification only:  
  - `fake` (0) — tweet is **not** truly reporting a disaster (sarcasm, ads, figurative use)  
  - `real` (1) — tweet **is** about an actual disaster/emergency  
- We optionally enrich text with **keyword** and **location** during preprocessing (`--use-keyword-location`).

---

## 🔍 Troubleshooting (from real errors you hit)

### Networking & ports
- **Port already allocated** (8000/8501):  
  ```powershell
  docker rm -f crisis-api crisis-ui
  ```
  Or use different ports and map them: `-p 8010:8000` and `-p 8510:8501`.

- **`Invoke-WebRequest` fails or closes connection** (proxy on Windows):  
  ```powershell
  Invoke-WebRequest "http://127.0.0.1:8000/docs" -Proxy $null
  # or use Windows curl.exe
  curl.exe -v http://127.0.0.1:8000/docs
  ```

- **Container logs**:
  ```powershell
  docker logs crisis-api --tail 200
  docker logs crisis-ui  --tail 200
  ```

### Paths, quoting & PowerShell gotchas
- **Backslash vs forward slash in `-Config`**: both work, but keep the literal path inside the container:
  ```powershell
  --config configs/bin_es.yaml
  ```

- **`invalid spec: :/app/data: empty section between colons`**  
  You referenced a variable that wasn’t set (e.g., `$Data`). Always bind‑mount with a literal path or `$PWD`:
  ```powershell
  -v "$PWD\data:/app/data"
  ```

- **Heredoc/array with `-v "$Var:/app/..."` shows `':' not followed by variable name`**  
  In PowerShell, wrap the var with `${}` if the next char is `:`:
  ```powershell
  "-v `${Arts}:/app/artifacts`"
  ```

- **Script not found**  
  ```text
  The argument '.\..\run_all_mounted.ps1' to the -File parameter does not exist.
  ```
  Fix the relative path or run from repo root:
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all_bin_mounted.ps1 -Full -ProjectRoot .
  ```

### Training & Transformers
- **`TrainingArguments.__init__() got an unexpected keyword 'evaluation_strategy'`**  
  This came from an older tri‑class script using mismatched Transformers. We use the **binary** path (`ai_bin`) which is compatible.

- **`--load_best_model_at_end requires save and eval strategy to match`**  
  Ensure both strategies are aligned (our configs use consistent settings).

- **Why did I see ~381000 steps?**  
  You asked for **1000 epochs** earlier; with ~381 steps/epoch, that’s `381,000` total. We now recommend **10 epochs** (≈3,810 steps total).

- **`Learning rate 0.001 is very high` warning**  
  For transformers, use **3e‑5** (we set this in `bin_es.yaml`).

### GPU / CUDA
- Check from inside your API container:
  ```powershell
  docker exec -it crisis-api bash -lc "python - << 'PY'\nimport torch;print(torch.__version__, torch.cuda.is_available());\nprint(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\nPY"
  ```

---

## 🧰 API reference

- `POST /predict`  
  Body: `{"text": "tweet text here"}`  
  Returns: `{"label": "fake|real", "score": float}`

- `GET /healthz` → `{"status":"ok"}`  
- `GET /docs` → Swagger UI

---

## 🧭 Repro: your successful run (for reference)

- Config: `configs/bin_es.yaml` (10 epochs, lr 3e‑5, batch 16)
- Validation metrics:
  - **Accuracy** ≈ 0.759
  - **Precision** ≈ 0.679
  - **Recall** ≈ 0.832
  - **F1** ≈ 0.748
- Confusion matrix saved to `artifacts/confusion_matrix_bin.png`

![Confusion Matrix](artifacts/confusion_matrix_bin.png)

---

## 📌 Tips & Next steps

- Try **longer max_length** (e.g., 192) if your GPU has room.
- Add text normalization for URLs/handles/hashtags during preprocessing.
- Log inference to spot hard examples and extend your training set.

---

## 📝 License & Acknowledgments

- Dataset: Kaggle *nlp-getting-started* (Figure Eight).
- Model: `microsoft/Multilingual-MiniLM-L12-H384` (Hugging Face).
- Frameworks: Transformers, FastAPI, Streamlit.
- MIT‑style licensing recommended (adapt to your needs).
