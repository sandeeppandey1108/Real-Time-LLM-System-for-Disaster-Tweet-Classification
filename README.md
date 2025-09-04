# Real-Time LLM System for **Disaster Tweet Classification**

### Windows-first README • Quick Start • Common Problems & Fixes

This project fine-tunes a multilingual MiniLM/BERT classifier on disaster tweets, then serves it via:

* **FastAPI** (inference REST API)
* **Streamlit** (simple web UI)
* **Docker** (GPU-ready, Windows-friendly)

It’s set up to run **entirely with Docker**, no local Python environment needed.

---

## What you need

* **Docker Desktop** (Windows, WSL2 backend recommended)
* **(Optional) NVIDIA GPU** + NVIDIA drivers + **NVIDIA Container Toolkit**
  If you have a GPU, containers will run with `--gpus all`. Without a GPU, they’ll still run on CPU.
* **PowerShell**

> If Windows blocks scripts, you’ll use `-ExecutionPolicy Bypass` once (shown below).

---

## Folder layout (important)

```
repo-root/
├─ data/
│  └─ processed/
│     ├─ train.csv         # your training data
│     └─ val.csv           # your validation data
├─ artifacts/
│  └─ checkpoints/
│     └─ final/            # model files will appear here after training
│        ├─ model.safetensors
│        ├─ config.json
│        ├─ tokenizer.json
│        ├─ special_tokens_map.json
│        ├─ tokenizer_config.json
│        └─ sentencepiece.bpe.model
├─ src/ai_tweets/
│  ├─ api.py               # FastAPI app
│  └─ streamlit_app.py     # Streamlit UI (supports local/offline model)
└─ run_all.ps1             # one-command runner (train + serve)
```

Put your **`train.csv`** and **`val.csv`** inside `data/processed/`.

---

## Quick Start (one command)

Open **PowerShell in the repo root** (the folder that contains `artifacts/` and `data/`).

```powershell
# Choose the Docker image (GPU build)
$Image = 'sandeep_pandey/crisis-llm:gpu-latest'

# Train + start API & Streamlit
# (If Windows blocks scripts, this bypass is safe for your local machine.)
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image
```

When it finishes, open:

* **API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Streamlit UI:** [http://localhost:8501](http://localhost:8501)

Prefer different ports? Add:
`-ApiPort 8010 -UiPort 8510`
Example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image -ApiPort 8010 -UiPort 8510
```

---

## Start Only (skip training if model already exists)

If you already see model files under `artifacts\checkpoints\final\`:

```powershell
$Image   = 'sandeep_pandey/crisis-llm:gpu-latest'
$Arts    = "$PWD\artifacts"
$ApiPort = 8010
$UiPort  = 8510

# Stop any leftovers (ignore errors)
docker rm -f crisis-api crisis-ui 2>$null | Out-Null

# --- API ---
docker run -d --rm --name crisis-api --gpus all `
  -p ${ApiPort}:8000 `
  -v "${Arts}:/app/artifacts" `
  -e MODEL_DIR="/app/artifacts/checkpoints/final" `
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 `
  $Image uvicorn ai_tweets.api:app --host 0.0.0.0 --port 8000 --log-level info

# --- Streamlit UI ---
docker run -d --rm --name crisis-ui --gpus all `
  -p ${UiPort}:8501 `
  -v "${Arts}:/app/artifacts" `
  -e MODEL_DIR="/app/artifacts/checkpoints/final" `
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 `
  $Image streamlit run src/ai_tweets/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
```

Open:

* API docs: `http://localhost:${ApiPort}/docs`
* UI:       `http://localhost:${UiPort}`

---

## Quick API test

Use PowerShell’s `Invoke-RestMethod` (safer than the `curl` alias):

```powershell
$ApiPort = 8010
$body = @{ text = "Wildfire near the highway, evacuations underway" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:${ApiPort}/predict" -ContentType "application/json" -Body $body
```

---

## Stop everything

```powershell
docker rm -f crisis-api crisis-ui
```

---

## Troubleshooting (errors & solutions)

### 1) **Script blocked / not digitally signed**

**Error:** `... cannot be loaded. The file ... run_all.ps1 is not digitally signed.`
**Fix:** run with execution policy bypass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image
```

---

### 2) **Port already allocated** (8000 or 8501)

**Error:** `Bind for 0.0.0.0:8000 failed: port is already allocated`
**Fix:** Either stop previous containers or use different ports:

```powershell
docker rm -f crisis-api crisis-ui
# or
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image -ApiPort 8010 -UiPort 8510
```

---

### 3) **Container name already in use** (`crisis-api` or `crisis-ui`)

**Error:** `Conflict. The container name "/crisis-ui" is already in use ...`
**Fix:** Remove the old one or pick a different name:

```powershell
docker rm -f crisis-ui
# then re-run your docker run command
```

---

### 4) **PowerShell path variables in `-v` mounts**

**Error:** Parse issues like `Variable reference is not valid. ':' was not followed by a valid variable name ...`
**Cause:** Windows drive letters contain `:`, so you must wrap variables in **braces**.
**Fix:** Always write:

```powershell
-v "${Arts}:/app/artifacts"        # ✅ correct
-v "$Arts:/app/artifacts"          # ❌ can break
```

---

### 5) **Empty reply from server** on `/healthz`

**Error:** `curl: (52) Empty reply from server`
**Fix:**

1. Give the API a few seconds after startup.
2. Check logs:

   ```powershell
   docker logs crisis-api --tail 200
   ```
3. Use `curl.exe` or `Invoke-RestMethod` (PowerShell’s `curl` is an alias and can behave differently):

   ```powershell
   curl.exe http://127.0.0.1:8010/healthz
   ```

---

### 6) **HFValidationError** from Streamlit (repo id format)

**Error:**
`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/app/artifacts/checkpoints/final'`

**Cause:** The pipeline was trying to fetch from the Hugging Face Hub instead of loading **local files**.

**Fix (already applied in this repo):**

* The UI reads `MODEL_DIR=/app/artifacts/checkpoints/final`
* We set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
* `streamlit_app.py` uses `local_files_only=True`

If you still see this error (e.g., you’re using an older file), you can **force a local/offline patch**:

```powershell
New-Item -ItemType Directory _patched -Force | Out-Null
@'
import os, torch, streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/artifacts/checkpoints/final")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

@st.cache_resource
def load_pipeline():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    device = 0 if torch.cuda.is_available() else -1
    return TextClassificationPipeline(model=mdl, tokenizer=tok, device=device, truncation=True, return_all_scores=True)

st.title("Disaster Tweet Classifier")
st.caption("Fast demo using your fine-tuned checkpoint (local/offline)")

clf = load_pipeline()
text = st.text_area("Tweet text", "")
if st.button("Predict") and text.strip():
    st.write(clf(text)[0])
'@ | Set-Content -Encoding utf8 _patched\streamlit_app.py

# Run the UI with the patch mounted
$Image   = 'sandeep_pandey/crisis-llm:gpu-latest'
$Arts    = "$PWD\artifacts"
$UiPort  = 8510
docker rm -f crisis-ui 2>$null | Out-Null
docker run -d --rm --name crisis-ui --gpus all `
  -p ${UiPort}:8501 `
  -v "${Arts}:/app/artifacts" `
  -v "$PWD\_patched\streamlit_app.py:/app/src/ai_tweets/streamlit_app.py:ro" `
  -e MODEL_DIR="/app/artifacts/checkpoints/final" `
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 `
  $Image streamlit run src/ai_tweets/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
```

---

### 7) **Accelerate / Transformers mismatch** during training

**Symptom:** Error mentioning `Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'`.

**Fix:** Ensure a recent `accelerate` inside the container before training. The training step in our scripts runs:

```bash
python -m pip install -q --upgrade "accelerate>=1.2.1" || true
```

If you train manually, add that line before calling the trainer.

---

### 8) **DNS / network retries** while downloading (xethub / HF)

If you see transient `dns error` warnings while models are downloaded, try again or ensure a stable connection.
Once your model is trained, the **UI/API use local files** and do not need Internet.

---

### 9) **GPU not used**

* Make sure Docker Desktop shows your GPU under **Settings → Resources → GPU**.
* Run containers with `--gpus all`.
* Logs should print something like `Device set to use cuda:0`.
  If not available, the app falls back to CPU.

---

## Manual training (optional)

If you want to run training by hand:

```powershell
$Image = 'sandeep_pandey/crisis-llm:gpu-latest'
$Proc  = "$PWD\data\processed"
$Arts  = "$PWD\artifacts"
mkdir $Arts -Force | Out-Null

docker run --rm --gpus all `
  -e TRANSFORMERS_VERBOSITY=info `
  -e HF_HUB_DISABLE_TELEMETRY=1 `
  -v "${Proc}:/app/data" `
  -v "${Arts}:/app/artifacts" `
  $Image sh -lc "
    python -m pip install -q --upgrade 'accelerate>=1.2.1' || true;
    export PYTHONPATH=/app/src;
    python -u -m ai_tweets.cli train \
      --config configs/gpu.yaml \
      --train-csv data/train.csv \
      --eval-csv  data/val.csv
  "
```

You should end up with files in `artifacts/checkpoints/final/` like:

```
config.json, model.safetensors, tokenizer.json,
special_tokens_map.json, tokenizer_config.json, sentencepiece.bpe.model
```

---

## Tips

* Use **`curl.exe`** or **`Invoke-RestMethod`** in PowerShell to avoid the `curl` alias quirks.
* When in doubt, **check the logs**:

  ```powershell
  docker logs crisis-api --tail 200
  docker logs crisis-ui  --tail 200
  ```
* If a command fails on Windows, ensure you’re running it in **PowerShell**, not CMD.

---

## You’re ready

With the steps above, first-time users can:

1. put `train.csv` / `val.csv` in `data/processed/`,
2. run the one-command script, and
3. open the API docs + UI to test predictions.

If you hit anything new, grab the **last 200 lines of logs** and we’ll troubleshoot fast.
