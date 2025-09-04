# Real-Time LLM System for Disaster Tweet Classification — Windows Quick Start (Patched)

This patch contains:
- **run_all.ps1** — one command to train and launch the API + Streamlit UI
- **src/ai_tweets/streamlit_app.py** — patched to **load your local checkpoint** (no HF Hub lookups)

Your model artifacts are expected at:
`artifacts/checkpoints/final/` (mapped inside the container to `/app/artifacts/checkpoints/final`).

If you already trained and see files like `model.safetensors`, `config.json`, and `tokenizer.json` in that folder,
you can skip the training step by running the start-only commands below.

---

## Prerequisites

- **Docker Desktop** (Windows, WSL2 backend recommended)
- **NVIDIA GPU** + drivers (optional but recommended) and **NVIDIA Container Toolkit** for Docker GPU support
- **PowerShell**

> Windows may block local scripts. Use `-ExecutionPolicy Bypass` as shown below.

---

## A) Train + Launch (recommended)

Open **PowerShell** in your repo root (the folder that contains `artifacts/` and `data/processed/`).

```powershell
# 1) Set your image (GPU build)
$Image = 'sandeep_pandey/crisis-llm:gpu-latest'

# 2) Run everything (train + start API & UI)
powershell -NoProfile -ExecutionPolicy Bypass -File .un_all.ps1 -Image $Image
```

When it finishes, open:
- API docs: **http://localhost:8000/docs**
- Streamlit UI: **http://localhost:8501**

> Use other ports:  
> `powershell -NoProfile -ExecutionPolicy Bypass -File .un_all.ps1 -Image $Image -ApiPort 8010 -UiPort 8510`

---

## B) Start Only (if you already have a trained checkpoint)

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

## E) Verify model artifacts

```powershell
Get-ChildItem "$PWDrtifacts\checkpointsinal"
# OR inside the image:
docker run --rm -v "$PWDrtifacts:/app/artifacts" sandeep_pandey/crisis-llm:gpu-latest bash -lc "ls -lah /app/artifacts/checkpoints/final"
```

You should see: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `sentencepiece.bpe.model`.

---

## Known Errors & Fixes (mapped to exact messages you saw)

### 1) **HFValidationError** in Streamlit
```
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/app/artifacts/checkpoints/final'...
```
**Cause:** The default pipeline tried to treat a local path like a Hub repo ID.  
**Fix:** This patch’s `src/ai_tweets/streamlit_app.py` forces **local, offline** loading using `AutoTokenizer/AutoModelForSequenceClassification(..., local_files_only=True)` and sets `HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE`. It **never** goes to the network.

### 2) **TypeError: Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'**
Happens during training due to incompatible `accelerate` version.
**Fix:** The training step in `run_all.ps1` upgrades `accelerate` inside the container:
```
python -m pip install -q --upgrade 'accelerate>=1.2.1'
```

### 3) **Port already allocated**
```
Bind for 0.0.0.0:8000 failed: port is already allocated
Bind for 0.0.0.0:8501 failed: port is already allocated
```
**Fix:** Either stop old containers:
```powershell
docker rm -f crisis-api crisis-ui
```
Or choose different ports (`-ApiPort 8010 -UiPort 8510`). The script checks and throws a clean error if busy.

### 4) **Empty reply from server** on `/healthz`
```
curl: (52) Empty reply from server
```
**Fixes to try:**
- Give the API a few seconds to fully start, then retry.
- Check logs:
  ```powershell
  docker logs crisis-api --tail 200
  ```
- Ensure no corporate proxy/firewall interferes with `localhost` ports.
- Confirm ports changed if you had collisions (see #3).

### 5) **Script is not digitally signed / UnauthorizedAccess**
```
File ...un_all.ps1 cannot be loaded. The file is not digitally signed.
```
**Fix:** invoke with policy bypass (already shown everywhere):
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .un_all.ps1 -Image $Image
```

### 6) **PowerShell parsing quirks (variable-in-path, JSON quotes)**

- When mounting volumes with `-v`, always use **`${Var}`** inside quotes (the script does this for you):
  ```powershell
  -v "${Arts}:/app/artifacts"
  ```
- For JSON in PowerShell, prefer `Invoke-RestMethod` with `ConvertTo-Json` (see section **C**) to avoid quoting issues.

### 7) **Dataset not found / `Test-Path : A parameter cannot be found that matches parameter name 'and'`**
If you saw this earlier, it came from a buggy line like:
```powershell
if (Test-Path $trainCsv -and Test-Path $valCsv) { ... }
```
**Fix:** The included script uses the correct `-and` operator in a single `if` (fixed) and gives a clear error if files are missing. Ensure `data/processed/train.csv` and `data/processed/val.csv` exist.

### 8) **DNS error / xethub / huggingface hub during training/UI**
If you saw noisy logs like `reqwest ... dns error ... xethub.hf.co`, these were Hub lookups.
**Fix:** The patch enforces **offline** for the UI; training can run online but does not need Hub during fine-tuning if the base model is cached inside the image. If your environment is fully offline, keep `HF_HUB_OFFLINE=1` and ensure the base model is present in the image (the provided image already caches the base model).

### 9) **Switch parameter error**
```
Cannot convert value "System.String" to type "SwitchParameter"
```
**Fix:** For switches in PowerShell, pass them like `-SkipTrain` (present = true) **or** explicitly `-NoApi:$true` / `-NoApi:$false`. The provided script accepts `-SkipTrain`, `-NoApi`, `-NoUI`.

---

## Tips

- To re-run **just** the API/UI without retraining:
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File .un_all.ps1 -Image $Image -SkipTrain
  ```
- To start **API only**:
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File .un_all.ps1 -Image $Image -NoUI
  ```
- To start **UI only** (assumes API already started, but not strictly required):
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File .un_all.ps1 -Image $Image -NoApi -SkipTrain
  ```

Good luck — and congrats on getting the full pipeline working on Windows! 🎉
