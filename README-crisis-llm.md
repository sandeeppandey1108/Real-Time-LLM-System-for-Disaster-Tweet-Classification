# Crisis LLM — Train, API, and UI (Windows + Docker)

This package contains a one-command workflow for training the disaster tweet classifier, then (optionally) starting the REST API and Streamlit UI.

## Prerequisites
- **Docker Desktop** with **WSL2** integration
- **NVIDIA GPU** + drivers + CUDA runtime for Docker. Test with:
  ```powershell
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
- The Docker image built locally (or pulled) e.g.:
  ```powershell
  docker build -f Dockerfile.gpu -t sandeep_pandey/crisis-llm:gpu-latest .
  ```

## Data layout
Put your CSVs here:
```text
data/processed/train.csv
data/processed/val.csv
```
Required columns:
- `text` — the tweet content
- `labels` (preferred) or `target` — integer 0 = non_disaster, 1 = disaster

## Quick start
Place `run_all.ps1` in the **project root** (same folder as `Dockerfile.gpu`). Then run:

```powershell
# From the project root
Unblock-File .\run_all.ps1   # avoid SmartScreen prompts (optional)
$Image = "sandeep_pandey/crisis-llm:gpu-latest"
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image
```

This will:
1) Validate the data,
2) **Train** the model (with verbose logs),
3) (Optional) Start the API and UI if you add `-StartApi -StartUI`.

### Train only
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image -StartApi:$false -StartUI:$false
```

### Start API + UI (after training)
```powershell
# API → http://localhost:8000/docs
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image -Prepare:$false -Train:$false -StartApi

# UI  → http://localhost:8501
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -Image $Image -Prepare:$false -Train:$false -StartUI
```

## Inference examples
**REST API** (after starting API):
```powershell
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{ \"text\": \"Fire at the mall, people evacuating\" }"
```

## Troubleshooting
- **No output during training**: We run Python unbuffered and increase verbosity. If you still see little output, use:
  ```powershell
  docker rm -f crisis-train 2>$null
  docker run -d --rm --name crisis-train --gpus all `
    -e TRANSFORMERS_VERBOSITY=info -e HF_HUB_DISABLE_TELEMETRY=1 `
    -v "${PWD}\data\processed:/app/data" -v "${PWD}\artifacts:/app/artifacts" `
    sandeep_pandey/crisis-llm:gpu-latest sh -lc "export PYTHONPATH=/app/src; python -u -m ai_tweets.cli train --config configs/gpu.yaml --train-csv data/train.csv --eval-csv data/val.csv"
  docker logs -f crisis-train
  ```
- **`Accelerator.unwrap_model() ... keep_torch_compile` error**: The script upgrades `accelerate` in the container to ≥1.2.1 before training.
- **PowerShell variable + colon volume path error**: Always use braces, e.g. `-v "${Proc}:/app/data"` rather than `-v "$Proc:/app/data"`.
- **Pip root warning**: Harmless in containers. We ignore it; or set `PIP_ROOT_USER_ACTION=ignore`.

## Outputs
After training, artifacts will appear in:
```
artifacts\checkpoints\final\   # model files
artifacts\metrics.json
artifacts\confusion_matrix.png
artifacts\label_map.json
```