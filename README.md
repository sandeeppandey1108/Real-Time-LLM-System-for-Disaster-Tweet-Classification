# Real-Time LLM System for Disaster Tweet Classification — Updated Runner

This package includes an updated **PowerShell runner** and supporting files that fix:
- `Test-Path ... -and ...` parsing in PowerShell
- `Accelerator.unwrap_model(... keep_torch_compile ...)` training crash by upgrading `accelerate` at runtime
- Safer Docker volume syntax on Windows (`"${Var}:..."`)

## Prerequisites
- Windows 10/11 with **Docker Desktop** (GPU support enabled)
- NVIDIA drivers + CUDA support for GPU containers
- Your dataset CSVs:
  - `data/processed/train.csv`
  - `data/processed/val.csv`

> The repository’s Python code and Docker image are assumed to be already set up. If you rebuild the image, use the included `Dockerfile.gpu` and `requirements.txt` (note the newer `accelerate`).

## Quick Start

1) Place these files in your repo root (same folder that contains `configs/`, `apps/`, `src/` etc.).  
   Overwrite the existing `run_all.ps1` and `README.md` if prompted.

2) (Optional) Rebuild the Docker image:
```powershell
$Image = "sandeep_pandey/crisis-llm:gpu-latest"
docker build -f Dockerfile.gpu -t $Image .
```

3) Train only (no services):
```powershell
& .\run_all.ps1 -Image $Image -StartApi:$false -StartUI:$false
```

4) Train + start services:
```powershell
& .\run_all.ps1 -Image $Image
# API:      http://localhost:8000/docs
# Streamlit http://localhost:8501
```

5) Stop services later:
```powershell
docker stop crisis-api crisis-ui
```

## Known Good Paths
- Training reads from inside the container at `/app/data`. We mount your local `data\processed` into that location.
- Artifacts (model, metrics) are written to your local `artifacts` directory and mapped to `/app/artifacts` in the container.

## Troubleshooting

**A. `Test-Path ... -and ...` error**  
Cause: PowerShell parsing when `-and` was broken across lines or without parentheses.  
Fix: The new `run_all.ps1` wraps each side in parentheses.

**B. `keep_torch_compile` / `unwrap_model` error**  
Cause: Incompatible `accelerate` vs `transformers`.  
Fix: The runner upgrades `accelerate>=1.2.1` before training. If rebuilding the image, keep that pin in `requirements.txt`.

**C. “docker: 'docker run' requires at least 1 argument”**  
Cause: Empty `$Image` or quoting issue.  
Fix: Ensure `$Image` is set (e.g., `sandeep_pandey/crisis-llm:gpu-latest`).

**D. “Variable reference is not valid. ':'...”**  
Cause: PowerShell sees `:` as a drive separator in `-v "$Proc:/app/data"`.  
Fix: Use **curly braces**: `-v "${Proc}:/app/data"` (as done in the new script).

**E. Slow / flaky `pip install` in Docker build**  
Try again, or use a faster mirror. You can also set:
```
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_ROOT_USER_ACTION=ignore
```

---

## Rebuild Notes (Optional)

If you want reproducible training _without_ upgrading at runtime, use the included files:

- **requirements.txt**: sets `accelerate>=1.2.1` (instead of old `1.1.1`).
- **Dockerfile.gpu**: installs requirements and the local package.

```powershell
$Image = "sandeep_pandey/crisis-llm:gpu-latest"
docker build -f Dockerfile.gpu -t $Image .
```

Then run the runner as shown above.
