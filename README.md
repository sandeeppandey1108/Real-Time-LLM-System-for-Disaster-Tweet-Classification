
# Crisis LLM Pro 🚨 (GPU)

End-to-end **disaster tweet classifier**: data prep → training → metrics → **FastAPI** → **Streamlit**.

## Quickstart (Windows PowerShell)

```powershell
cd .\crisis-llm-pro
$Image = "yourname/crisis-llm:gpu-latest"
docker build -f Dockerfile.gpu -t $Image .
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_all.ps1" -Image $Image
```

- API docs: http://localhost:8000/docs  
- Streamlit UI: http://localhost:8501  
- Artifacts: `./artifacts/`

## CLI

```bash
ai-tweets prepare --raw-dir data/raw --out-dir data/processed --val-size 0.2
ai-tweets train --config configs/gpu.yaml --train-csv data/processed/train.csv --eval-csv data/processed/val.csv
ai-tweets serve --model-dir /app/artifacts/checkpoints/final --host 0.0.0.0 --port 8000
```
