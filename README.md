# Crisis LLM Pro — Disaster Tweet Classification

End‑to‑end project to train a multilingual disaster tweet classifier and serve it via a FastAPI and a Streamlit UI.
Works on Windows with Docker + (optional) NVIDIA GPU.

## Quickstart (Windows PowerShell)

```powershell
# From the project root
$Image = "sandeep_pandey/crisis-llm:gpu-latest"   # or your own tag
docker build -f Dockerfile.gpu -t $Image .

# Put your csv files into .\data\raw\ (train.csv, val.csv, test.csv, submission.csv — any/all)
# Then run the whole pipeline (prepare -> train -> stamp -> API -> Streamlit)
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_all.ps1" -Image $Image
```

- API docs: <http://localhost:8000/docs>
- Streamlit UI: <http://localhost:8501>

## What the pipeline does

1) **Prepare** — merge every CSV in `data/raw/`, keep rows with labels (`target` or `label`), clean text, dedupe, and stratified split to `data/processed/train.csv` and `data/processed/val.csv`.
2) **Train** — fine‑tune `microsoft/Multilingual-MiniLM-L12-H384` for binary classification (disaster vs non_disaster).
3) **Stamp labels** — ensures `id2label` & `label2id` are in the saved checkpoint.
4) **Serve API** — FastAPI on `:8000` with `/healthz` and `/predict`.
5) **Streamlit UI** — simple web app at `:8501` that calls the API and shows predictions and confidence.

## Local smoke tests

```powershell
Invoke-RestMethod http://localhost:8000/healthz
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType 'application/json' `
  -Body (@{ text = 'There is a fire downtown!' } | ConvertTo-Json)
```

## Notes

- If you see tokenizer/tiktoken errors, this image installs `tiktoken` and `sentencepiece` already.
- If you see dependency conflicts at build time, we pin a compatible `huggingface_hub` version.
- If you don’t have a GPU, the API automatically falls back to CPU.
