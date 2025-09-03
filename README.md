# Crisis LLM (GPU) — End-to-End
A clean, Dockerized pipeline to **prepare data**, **train** a transformer classifier for disaster tweets, and serve both a **FastAPI** endpoint and **Streamlit** UI.

## Features
- **CSV auto-merge**: drops in any mix of `train.csv`, `val.csv`, `test.csv`, `submission.csv` (and others). We detect columns, normalize to `text`/`target`, dedupe, and stratified split 80/20.
- **Robust training**: `microsoft/Multilingual-MiniLM-L12-H384` with pinned dependencies (`transformers==4.56.0`, `huggingface_hub==0.34.4`, `tiktoken`, `sentencepiece`, etc.).
- **Human-readable labels**: final checkpoint stamped with `id2label` / `label2id`.
- **APIs + UI**: FastAPI at `:8000`, Streamlit at `:8501` (with Docker Desktop friendly networking).

---

## Quickstart

### 0) Prereqs
- **Docker Desktop** with GPU support
- **Windows PowerShell**

### 1) Build the image
```powershell
$Image = "sandeep_pandey/crisis-llm:gpu-latest"
docker build -f Dockerfile.gpu -t $Image .
```

### 2) Add your CSVs
Put your CSV files in `.\data\`. Typical Kaggle columns like `text`, `target` are detected automatically.
- If a CSV **has no labels**, it's ignored for training (e.g., `test.csv`).
- Labeled rows across all CSVs are merged and split into `.\data\processed\train.csv` and `.\data\processed\val.csv`.

### 3) One command: prepare → train → serve
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_all.ps1" -Image $Image
```

This will:
1. Merge + clean + split your CSVs
2. Train the classifier
3. Stamp labels on the final checkpoint
4. Start **FastAPI** on `http://localhost:8000` (docs at `/docs`)
5. Start **Streamlit** on `http://localhost:8501`

### 4) Test
**API health**
```powershell
Invoke-RestMethod http://localhost:8000/healthz
```
**Prediction**
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body (@{text='There is a fire downtown!'} | ConvertTo-Json)
```

Or open **http://localhost:8501** and try the UI.

---

## Notes & Troubleshooting

- **Slow first run**: the base CUDA image + model downloads can take time; subsequent runs are much faster.
- **Tokenizer errors**: We pre-install `tiktoken` and `sentencepiece`. If you swap models, make sure the tokenizer deps fit.
- **Couldn't connect to API from Streamlit**: we default to `API_BASE=http://host.docker.internal:8000` which works on Docker Desktop (Windows/Mac). If you're on Linux, set `API_BASE=http://localhost:8000` and rebuild or override the env when running the UI container.
- **GPU vs CPU**: API auto-detects CUDA. Training uses mixed precision (`fp16`) only if a GPU is present.

---

## Project Layout
```
configs/gpu.yaml        # training hyperparams
tools/prep_data.py      # merge/clean/split CSVs
src/train.py            # train + save metrics + CM
api/app.py              # FastAPI inference server
ui/streamlit_app.py     # interactive test UI
scripts/run_all.ps1     # end-to-end runner
Dockerfile.gpu          # GPU-enabled Dockerfile
requirements.txt        # pinned deps
data/                   # put your CSVs here
artifacts/              # model + metrics output here
```

---

## License
MIT (sample code). Replace, extend, and ship!
