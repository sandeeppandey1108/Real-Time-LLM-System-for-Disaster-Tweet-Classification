# Crisis LLM: Disaster Tweet Classifier (GPU)

A tidy, production-ish bundle that trains a multilingual MiniLM classifier,
serves it with FastAPI, and ships a tiny Streamlit UI + website demo.

## Features
- **Training** via your existing `ai-tweets` CLI (confusion matrix + metrics saved).
- **FastAPI** with `/predict`, `/batch`, `/labels`, `/healthz`, `/metrics`.
- **Streamlit** app (local model or REST API mode).
- **Prometheus** metrics.
- One-liners for **Windows PowerShell** and **Bash**.
- Single `docker-compose.yml` to run API + UI together.

---

## Quickstart (Windows PowerShell)

```powershell
$Image = 'yourname/crisis-llm:gpu-latest'

# Build
docker build --no-cache -f Dockerfile.gpu -t $Image .

# Train (installs tokenizer deps inside the run)
docker run --rm --gpus all `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\artifacts:/app/artifacts" `
  $Image `
  sh -lc "python -m pip install -q tiktoken sentencepiece &&               ai-tweets train --config configs/gpu.yaml --train-csv data/train.csv --eval-csv data/val.csv"

# (Optional) Ensure human-readable labels on the saved checkpoint
docker run --rm -v "${PWD}\artifacts:/app/artifacts" $Image `
  python -c "from transformers import AutoConfig; p=r'/app/artifacts/checkpoints/final'; cfg=AutoConfig.from_pretrained(p); cfg.id2label={0:'non_disaster',1:'disaster'}; cfg.label2id={'non_disaster':0,'disaster':1}; cfg.save_pretrained(p); print('labels saved')"

# Serve API
docker compose up -d api
# or:
# docker run -d --gpus all -p 8000:8000 -v "${PWD}\artifacts:/app/artifacts" $Image

# Streamlit UI
docker compose up -d ui
# or:
# docker run -d --gpus all -p 8501:8501 -v "${PWD}\artifacts:/app/artifacts" $Image `
#   sh -lc "streamlit run src/ai_tweets/streamlit_app.py --server.address=0.0.0.0 --server.port=8501"
```

Open:
- API: http://localhost:8000/docs
- UI:  http://localhost:8501
- Demo site: open `web/test.html` in your browser

### Example curl
```bash
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"Road is flooded!","threshold":0.5}'
curl -s http://localhost:8000/batch -H "Content-Type: application/json" -d '{"texts":["Earthquake near city","Coffee time :)"],"threshold":0.5}'
```

---

## docker-compose
The compose file starts both **api** and **ui** (UI calls API at `http://localhost:8000`).

```bash
IMAGE=yourname/crisis-llm:gpu-latest docker compose up -d
```

> GPU usage in Compose depends on Docker Desktop / NVIDIA setup. If needed, start the containers with `docker run --gpus all` manually (see commands above).

---

## Troubleshooting

- **Tokenizer errors** (`tiktoken` / `sentencepiece`) → the Dockerfile pins and installs both. If you're running the training in a bare container, install them first:
  `python -m pip install -q tiktoken sentencepiece`.

- **API is up but `/docs` hangs** → check logs:
  `docker logs -f crisis_api`.

- **All predictions are ~0.50** → train with more data and/or more epochs. The toy set is tiny.

- **Confusion matrix & metrics** → saved to `artifacts/confusion_matrix.png` and `artifacts/metrics.json` by the training CLI.

---

## Endpoints

- `GET /healthz` – server status + model path
- `GET /labels` – id2label/label2id
- `POST /predict` – `{"text": "...", "threshold": 0.5}` → predicted label + score
- `POST /batch` – `{"texts": [...], "threshold": 0.5}` → vectorized predictions
- `GET /metrics` – Prometheus exposition

---

## License
MIT
