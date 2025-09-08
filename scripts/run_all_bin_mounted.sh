#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${1:-.}"
CONFIG="${2:-configs/bin_es.yaml}"
IMAGE="${3:-sandeep_pandey/crisis-llm:gpu-latest}"
API_PORT="${4:-8000}"
UI_PORT="${5:-8501}"

HERE="$(cd "$PROJECT_ROOT" && pwd)"
ARTS="$HERE/artifacts"
DATA="$HERE/data"
SRC="$HERE/src/ai_bin"
CFG="$HERE/configs"

mkdir -p "$ARTS" "$DATA/processed"

echo "==> Prepare"
docker run --rm \
  -e PYTHONPATH="/app/src" \
  -v "$DATA:/app/data" \
  -v "$SRC:/app/src/ai_bin" \
  -v "$CFG:/app/configs" \
  "$IMAGE" bash -lc \
  "python -m ai_bin.cli prepare_bin --train data/raw/train.csv --out-train data/processed/train.csv --out-val data/processed/val.csv --use-keyword-location --lowercase"

echo "==> Train ($CONFIG)"
docker run --rm --gpus all \
  -e PYTHONPATH="/app/src" \
  -v "$ARTS:/app/artifacts" -v "$DATA:/app/data" \
  -v "$SRC:/app/src/ai_bin" -v "$CFG:/app/configs" \
  "$IMAGE" bash -lc \
  "python -m ai_bin.train --config $CONFIG --train-csv data/processed/train.csv --eval-csv data/processed/val.csv --out-dir artifacts"

echo "==> Serve"
docker network create crisis-net >/dev/null 2>&1 || true
docker rm -f crisis-api crisis-ui >/dev/null 2>&1 || true

docker run -d --name crisis-api --network crisis-net --gpus all \
  -e PYTHONPATH="/app/src" \
  -p ${API_PORT}:8000 \
  -v "$ARTS:/app/artifacts" -v "$SRC:/app/src/ai_bin" -v "$CFG:/app/configs" \
  -e MODEL_DIR="/app/artifacts/checkpoints/final" \
  "$IMAGE" uvicorn ai_bin.api:app --host 0.0.0.0 --port 8000 --log-level info

docker run -d --name crisis-ui --network crisis-net \
  -e PYTHONPATH="/app/src" \
  -p ${UI_PORT}:8501 \
  -v "$ARTS:/app/artifacts" -v "$SRC:/app/src/ai_bin" \
  -e MODEL_DIR="/app/artifacts/checkpoints/final" -e API_URL="http://crisis-api:8000" \
  "$IMAGE" streamlit run src/ai_bin/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false

echo "API: http://localhost:${API_PORT}/docs"
echo "UI : http://localhost:${UI_PORT}"
