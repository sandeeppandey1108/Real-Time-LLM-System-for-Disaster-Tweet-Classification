#!/usr/bin/env bash
set -euo pipefail
IMAGE="${1:-yourname/crisis-llm:gpu-latest}"

echo "[1/5] Prepare..."
docker run --rm -v "$PWD/data:/app/data" "$IMAGE" ai-tweets prepare

echo "[2/5] Train..."
docker run --rm --gpus all       -v "$PWD/data:/app/data"       -v "$PWD/artifacts:/app/artifacts"       "$IMAGE" ai-tweets train --config configs/gpu.yaml         --train-csv data/processed/train.csv --eval-csv data/processed/val.csv

echo "[3/5] Stamp labels..."
docker run --rm -v "$PWD/artifacts:/app/artifacts" "$IMAGE"       python -c "from transformers import AutoConfig; p='/app/artifacts/checkpoints/final'; cfg=AutoConfig.from_pretrained(p); cfg.id2label={0:'non_disaster',1:'disaster'}; cfg.label2id={'non_disaster':0,'disaster':1}; cfg.save_pretrained(p); print('labels saved')"

echo "[4/5] API -> http://localhost:8000"
docker rm -f crisis_api >/dev/null 2>&1 || true
docker run -d --gpus all --name crisis_api -p 8000:8000       -v "$PWD/artifacts:/app/artifacts" "$IMAGE" ai-tweets serve >/dev/null

echo "[5/5] Streamlit -> http://localhost:8501"
docker rm -f crisis_ui >/dev/null 2>&1 || true
docker run -d --gpus all --name crisis_ui -p 8501:8501       -v "$PWD/artifacts:/app/artifacts" "$IMAGE" ai-tweets streamlit >/dev/null

echo "Done."
