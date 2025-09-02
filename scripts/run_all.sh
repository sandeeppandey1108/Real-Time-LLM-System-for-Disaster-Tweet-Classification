#!/usr/bin/env bash
set -euo pipefail
IMAGE="${1:-yourname/crisis-llm:gpu-latest}"
echo "Image: $IMAGE"

mkdir -p artifacts data
docker build --no-cache -f Dockerfile.gpu -t "$IMAGE" .

docker run --rm --gpus all       -v "$PWD/data:/app/data"       -v "$PWD/artifacts:/app/artifacts"       "$IMAGE"       sh -lc "python -m pip install -q tiktoken sentencepiece &&               ai-tweets train --config configs/gpu.yaml --train-csv data/train.csv --eval-csv data/val.csv"

docker run --rm -v "$PWD/artifacts:/app/artifacts" "$IMAGE"       python -c "from transformers import AutoConfig; p=r'/app/artifacts/checkpoints/final'; cfg=AutoConfig.from_pretrained(p); cfg.id2label={0:'non_disaster',1:'disaster'}; cfg.label2id={'non_disaster':0,'disaster':1}; cfg.save_pretrained(p); print('labels saved')"

IMAGE="$IMAGE" docker compose up -d
echo "API: http://localhost:8000/docs | UI: http://localhost:8501"
