\
    param(
      [string]$Image = "yourname/crisis-llm:gpu-latest"
    )
    Write-Host "Image: $Image" -ForegroundColor Cyan

    if (!(Test-Path ".\artifacts")) { New-Item -ItemType Directory -Path ".\artifacts" | Out-Null }
    if (!(Test-Path ".\data")) { New-Item -ItemType Directory -Path ".\data" | Out-Null }

    docker build --no-cache -f Dockerfile.gpu -t $Image .

    docker run --rm --gpus all `
      -v "${PWD}\data:/app/data" `
      -v "${PWD}\artifacts:/app/artifacts" `
      $Image `
      sh -lc "python -m pip install -q tiktoken sentencepiece && \
              ai-tweets train --config configs/gpu.yaml --train-csv data/train.csv --eval-csv data/val.csv"

    docker run --rm -v "${PWD}\artifacts:/app/artifacts" $Image `
      python -c "from transformers import AutoConfig; p=r'/app/artifacts/checkpoints/final'; cfg=AutoConfig.from_pretrained(p); cfg.id2label={0:'non_disaster',1:'disaster'}; cfg.label2id={'non_disaster':0,'disaster':1}; cfg.save_pretrained(p); print('labels saved')"

    docker compose up -d
    Write-Host "API: http://localhost:8000/docs | UI: http://localhost:8501" -ForegroundColor Green
