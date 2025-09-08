
param(
  [switch]$Full = $false,
  [string]$ProjectRoot = ".",
  [string]$Config = "configs/bin_es.yaml",
  [int]$ApiPort = 8000,
  [int]$UiPort = 8501,
  [string]$Image = "sandeep_pandey/crisis-llm:gpu-latest"
)

if ($ProjectRoot) {
  $Here = (Resolve-Path $ProjectRoot).Path
} else {
  $Here = (Get-Location).Path
}
$Arts = Join-Path $Here "artifacts"
$Data = Join-Path $Here "data"
$Src  = Join-Path $Here "src\ai_bin"
$Cfg  = Join-Path $Here "configs"

Write-Host "ProjectRoot: $Here"
Write-Host "Artifacts : $Arts"
Write-Host "Data      : $Data"
Write-Host "Src       : $Src"
Write-Host "Configs   : $Cfg"
Write-Host "Config YML: $Config"

New-Item -ItemType Directory -Force -Path $Arts | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Data "processed") | Out-Null

if ($Full) {
  Write-Host "==> Preparing binary dataset"
  docker run --rm `
    -e PYTHONPATH="/app/src" `
    -v "${Data}:/app/data" `
    -v "${Src}:/app/src/ai_bin" `
    -v "${Cfg}:/app/configs" `
    $Image bash -lc `
    "python -m ai_bin.cli prepare_bin --train data/raw/train.csv --out-train data/processed/train.csv --out-val data/processed/val.csv --use-keyword-location --lowercase"
}

Write-Host "==> Training (config: $Config)"
docker run --rm --gpus all `
  -e PYTHONPATH="/app/src" `
  -v "${Arts}:/app/artifacts" -v "${Data}:/app/data" `
  -v "${Src}:/app/src/ai_bin" -v "${Cfg}:/app/configs" `
  $Image bash -lc `
  "python -m ai_bin.train --config $Config --train-csv data/processed/train.csv --eval-csv data/processed/val.csv --out-dir artifacts"

Write-Host "==> Starting API + UI"
docker network create crisis-net 2>$null | Out-Null
docker rm -f crisis-api crisis-ui 2>$null | Out-Null

docker run -d --name crisis-api --network crisis-net --gpus all `
  -e PYTHONPATH="/app/src" `
  -p ${ApiPort}:8000 `
  -v "${Arts}:/app/artifacts" -v "${Src}:/app/src/ai_bin" -v "${Cfg}:/app/configs" `
  -e MODEL_DIR="/app/artifacts/checkpoints/final" `
  $Image uvicorn ai_bin.api:app --host 0.0.0.0 --port 8000 --log-level info

docker run -d --name crisis-ui --network crisis-net `
  -e PYTHONPATH="/app/src" `
  -p ${UiPort}:8501 `
  -v "${Arts}:/app/artifacts" -v "${Src}:/app/src/ai_bin" `
  -e MODEL_DIR="/app/artifacts/checkpoints/final" -e API_URL="http://crisis-api:8000" `
  $Image streamlit run src/ai_bin/streamlit_app.py --server.port 8501 --browser.gatherUsageStats false

Write-Host ""
Write-Host "API: http://localhost:$ApiPort/docs"
Write-Host "UI : http://localhost:$UiPort"
