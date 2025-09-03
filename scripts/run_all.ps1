param(
  [string]$Image = "sandeep_pandey/crisis-llm:gpu-latest"
)

$ErrorActionPreference = "Stop"

# Resolve repo root relative to this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir

# Folders on host
$Raw  = Join-Path $RepoRoot "data\raw"
$Proc = Join-Path $RepoRoot "data\processed"
$Arts = Join-Path $RepoRoot "artifacts"

# Ensure folders exist
New-Item -ItemType Directory -Force -Path $Raw  | Out-Null
New-Item -ItemType Directory -Force -Path $Proc | Out-Null
New-Item -ItemType Directory -Force -Path $Arts | Out-Null

function Run-Docker {
  param([string[]]$Args)
  Write-Host "docker run $($Args -join ' ')" -ForegroundColor Cyan
  docker run @Args
  if ($LASTEXITCODE -ne 0) { throw "docker run failed with exit code $LASTEXITCODE" }
}

Write-Host "[1/5] Preparing data..." -ForegroundColor Green
$trainCsv = Join-Path $Proc "train.csv"
$valCsv   = Join-Path $Proc "val.csv"
if (-not (Test-Path $trainCsv) -or -not (Test-Path $valCsv)) {
  $args1 = @(
    "--rm","--gpus","all",
    "-v","${Raw}:/app/data/raw",
    "-v","${Proc}:/app/data/processed",
    "-v","${RepoRoot}:/app",
    $Image,
    "sh","-lc",
    "export PYTHONPATH=/app/src; python -m ai_tweets.prepare --raw-dir /app/data/raw --out-dir /app/data/processed"
  )
  Run-Docker -Args $args1
} else {
  Write-Host "  Skipping: found processed/train.csv and processed/val.csv" -ForegroundColor Yellow
}

Write-Host "[2/5] Training model..." -ForegroundColor Green
$args2 = @(
  "--rm","--gpus","all",
  "-v","${Proc}:/app/data",
  "-v","${Arts}:/app/artifacts",
  "-v","${RepoRoot}:/app",
  $Image,
  "sh","-lc",
  "export PYTHONPATH=/app/src; python -m ai_tweets.cli train --config configs/gpu.yaml --train-csv data/train.csv --eval-csv data/val.csv"
)
Run-Docker -Args $args2

Write-Host "[3/5] Stamping label maps (id2label/label2id)..." -ForegroundColor Green
# Write a tiny Python script to host, then run it inside the container to avoid quoting issues
$StampFile = Join-Path $Arts "stamp_labels.py"
@"
from transformers import AutoConfig
p='artifacts/checkpoints/final'
cfg=AutoConfig.from_pretrained(p)
cfg.id2label={0:'non_disaster',1:'disaster'}
cfg.label2id={'non_disaster':0,'disaster':1}
cfg.save_pretrained(p)
print('Stamped labels into', p)
"@ | Out-File -FilePath $StampFile -Encoding utf8

$args3 = @(
  "--rm","--gpus","all",
  "-v","${Arts}:/app/artifacts",
  "-v","${RepoRoot}:/app",
  $Image,
  "sh","-lc",
  "python /app/artifacts/stamp_labels.py"
)
Run-Docker -Args $args3

Write-Host "[4/5] Starting API @ http://localhost:8000 ..." -ForegroundColor Green
# Stop previous if running
docker rm -f crisis_api 2>$null | Out-Null
$args4 = @(
  "-d","--gpus","all","--name","crisis_api",
  "-p","8000:8000",
  "-v","${Arts}:/app/artifacts",
  "-v","${RepoRoot}:/app",
  $Image,
  "sh","-lc",
  "export PYTHONPATH=/app/src; uvicorn api.app:app --host 0.0.0.0 --port 8000"
)
Run-Docker -Args $args4

Write-Host "[5/5] Starting Streamlit @ http://localhost:8501 ..." -ForegroundColor Green
docker rm -f crisis_ui 2>$null | Out-Null
$args5 = @(
  "-d","--gpus","all","--name","crisis_ui",
  "-p","8501:8501",
  "-v","${Arts}:/app/artifacts",
  "-v","${RepoRoot}:/app",
  $Image,
  "sh","-lc",
  "export PYTHONPATH=/app/src; streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"
)
Run-Docker -Args $args5

Write-Host ""
Write-Host "Done. Visit:" -ForegroundColor Green
Write-Host "  - API docs:     http://localhost:8000/docs"
Write-Host "  - Streamlit UI: http://localhost:8501"
Write-Host "Artifacts in $Arts" -ForegroundColor Green
