\
param(
  [Parameter(Mandatory=$true)][string]$Image
)

$ErrorActionPreference = "Stop"
Write-Host "Image: $Image" -ForegroundColor Cyan

# Ensure folders
New-Item -ItemType Directory -Force -Path ".\data\raw"        | Out-Null
New-Item -ItemType Directory -Force -Path ".\data\processed"  | Out-Null
New-Item -ItemType Directory -Force -Path ".\artifacts"       | Out-Null

# If CSVs are in the repo root, stage them into data\raw
Get-ChildItem -Path . -Filter *.csv -File | ForEach-Object {
  Copy-Item -Force $_.FullName ".\data\raw\"
}

# 1) Prepare data (combine + clean + split)
Write-Host "`n[1/5] Preparing data..." -ForegroundColor Yellow
docker run --rm -v "${PWD}\data:/app/data" $Image sh -lc @'
python - << "PY"
import os, glob, sys, pandas as pd
from sklearn.model_selection import train_test_split

RAW = "/app/data/raw"
OUT = "/app/data/processed"
os.makedirs(OUT, exist_ok=True)

def pick_text_col(df):
    candidates = [c for c in df.columns if str(c).lower() in {"text","tweet","message","content"}]
    if candidates: return candidates[0]
    # fallback: first object dtype col
    for c in df.columns:
        if df[c].dtype == "object":
            return c
    return df.columns[0]

dfs_labeled = []
for p in glob.glob(os.path.join(RAW, "*.csv")):
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"Skipping {p}: {e}", file=sys.stderr)
        continue
    cols = {c.lower(): c for c in df.columns}
    # label/target unification
    label_col = None
    if "target" in cols: label_col = cols["target"]
    elif "label" in cols: label_col = cols["label"]
    else:
        # no labels → ignore
        continue
    text_col = pick_text_col(df)
    sub = df[[text_col, label_col]].rename(columns={text_col:"text", label_col:"target"})
    # normalize labels
    if sub["target"].dtype == "O":
        sub["target"] = sub["target"].str.strip().str.lower().map({"disaster":1,"non_disaster":0,"1":1,"0":0})
    sub["target"] = sub["target"].astype("Int64")
    sub = sub.dropna(subset=["text","target"])
    # basic cleaning
    sub["text"] = sub["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    sub = sub.drop_duplicates(subset=["text","target"])
    dfs_labeled.append(sub)

if not dfs_labeled:
    raise SystemExit("No labeled CSVs found in /app/data/raw. Put train/val csvs there.")

df_all = pd.concat(dfs_labeled, ignore_index=True)
# guard small/edge class splits
test_size = 0.2
if df_all["target"].nunique() < 2 or df_all["target"].value_counts().min() < 2:
    # too small/imbalanced for stratify; do simple split
    df_train, df_val = df_all.iloc[:-max(1, int(len(df_all)*test_size))], df_all.iloc[-max(1, int(len(df_all)*test_size)):] 
else:
    df_train, df_val = train_test_split(
        df_all, test_size=test_size, random_state=42, stratify=df_all["target"]
    )

df_train.to_csv(os.path.join(OUT, "train.csv"), index=False)
df_val.to_csv(os.path.join(OUT, "val.csv"), index=False)
print({"train": len(df_train), "val": len(df_val)})
PY
'@
if ($LASTEXITCODE -ne 0) { throw "Data prep failed." }

# 2) Train
Write-Host "`n[2/5] Training..." -ForegroundColor Yellow
docker run --rm --gpus all -v "${PWD}\data:/app/data" -v "${PWD}\artifacts:/app/artifacts" $Image `
  ai-tweets train --config configs/gpu.yaml --train-csv data/processed/train.csv --eval-csv data/processed/val.csv

# 3) Stamp labels
Write-Host "`n[3/5] Stamping labels..." -ForegroundColor Yellow
docker run --rm -v "${PWD}\artifacts:/app/artifacts" $Image `
  python - <<'PY'
from transformers import AutoConfig
p = "/app/artifacts/checkpoints/final"
cfg = AutoConfig.from_pretrained(p)
cfg.id2label = {0:"non_disaster", 1:"disaster"}
cfg.label2id = {"non_disaster":0, "disaster":1}
cfg.save_pretrained(p)
print("labels saved")
PY

# 4) Start API
Write-Host "`n[4/5] Starting API on http://localhost:8000 ..." -ForegroundColor Yellow
docker rm -f crisis_api 2>$null | Out-Null
docker run -d --gpus all --name crisis_api -p 8000:8000 -v "${PWD}\artifacts:/app/artifacts" $Image sh -lc @'
python - << "PY"
import os, torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Disaster Tweet Classifier", version="1.0")
MODEL_DIR="/app/artifacts/checkpoints/final"
device = 0 if torch.cuda.is_available() else -1
clf = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, device=device, truncation=True)

class Item(BaseModel):
    text: str

@app.get("/healthz")
def healthz(): return {"ok": True, "device": ("cuda" if device==0 else "cpu")}

@app.post("/predict")
def predict(item: Item):
    out = clf(item.text)[0]
    return {"label": out["label"], "score": float(out["score"])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
PY
'@ | Out-Null

Start-Sleep -Seconds 2
try {
  Write-Host "Healthz:" -NoNewline
  Invoke-RestMethod http://localhost:8000/healthz | ConvertTo-Json
} catch {
  Write-Host "`n(API not responding; last logs)" -ForegroundColor Red
  docker logs --tail=100 crisis_api
}

# 5) Launch Streamlit
Write-Host "`n[5/5] Launching Streamlit on http://localhost:8501 ..." -ForegroundColor Yellow
docker rm -f crisis_ui 2>$null | Out-Null
docker run -d --name crisis_ui -p 8501:8501 --env API_URL="http://host.docker.internal:8000" $Image sh -lc @'
python - << "PY"
import os, requests, streamlit as st
API = os.environ.get("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨", layout="centered")
st.title("🚨 Disaster Tweet Classifier")
st.caption(f"API: {API}")

txt = st.text_area("Enter a tweet:", "There is a fire downtown!")
if st.button("Predict"):
    try:
        r = requests.post(f"{API}/predict", json={"text": txt}, timeout=10)
        r.raise_for_status()
        out = r.json()
        st.success(f"**{out['label']}**  (score: {out['score']:.3f})")
    except Exception as e:
        st.error(str(e))

st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("API Health"):
        st.write(requests.get(f"{API}/healthz").json())
with col2:
    st.write("Tips: paste a few tweets to test.")

import subprocess, sys, os
from threading import Thread
def run():
    os.execvp("streamlit", ["streamlit","run","-q","/tmp/app.py","--server.port","8501","--server.address","0.0.0.0"])
# write ourselves to /tmp/app.py and exec streamlit
import inspect, pathlib
p = pathlib.Path("/tmp/app.py"); p.write_text(inspect.getsource(sys.modules[__name__]))
run()
PY
'@ | Out-Null

Write-Host "`nDone. Visit:" -ForegroundColor Green
Write-Host "  - API docs:     http://localhost:8000/docs"
Write-Host "  - Streamlit UI: http://localhost:8501"
Write-Host "Artifacts in .\artifacts (model, metrics, confusion_matrix.png)"
