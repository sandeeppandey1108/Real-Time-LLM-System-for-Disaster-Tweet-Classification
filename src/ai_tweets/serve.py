from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/artifacts/checkpoints/final")

app = FastAPI(title="Disaster Tweet Classifier", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clf = None

class Item(BaseModel):
    text: str

class BatchItems(BaseModel):
    texts: list[str]

@app.on_event("startup")
def _load():
    global clf
    clf = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, device=0, truncation=True)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/readiness")
def ready():
    # Try a no-op call
    out = clf("ok")
    return {"ready": True, "label": out[0]["label"]}

@app.post("/predict")
def predict(item: Item):
    out = clf(item.text, return_all_scores=False)[0]
    return {"label": out["label"], "score": float(out["score"])}

@app.post("/predict_proba")
def predict_proba(item: Item):
    scores = clf(item.text, return_all_scores=True)[0]
    return {s["label"]: float(s["score"]) for s in scores}

@app.post("/batch_predict")
def batch_predict(items: BatchItems):
    outs = clf(items.texts, return_all_scores=False)
    return [{"label": o["label"], "score": float(o["score"])} for o in outs]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
