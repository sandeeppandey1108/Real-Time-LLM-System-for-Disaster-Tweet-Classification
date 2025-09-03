
import os, torch
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/artifacts/checkpoints/final")
DEVICE = 0 if torch.cuda.is_available() else -1
clf = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, device=DEVICE, truncation=True)

app = FastAPI(title="Disaster Tweet Classifier", version="1.0.0")

class Item(BaseModel): text: str
class BatchItems(BaseModel): texts: List[str]

@app.get("/healthz")
def healthz(): return {"ok": True, "device": "cuda" if DEVICE == 0 else "cpu"}

@app.post("/predict")
def predict(item: Item):
    out = clf(item.text)[0]
    return {"label": out["label"], "score": float(out["score"])}

@app.post("/batch_predict")
def batch_predict(items: BatchItems):
    outs = clf(items.texts)
    return [{"label": o[0]["label"], "score": float(o[0]["score"])} for o in outs]
