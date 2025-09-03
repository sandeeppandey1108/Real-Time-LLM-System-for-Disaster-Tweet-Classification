from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Disaster Tweet Classifier", version="1.0.0")

MODEL_DIR = "/app/artifacts/checkpoints/final"
DEVICE = 0 if torch.cuda.is_available() else -1

clf = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    device=DEVICE,
    truncation=True
)

PRED_COUNTER = Counter("pred_requests_total", "Number of prediction requests")

class Item(BaseModel):
    text: str

@app.get("/healthz")
def healthz():
    return {"ok": True, "device": "cuda" if DEVICE == 0 else "cpu"}

@app.post("/predict")
def predict(item: Item):
    PRED_COUNTER.inc()
    out = clf(item.text)[0]
    return {"label": out["label"], "score": float(out["score"])}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
