from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch, os

app = FastAPI(title="Disaster Tweet Classifier", version="0.3.0")

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/artifacts/checkpoints/final")
DEVICE = 0 if torch.cuda.is_available() else -1

clf = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, device=DEVICE, truncation=True)

class Item(BaseModel):
    text: str

@app.get("/healthz")
def healthz():
    return {"ok": True, "device": ("cuda" if DEVICE == 0 else "cpu")}

@app.post("/predict")
def predict(item: Item):
    out = clf(item.text)[0]
    return {"label": out["label"], "score": float(out["score"])}
