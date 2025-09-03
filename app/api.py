from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI(title="Disaster Tweet Classifier")

MODEL_DIR = "/app/artifacts/checkpoints/final"
DEVICE = 0 if torch.cuda.is_available() else -1
clf = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, device=DEVICE, truncation=True)

class Item(BaseModel):
    text: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict")
def predict(item: Item):
    out = clf(item.text)[0]
    return {"label": out["label"], "score": float(out["score"])}
