from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Disaster Tweet Classifier", version="1.0")
MODEL_DIR = "/app/artifacts/checkpoints/final"
clf = None

class Item(BaseModel):
    text: str

@app.on_event("startup")
def _load():
    global clf
    clf = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, device=0, truncation=True)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict")
def predict(item: Item):
    out = clf(item.text)[0]
    return {"label": out["label"], "score": float(out["score"])}
