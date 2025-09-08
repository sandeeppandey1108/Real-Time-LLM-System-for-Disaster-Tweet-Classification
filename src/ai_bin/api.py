
import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/checkpoints/final")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Disaster Tweet Classifier - Binary", version="1.0")

class Req(BaseModel):
    text: str

class Resp(BaseModel):
    label: str
    score: float

tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(DEVICE)
model.eval()

@app.get("/healthz")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict", response_model=Resp)
def predict(r: Req):
    enc = tok([r.text], padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k,v in enc.items()}
    with torch.no_grad():
        out = model(**enc).logits.softmax(-1).squeeze(0).cpu().numpy().tolist()
    # id2label: 0=fake, 1=real
    score_real = out[1]
    label = "real" if score_real >= 0.5 else "fake"
    return {"label": label, "score": float(score_real)}
