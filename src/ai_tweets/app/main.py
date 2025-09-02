import os, time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest
from fastapi import Response
from transformers import pipeline, AutoConfig
import torch

# ---------- Settings ----------
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/artifacts/checkpoints/final")
THRESHOLD_DEFAULT = float(os.environ.get("THRESHOLD", "0.50"))
DEVICE = 0 if torch.cuda.is_available() else -1

# ---------- App ----------
app = FastAPI(
    title="Disaster Tweet Classifier",
    description="FastAPI service for binary disaster classification (disaster / non_disaster).",
    version="1.0.0",
    contact={"name": "crisis-llm", "url": "https://example.com"},
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
registry = CollectorRegistry()
REQ_COUNT = Counter("requests_total", "Total HTTP requests", ["route", "method", "status"], registry=registry)
REQ_LATENCY = Histogram("request_latency_seconds", "Request latency (s)", ["route", "method"], registry=registry)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    route = request.url.path
    method = request.method
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        REQ_COUNT.labels(route=route, method=method, status=status if 'status' in locals() else 500).inc()
        REQ_LATENCY.labels(route=route, method=method).observe(time.perf_counter() - start)

# ---------- Models ----------
class PredictIn(BaseModel):
    text: str = Field(..., examples=["There is a fire downtown!"])
    threshold: Optional[float] = Field(THRESHOLD_DEFAULT, ge=0.0, le=1.0)

class PredictOut(BaseModel):
    label: str
    score: float
    above_threshold: bool

class BatchIn(BaseModel):
    texts: List[str] = Field(..., examples=[["Road is flooded!", "Coffee time :)"]])
    threshold: Optional[float] = Field(THRESHOLD_DEFAULT, ge=0.0, le=1.0)

class BatchOut(BaseModel):
    results: List[PredictOut]

# Load pipeline once
pipe = pipeline(
    "text-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    device=DEVICE,
    truncation=True,
    return_all_scores=True,
)

# Read labels from config (ensures pretty names)
cfg = AutoConfig.from_pretrained(MODEL_DIR)
id2label: Dict[int, str] = getattr(cfg, "id2label", {0: "LABEL_0", 1: "LABEL_1"})
label2id: Dict[str, int] = {v: k for k, v in id2label.items()}

def score_disaster(probs: List[Dict[str, Any]]) -> float:
    # probs: e.g. [{'label': 'disaster', 'score': 0.61}, {'label':'non_disaster', 'score': 0.39}]
    for d in probs:
        if d["label"] in ("disaster", "LABEL_1"):
            return float(d["score"])
    # fallback: take max
    return float(max(probs, key=lambda x: x["score"])["score"])

@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "device": "cuda:0" if DEVICE >= 0 else "cpu", "model_dir": MODEL_DIR, "labels": id2label}

@app.get("/labels")
def labels() -> Dict[str, Any]:
    return {"id2label": id2label, "label2id": label2id}

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    try:
        out = pipe(inp.text)[0]  # list of dicts with return_all_scores=True
        p1 = score_disaster(out)
        label = "disaster" if p1 >= (inp.threshold or THRESHOLD_DEFAULT) else "non_disaster"
        return {"label": label, "score": p1, "above_threshold": p1 >= (inp.threshold or THRESHOLD_DEFAULT)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch", response_model=BatchOut)
def batch(inp: BatchIn):
    try:
        raw = pipe(inp.texts)
        res = []
        thr = inp.threshold or THRESHOLD_DEFAULT
        for probs in raw:
            p1 = score_disaster(probs)
            label = "disaster" if p1 >= thr else "non_disaster"
            res.append({"label": label, "score": p1, "above_threshold": p1 >= thr})
        return {"results": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
