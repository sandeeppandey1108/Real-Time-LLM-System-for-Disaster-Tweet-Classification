from __future__ import annotations
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from pydantic import BaseModel
from transformers import pipeline
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time, json
from typing import Optional, List
from pathlib import Path

REQ = Counter("inference_requests_total", "Total inference requests", ["ep"])
ERR = Counter("inference_errors_total", "Inference request errors", ["ep"])
LAT = Histogram("inference_latency_seconds", "Inference latency (s)", ["ep"])

class PredictIn(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None

def create_app(model_dir: str | Path = "artifacts/model", model_name: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="LLM Text Classification Service", version="1.0.0")
    if Path(model_dir).exists():
        clf = pipeline("text-classification", model=str(model_dir), tokenizer=str(model_dir))
    else:
        name = model_name or "distilbert-base-uncased"
        clf = pipeline("text-classification", model=name, tokenizer=name)

    @app.get("/health")
    def health():
        return {"status":"ok"}

    @app.post("/predict")
    def predict(inp: PredictIn):
        REQ.labels(ep="/predict").inc(); t=time.perf_counter()
        try:
            xs = inp.texts or ([inp.text] if inp.text else None)
            if not xs: raise ValueError("Provide 'text' or 'texts'")
            ys = clf(xs, truncation=True)
            out = [{"label": y["label"].replace("LABEL_",""), "score": float(y["score"])} for y in ys]
            return {"predictions": out}
        except Exception:
            ERR.labels(ep="/predict").inc(); raise
        finally:
            LAT.labels(ep="/predict").observe(time.perf_counter()-t)

    @app.get("/metrics")
    def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                msg = await sock.receive_text()
                REQ.labels(ep="/ws").inc(); t=time.perf_counter()
                try:
                    data = json.loads(msg); text = data.get("text")
                    if not text:
                        await sock.send_text(json.dumps({"error":"send JSON with 'text'"})); continue
                    y = clf([text], truncation=True)[0]
                    await sock.send_text(json.dumps({"label": y["label"].replace("LABEL_",""), "score": float(y["score"])}))
                except Exception as e:
                    ERR.labels(ep="/ws").inc(); await sock.send_text(json.dumps({"error": str(e)}))
                finally:
                    LAT.labels(ep="/ws").observe(time.perf_counter()-t)
        except WebSocketDisconnect:
            pass

    return app

app = create_app()
