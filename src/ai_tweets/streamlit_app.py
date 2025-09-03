
import os, json
import streamlit as st
import torch
from transformers import pipeline
from pathlib import Path

st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨", layout="centered")
ART_DIR = Path("/app/artifacts") if Path("/app").exists() else Path("artifacts")
MODEL_DIR = os.environ.get("MODEL_DIR", str(ART_DIR / "checkpoints" / "final"))
DEVICE = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, device=DEVICE, truncation=True)

clf = load_pipeline()
st.title("🚨 Disaster Tweet Classifier")
st.caption("GPU aware • HF Transformers • FastAPI & Streamlit")

col1, col2 = st.columns(2)
with col1: th = st.slider("Decision threshold (disaster)", 0.0, 1.0, 0.5, 0.01)
with col2: st.write("Device:", "CUDA" if DEVICE == 0 else "CPU")

st.header("Try it")
txt = st.text_area("Tweet text", "There is a fire downtown!")
if st.button("Predict"):
    out = clf(txt)[0]
    prob = out["score"] if out["label"].lower().endswith("disaster") else 1 - out["score"]
    pred = "disaster" if prob >= th else "non_disaster"
    st.write(f"**Pred:** {pred}  •  **Score:** {prob:.3f}  •  Raw: {out}")

st.header("Batch")
batch = st.text_area("One tweet per line", "Flood on Main St\nCoffee time :)")
if st.button("Batch predict"):
    lines = [l.strip() for l in batch.splitlines() if l.strip()]
    outs = clf(lines)
    rows = []
    for text, o in zip(lines, outs):
        prob = o[0]["score"] if o[0]["label"].lower().endswith("disaster") else 1 - o[0]["score"]
        pred = "disaster" if prob >= th else "non_disaster"
        rows.append({"text": text, "pred": pred, "score": round(prob,3)})
    st.dataframe(rows, use_container_width=True)

st.header("Metrics & Confusion Matrix")
mpath = ART_DIR / "metrics.json"
if mpath.exists(): st.json(json.loads(mpath.read_text()))
img = ART_DIR / "confusion_matrix.png"
if img.exists(): st.image(str(img), caption="Confusion Matrix", use_column_width=True)

st.divider()
st.caption("Tip: Use the FastAPI at /predict and /batch_predict for production.")
