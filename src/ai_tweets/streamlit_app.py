import os
import streamlit as st
from transformers import pipeline, AutoConfig
import torch
import requests

st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨", layout="centered")
st.title("🚨 Disaster Tweet Classifier")
st.caption("Binary classifier: **disaster** vs **non_disaster**")

mode = st.sidebar.radio("Mode", ["Use REST API", "Use Local Model"], index=0)
api_url = st.sidebar.text_input("API URL", os.environ.get("API_URL", "http://localhost:8000"))
threshold = st.sidebar.slider("Decision threshold (disaster)", 0.0, 1.0, 0.50, 0.01)

examples = [
    "There is a fire downtown!",
    "Earthquake near city",
    "Lovely day in the park",
    "Coffee time :)",
    "Road is flooded!",
    "Nothing to report",
]
ex = st.selectbox("Examples", examples, index=0)
text = st.text_area("Your tweet", ex, height=100)

def call_api(t: str):
    r = requests.post(f"{api_url}/predict", json={"text": t, "threshold": threshold}, timeout=20)
    r.raise_for_status()
    return r.json()

if mode == "Use Local Model":
    MODEL_DIR = os.environ.get("MODEL_DIR", "/app/artifacts/checkpoints/final")
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, device=device, return_all_scores=True, truncation=True)
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    id2label = getattr(cfg, "id2label", {0:"LABEL_0", 1:"LABEL_1"})
    st.info(f"Using local model from {MODEL_DIR} on {'GPU' if device>=0 else 'CPU'}")

    def local_predict(t: str):
        probs = clf(t)[0]
        p1 = [d for d in probs if d['label'] in ('disaster', 'LABEL_1')]
        score = float(p1[0]['score']) if p1 else float(max(probs, key=lambda x:x['score'])['score'])
        label = "disaster" if score >= threshold else "non_disaster"
        return {"label": label, "score": score, "above_threshold": score >= threshold}

if st.button("Classify"):
    with st.spinner("Running..."):
        try:
            result = call_api(text) if mode == "Use REST API" else local_predict(text)
            score_pct = round(result["score"] * 100, 1)
            if result["label"] == "disaster":
                st.success(f"Prediction: **DISASTER** ({score_pct}%)")
            else:
                st.warning(f"Prediction: **NON-DISASTER** ({score_pct}%)")
            st.progress(result["score"])
        except Exception as e:
            st.error(str(e))

st.divider()
st.subheader("Batch test")
batch = st.text_area("One per line", "\n".join(examples[:3]), height=140)
if st.button("Run batch"):
    items = [s for s in batch.splitlines() if s.strip()]
    with st.spinner("Batching..."):
        try:
            if mode == "Use REST API":
                r = requests.post(f"{api_url}/batch", json={"texts": items, "threshold": threshold}, timeout=60)
                r.raise_for_status()
                out = r.json()["results"]
            else:
                out = [local_predict(s) for s in items]
            for s, o in zip(items, out):
                st.write(f"- {s} → **{o['label']}** ({o['score']:.3f})")
        except Exception as e:
            st.error(str(e))
