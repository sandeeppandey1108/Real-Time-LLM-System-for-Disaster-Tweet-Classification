import streamlit as st
import requests, os, torch
from transformers import pipeline

st.set_page_config(page_title="Disaster Tweet Classifier", layout="centered")

st.title("🔎 Disaster Tweet Classifier")
st.caption("Type a tweet and classify it as **disaster** or **non_disaster**.")

text = st.text_area("Tweet text", "", height=120, placeholder="There is a fire downtown!")

col1, col2 = st.columns(2)
use_api = col1.toggle("Use local API (http://localhost:8000)", value=True)
score_bar = col2.toggle("Show score bar", value=True)

btn = st.button("Predict", type="primary")

API_URL = "http://host.docker.internal:8000/predict"  # works on Docker Desktop (Windows/Mac)

@st.cache_resource(show_spinner=False)
def get_local_pipeline():
    try:
        MODEL_DIR = "/app/artifacts/checkpoints/final"
        DEVICE = 0 if torch.cuda.is_available() else -1
        return pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, device=DEVICE, truncation=True)
    except Exception as e:
        st.warning(f"Pipeline fallback failed: {e}")
        return None

if btn and text.strip():
    if use_api:
        try:
            r = requests.post(API_URL, json={"text": text}, timeout=10)
            r.raise_for_status()
            data = r.json()
            label, score = data["label"], float(data["score"])
        except Exception as e:
            st.warning(f"API not reachable ({e}). Falling back to local pipeline.")
            pipe = get_local_pipeline()
            if pipe is None:
                st.error("No inference path available.")
            else:
                out = pipe(text)[0]
                label, score = out["label"], float(out["score"])
    else:
        pipe = get_local_pipeline()
        if pipe is None:
            st.error("Local pipeline not available.")
        else:
            out = pipe(text)[0]
            label, score = out["label"], float(out["score"])

    if 'label' in locals():
        st.markdown(f"### Prediction: `{label}`")
        if score_bar:
            st.progress(min(max(score, 0.0), 1.0), text=f"Confidence: {score:.3f}")
