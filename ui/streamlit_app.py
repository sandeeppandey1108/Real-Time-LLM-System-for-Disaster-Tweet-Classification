import os
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://host.docker.internal:8000")

st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨", layout="centered")

st.title("🚨 Disaster Tweet Classifier")
st.caption("Transformers • FastAPI • CUDA")

with st.expander("Service status", expanded=False):
    try:
        r = requests.get(f"{API_BASE}/healthz", timeout=5)
        st.json(r.json())
    except Exception as e:
        st.error(f"Health check failed: {e}")

txt = st.text_area("Enter a tweet", height=120, placeholder="There is a fire downtown!")
if st.button("Predict", type="primary"):
    if not txt.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Calling API..."):
            try:
                r = requests.post(f"{API_BASE}/predict", json={"text": txt}, timeout=20)
                r.raise_for_status()
                out = r.json()
                st.success(f"**{out['label']}** (score: {out['score']:.4f})")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Metrics available at `/metrics` on the API.")
