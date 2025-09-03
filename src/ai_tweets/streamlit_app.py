import os, requests, json
import streamlit as st

st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨", layout="centered")
st.title("🚨 Disaster Tweet Classifier")
st.caption("FastAPI + Streamlit · Multilingual MiniLM")

API_URL = os.environ.get("API_URL", "http://localhost:8000")

with st.sidebar:
    st.markdown("**API**")
    st.write(API_URL)
    if st.button("Health check"):
        try:
            resp = requests.get(f"{API_URL}/healthz", timeout=10).json()
            st.success(resp)
        except Exception as e:
            st.error(f"Health check failed: {e}")

txt = st.text_area("Enter a tweet", "There is a fire downtown!", height=120)
if st.button("Classify"):
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": txt}, timeout=30)
        r.raise_for_status()
        data = r.json()
        label = data.get("label")
        score = data.get("score")
        st.metric("Prediction", label, delta=f"{score:.4f}")
        st.progress(min(max(score, 0.0), 1.0))
    except Exception as e:
        st.error(f"Request failed: {e}")
