
import os, requests, streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Disaster Tweet Classifier (Binary)", layout="centered")
st.title("🌪️ Disaster Tweet Classifier — Real vs Fake")

txt = st.text_area("Enter a tweet:", height=160, placeholder="Wildfire near the highway, evacuations underway")
if st.button("Predict"):
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": txt}, timeout=10)
        if r.ok:
            data = r.json()
            st.success(f"Prediction: **{data['label']}** (P(real)={data['score']:.4f})")
        else:
            st.error(f"API error: {r.status_code} {r.text}")
    except Exception as e:
        st.error(str(e))
