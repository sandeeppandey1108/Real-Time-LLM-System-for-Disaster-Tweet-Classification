import os
import requests
import streamlit as st

st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨")
st.title("🚨 Disaster Tweet Classifier")

api_url = os.environ.get("API_URL", "http://localhost:8000")
st.caption(f"API: {api_url}")

txt = st.text_area("Enter a tweet")
if st.button("Predict"):
    try:
        r = requests.post(f"{api_url}/predict", json={"text": txt}, timeout=15)
        r.raise_for_status()
        data = r.json()
        st.success(f"**{data['label']}** (score: {data['score']:.3f})")
    except Exception as e:
        st.error(str(e))
