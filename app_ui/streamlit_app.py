import os, requests, streamlit as st

st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨", layout="centered")
st.title("🚨 Disaster Tweet Classifier")

api = st.sidebar.text_input("API URL", value=os.environ.get("API_URL", "http://localhost:8000"))
st.sidebar.caption("Change if your API is running elsewhere")
text = st.text_area("Enter a tweet", height=120, placeholder="There is a fire downtown!")

if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter a tweet.")
    else:
        try:
            r = requests.post(f"{api}/predict", json={"text": text}, timeout=10)
            r.raise_for_status()
            out = r.json()
            st.success(f"**{out['label']}**  ·  score={out['score']:.4f}")
        except Exception as e:
            st.error(f"Request failed: {e}")

st.markdown("---")
st.caption("Tip: start the API with the PowerShell script, then open this page.")
