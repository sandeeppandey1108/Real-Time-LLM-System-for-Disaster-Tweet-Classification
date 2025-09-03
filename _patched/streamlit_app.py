import os, torch, streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/artifacts/checkpoints/final")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

@st.cache_resource
def load_pipeline():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    device = 0 if torch.cuda.is_available() else -1
    return TextClassificationPipeline(model=mdl, tokenizer=tok, device=device, truncation=True, return_all_scores=True)

st.title("Disaster Tweet Classifier")
st.caption("Fast demo using your fine-tuned checkpoint")

clf = load_pipeline()
text = st.text_area("Tweet text", "")
if st.button("Predict") and text.strip():
    result = clf(text)[0]
    st.write(result)
