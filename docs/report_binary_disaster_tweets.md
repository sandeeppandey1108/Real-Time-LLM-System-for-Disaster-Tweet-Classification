
# Real vs Fake Disaster Tweet Classifier — Clean Project (Full)

- **Goal:** classify tweets as **real (1)** or **fake/not-real (0)** for crisis monitoring.
- **Dataset:** Kaggle NLP Getting Started (train/test/sample_submission).
- **Model:** microsoft/Multilingual-MiniLM-L12-H384 with 2-class head.
- **Serving:** FastAPI `/predict`, Streamlit UI.
- **Training:** Hugging Face Trainer, early stopping (patience=10).

See `configs/bin.yaml` (baseline) and `configs/bin_es.yaml` (your 1000 epochs, lr=0.001).
