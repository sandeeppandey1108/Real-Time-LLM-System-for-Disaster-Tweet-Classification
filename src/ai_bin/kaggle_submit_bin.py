
import os, argparse, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="data/raw/test.csv")
    ap.add_argument("--out", required=True, help="output path for submission.csv")
    ap.add_argument("--model_dir", default=os.environ.get("MODEL_DIR","artifacts/checkpoints/final"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    df = pd.read_csv(args.test)
    texts = (df["text"].fillna("").astype(str)).tolist()
    preds = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i+64]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            probs = model(**enc).logits.softmax(-1)[:,1].detach().cpu().numpy()
        preds.extend((probs >= 0.5).astype(int).tolist())
    out = pd.DataFrame({"id": df["id"].astype(int), "target": preds})
    out.to_csv(args.out, index=False)
    print({"wrote": args.out, "n": len(out)})

if __name__ == "__main__":
    main()
