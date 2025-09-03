import os, glob, re, json, sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path(os.environ.get("RAW_DIR", "/app/data"))
OUT_DIR = RAW_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_col(cands, cols):
    cols_lower = {c.lower(): c for c in cols}
    for c in cands:
        if c in cols_lower:
            return cols_lower[c]
    return None

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # heuristics to find text/label columns
    text_col = find_col(["text","tweet","message","content","body"], df.columns) or df.columns[0]
    label_col = None
    # typical label names
    label_col = find_col(["target","label","is_disaster","is_disaster_tweet"], df.columns)
    if label_col is None and "target" not in df.columns and "label" not in df.columns:
        # unlabeled set (e.g., test.csv) -> ignore for training
        return pd.DataFrame(columns=["text","target"])

    out = pd.DataFrame({
        "text": df[text_col].astype(str),
        "target": df[label_col].astype(int) if label_col else pd.Series(dtype="int"),
    })
    return out

def main():
    csvs = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not csvs:
        print(f"No CSVs found in {RAW_DIR}. Put your train/val/test/submission CSVs there.", file=sys.stderr)
        sys.exit(1)

    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Skip {p}: {e}", file=sys.stderr)
            continue
        norm = normalize_df(df)
        if not norm.empty:
            frames.append(norm)

    if not frames:
        print("No labeled rows found across CSVs. Ensure at least one CSV has a 'text' + 'target' column.", file=sys.stderr)
        sys.exit(2)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["text"]).drop_duplicates(subset=["text","target"])

    # Ensure labels are 0/1
    all_df["target"] = all_df["target"].astype(int)
    if not set(all_df["target"].unique()).issubset({0,1}):
        # try to coerce common strings
        mapping = {"disaster":1,"non_disaster":0,"not_disaster":0,"negative":0,"positive":1,"true":1,"false":0}
        all_df["target"] = all_df["target"].map(lambda x: mapping.get(str(x).strip().lower(), x)).astype(int)

    test_size = float(os.environ.get("TEST_SIZE", "0.2"))
    stratify = all_df["target"] if len(all_df["target"].unique()) > 1 else None
    train_df, val_df = train_test_split(all_df, test_size=test_size, random_state=42, stratify=stratify)

    train_out = OUT_DIR / "train.csv"
    val_out = OUT_DIR / "val.csv"
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    print(json.dumps({
        "input_csvs": csvs,
        "rows_total": len(all_df),
        "rows_train": len(train_df),
        "rows_val": len(val_df),
        "train_csv": str(train_out),
        "val_csv": str(val_out),
    }, indent=2))

if __name__ == "__main__":
    main()
