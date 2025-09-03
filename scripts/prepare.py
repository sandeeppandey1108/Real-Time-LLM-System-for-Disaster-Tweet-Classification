import argparse, os, glob, pandas as pd
from sklearn.model_selection import train_test_split

TEXT_CANDIDATES = ["text","tweet","message","content","body","sentence"]
LABEL_CANDIDATES = ["target","label","labels","class","category"]

def _guess_text_col(cols):
    for c in TEXT_CANDIDATES:
        if c in cols: return c
    # fallback to first
    return cols[0]

def _guess_label_col(cols):
    for c in LABEL_CANDIDATES:
        if c in cols: return c
    return None

def _coerce_label(x):
    if pd.isna(x): return None
    if isinstance(x, (int,float)):
        try:
            v = int(x)
            if v in (0,1): return v
        except: pass
    s = str(x).strip().lower()
    if s in ("1","true","yes","y","disaster","positive","pos"): return 1
    if s in ("0","false","no","n","non_disaster","negative","neg","normal","other"): return 0
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="folder containing one or more CSV files")
    ap.add_argument("--out-dir", required=True, help="output folder")
    ap.add_argument("--val-size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csvs = sorted(glob.glob(os.path.join(args.raw_dir, "*.csv")))
    if not csvs:
        raise SystemExit(f"No CSV files found in {args.raw_dir}")

    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
        except Exception:
            # try with encoding issues
            df = pd.read_csv(p, encoding_errors="ignore")
        cols = [c.strip() for c in df.columns]
        df.columns = cols
        tcol = _guess_text_col(cols)
        lcol = _guess_label_col(cols)
        if lcol is None:
            # unlabeled (like test/submission) -> skip
            continue
        df = df[[tcol, lcol]].rename(columns={tcol:"text", lcol:"target"})
        # normalize labels
        df["target"] = df["target"].apply(_coerce_label)
        df = df.dropna(subset=["text","target"])
        frames.append(df)

    if not frames:
        raise SystemExit("No labeled rows found across CSVs (need a label column)")

    data = pd.concat(frames, ignore_index=True)
    # de-dup and basic cleanup
    data["text"] = data["text"].astype(str).str.strip()
    data = data.dropna(subset=["text","target"])
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # stratified split
    train_df, val_df = train_test_split(
        data, test_size=args.val_size, random_state=42, stratify=data["target"]
    )

    out_train = os.path.join(args.out_dir, "train.csv")
    out_val = os.path.join(args.out_dir, "val.csv")
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)

    print({"train": out_train, "val": out_val, "n_train": len(train_df), "n_val": len(val_df)})

if __name__ == "__main__":
    main()
