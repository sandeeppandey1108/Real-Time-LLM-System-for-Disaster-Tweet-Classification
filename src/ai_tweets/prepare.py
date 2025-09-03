from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Any

LABELS = {"non_disaster": 0, "disaster": 1}

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    # find text col
    text_col = "text" if "text" in df.columns else cols[0]
    # find label col
    label_col = "target" if "target" in df.columns else ("label" if "label" in df.columns else None)
    if label_col is None:
        # return empty to signal unlabeled
        return pd.DataFrame(columns=["text", "target"])
    out = pd.DataFrame({
        "text": df[text_col].astype(str).str.strip(),
        "target": df[label_col],
    })
    # map string labels if any
    if out["target"].dtype == "object":
        out["target"] = out["target"].str.lower().map(LABELS)
    out = out.dropna(subset=["text", "target"])
    out = out[out["text"].str.len() > 0]
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    # coerce to int 0/1 if possible
    out["target"] = out["target"].astype(int)
    out = out[out["target"].isin([0,1])]
    return out

def prepare(raw_dir: str, out_dir: str, val_ratio: float=0.2) -> Dict[str, Any]:
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dfs = []
    for p in raw.glob("*.csv"):
        try:
            df = pd.read_csv(p)
            df = _normalize_df(df)
            if len(df):
                dfs.append(df)
        except Exception as e:
            print(f"Skipping {p.name}: {e}")
    if not dfs:
        raise FileNotFoundError(f"No labeled rows found in {raw.resolve()}")

    all_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Stratified split
    train_df, val_df = train_test_split(all_df, test_size=val_ratio, random_state=42, stratify=all_df["target"])

    train_path = out / "train.csv"
    val_path = out / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    return {
        "raw_dir": str(raw.resolve()),
        "out_dir": str(out.resolve()),
        "n_all": len(all_df),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "train_csv": str(train_path),
        "val_csv": str(val_path),
    }
