from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WHITESPACE_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    # Keep hashtag words without '#'
    s = HASHTAG_RE.sub(r"\1", s)
    s = WHITESPACE_RE.sub(" ", s)
    return s.lower().strip()

def load_many_csv(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source"] = p.name
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    if not frames:
        raise FileNotFoundError("No CSVs found to load.")
    return pd.concat(frames, ignore_index=True)

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Accept either numeric targets (0/1) or labels ("disaster"/"non_disaster"/"Not Disaster", etc.)
    if "target" not in df.columns:
        # Try 'label' -> target
        if "label" in df.columns:
            df = df.rename(columns={"label": "target"})
        else:
            # If there's no label column, drop rows (e.g., Kaggle test.csv)
            df["target"] = None

    # Map string labels if present
    def map_val(v):
        if pd.isna(v):
            return None
        if isinstance(v, (int, float)) and v in (0, 1):
            return int(v)
        s = str(v).strip().lower()
        if s in ("1", "disaster", "true", "yes", "y"):
            return 1
        if s in ("0", "non_disaster", "non-disaster", "not disaster", "false", "no", "n"):
            return 0
        # unknown -> None
        return None

    df["target"] = df["target"].map(map_val)
    return df

def prepare_data(input_glob: str, out_train: str, out_val: str, val_size: float=0.2, seed: int=42) -> Tuple[str,str]:
    data_dir = Path(".")
    paths = sorted(Path(".").glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched '{input_glob}'")

    df = load_many_csv(paths)
    # Choose text column
    text_col = "text" if "text" in df.columns else df.columns[0]
    df = df.rename(columns={text_col: "text"})[["text"] + [c for c in df.columns if c != "text"]]

    # Clean + normalize labels
    df["text"] = df["text"].map(clean_text)
    df = normalize_labels(df)

    # Only rows with labels go to training
    df_labeled = df.dropna(subset=["target"]).copy()
    df_labeled["target"] = df_labeled["target"].astype(int)

    # Remove empties/dupes
    df_labeled = df_labeled[df_labeled["text"].str.len() > 0]
    df_labeled = df_labeled.drop_duplicates(subset=["text", "target"])

    if len(df_labeled) < 4:
        print(f"[WARN] Very small labeled dataset ({len(df_labeled)} rows).")
    # Stratified split (fallback if class imbalance)
    try:
        tr, va = train_test_split(df_labeled, test_size=val_size, random_state=seed, stratify=df_labeled["target"])
    except Exception:
        tr, va = train_test_split(df_labeled, test_size=val_size, random_state=seed)

    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(out_val).parent.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out_train, index=False)
    va.to_csv(out_val, index=False)

    # Save also full merged (for reference)
    merged_path = Path(out_train).parent/"merged_all.csv"
    df.to_csv(merged_path, index=False)

    return str(Path(out_train)), str(Path(out_val))
