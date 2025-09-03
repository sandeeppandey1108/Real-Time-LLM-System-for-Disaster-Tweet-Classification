from __future__ import annotations
import os, re, html, json
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")

def _read_csvs(folder: Path) -> List[pd.DataFrame]:
    dfs = []
    for name in ["train.csv", "val.csv", "valid.csv", "dev.csv", "test.csv", "submission.csv"]:
        p = folder / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                df["__source"] = name
                dfs.append(df)
            except Exception:
                pass
    return dfs

def _detect_cols(df: pd.DataFrame, text_col: Optional[str], label_col: Optional[str]):
    tcol = text_col or ("text" if "text" in df.columns else df.columns[0])
    lcol = None
    for cand in ["target", "label", "labels", "y"]:
        if cand in df.columns:
            lcol = cand
            break
    if label_col:
        lcol = label_col
    return tcol, lcol

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = html.unescape(s)
    s = URL_RE.sub(" ", s)
    s = s.replace("&amp;", "&")
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

def prepare(raw_dir: str, out_dir: str, text_col: Optional[str]=None, label_col: Optional[str]=None, val_size: float=0.2, seed: int=42):
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dfs = _read_csvs(raw)
    if not dfs:
        return str(out / "train.csv"), str(out / "val.csv")

    labeled = []
    for df in dfs:
        tcol, lcol = _detect_cols(df, text_col, label_col)
        if lcol is None:
            continue
        sub = df[[tcol, lcol]].rename(columns={tcol: "text", lcol: "target"}).copy()
        labeled.append(sub)

    if not labeled:
        raise ValueError("No labeled CSV with a target/label column was found in raw dir.")

    data = pd.concat(labeled, axis=0, ignore_index=True)
    data["text"] = data["text"].astype(str).map(_clean_text)

    if data["target"].dtype == "object":
        data["target"] = data["target"].str.strip().str.lower().map({"non_disaster":0,"disaster":1}).fillna(data["target"])
    data["target"] = pd.to_numeric(data["target"], errors="coerce").fillna(0).astype(int).clip(0,1)

    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

    train_df, val_df = train_test_split(
        data, test_size=val_size, random_state=seed, stratify=data["target"] if data["target"].nunique()>1 else None
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_path = out / "train.csv"
    val_path = out / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    return str(train_path), str(val_path)
