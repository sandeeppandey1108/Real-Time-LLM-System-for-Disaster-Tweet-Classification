
import re
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
from sklearn.model_selection import train_test_split

CLEAN_RE = re.compile(r"\s+")

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = None
    for cand in ["text","tweet","message","content"]:
        if cand in cols:
            text_col = cols[cand]; break
    if text_col is None:
        text_col = df.columns[0]
    label_col = None
    for cand in ["target","label","labels","y"]:
        if cand in cols:
            label_col = cols[cand]; break
    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    if label_col is not None:
        y = df[label_col]
        if y.dtype == bool: y = y.astype(int)
        out["target"] = y.map({"non_disaster":0,"disaster":1}).fillna(y)
    return out

def _clean_text(s: str) -> str:
    import re
    s = s.strip()
    s = re.sub(r"http\S+","", s)
    s = re.sub(r"@[A-Za-z0-9_]+","@user", s)
    s = re.sub(r"#","", s)
    s = CLEAN_RE.sub(" ", s)
    return s

def prepare(raw_dir: str="data/raw", out_dir: str="data/processed", val_size: float=0.2, seed: int=42) -> Tuple[str,str]:
    raw = Path(raw_dir); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    csvs = sorted(list(raw.glob("*.csv")))
    if not csvs: raise FileNotFoundError(f"No CSVs found in {raw.resolve()}")

    dfs_labeled, dfs_unlabeled = [], []
    for p in csvs:
        df = pd.read_csv(p)
        df = _normalize_columns(df)
        df["text"] = df["text"].astype(str).map(_clean_text)
        df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
        if "target" in df.columns and df["target"].notna().any():
            df_l = df.dropna(subset=["target"])
            df_l["target"] = df_l["target"].astype(int)
            dfs_labeled.append(df_l[["text","target"]])
        else:
            dfs_unlabeled.append(df[["text"]])

    if not dfs_labeled: raise ValueError("No labeled rows found.")

    import pandas as pd
    df_all = pd.concat(dfs_labeled, ignore_index=True)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df_all, test_size=val_size, random_state=seed, stratify=df_all["target"])
    train_out = out/"train.csv"; val_out = out/"val.csv"
    train_df.to_csv(train_out, index=False); val_df.to_csv(val_out, index=False)
    if dfs_unlabeled:
        pd.concat(dfs_unlabeled, ignore_index=True).to_csv(out/"test.csv", index=False)
    return str(train_out), str(val_out)
