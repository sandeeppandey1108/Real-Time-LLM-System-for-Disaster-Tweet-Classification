
import re, pandas as pd, numpy as np
from typing import Tuple

URL_RE = re.compile(r'https?://\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#\w+')
WS_RE = re.compile(r'\s+')

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = URL_RE.sub("", s)
    s = MENTION_RE.sub("", s)
    s = HASHTAG_RE.sub("", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def prepare_binary(df: pd.DataFrame, use_keyword_location: bool=True, lowercase: bool=True) -> pd.DataFrame:
    df = df.copy()
    for col in ["text","keyword","location"]:
        if col not in df.columns:
            df[col] = ""
    txt = df["text"].fillna("")
    if use_keyword_location:
        kw = df["keyword"].fillna("")
        loc = df["location"].fillna("")
        addon = []
        for k,l in zip(kw, loc):
            parts = []
            if k: parts.append(f"[kw={k}]")
            if l: parts.append(f"[loc={l}]")
            addon.append(" ".join(parts))
        txt = (txt + " " + pd.Series(addon)).str.strip()
    txt = txt.apply(clean_text)
    if lowercase:
        txt = txt.str.lower()
    df_out = pd.DataFrame({"text": txt})
    if "target" in df.columns:
        df_out["label"] = df["target"].astype(int)
    return df_out

def stratified_split(df: pd.DataFrame, test_size: float=0.2, seed: int=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Manual stratified split to avoid sklearn dependency in runtime environment
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    parts = []
    for lbl, sub in df.groupby("label"):
        idx = sub.index.to_numpy()
        rng.shuffle(idx)
        cut = int(round(len(idx) * (1 - test_size)))
        parts.append((sub.loc[idx[:cut]], sub.loc[idx[cut:]]))
    train = pd.concat([a for a,_ in parts], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val   = pd.concat([b for _,b in parts], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val
