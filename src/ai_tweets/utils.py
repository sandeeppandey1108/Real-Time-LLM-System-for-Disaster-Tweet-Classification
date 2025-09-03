from pathlib import Path
import re

def ensure_dir(p: str | Path) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp

def clean_text(s: str) -> str:
    s = s or ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s
