from __future__ import annotations
from pathlib import Path
import yaml, random, numpy as np

def load_config(path: str | Path | None = None) -> dict:
    p = Path(path or "configs/default.yaml")
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_parent(path: str | Path) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); return p
