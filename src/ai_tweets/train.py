
import json, os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, set_seed)
import yaml

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)

def train(config: str, train_csv: str, eval_csv: Optional[str]=None, text_col: Optional[str]=None) -> Dict[str, Any]:
    cfg = _load_cfg(config); set_seed(cfg.get("seed",42))
    art_dir = Path("/app/artifacts") if Path("/app").exists() else Path("artifacts"); _ensure_dir(art_dir)
    out_dir = art_dir/"checkpoints"/"final"; _ensure_dir(out_dir)

    df_train = pd.read_csv(train_csv)
    if eval_csv is None: raise ValueError("eval_csv is required")
    df_eval = pd.read_csv(eval_csv)

    text_col = text_col or "text"; y_col = "target"; num_labels = 2
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])

    def tok_fn(batch): return tok(batch[text_col], truncation=True, max_length=cfg.get("max_length",192))
    ds_train = Dataset.from_pandas(df_train[[text_col, y_col]]).map(tok_fn, batched=True).rename_column(y_col,"labels")
    ds_eval  = Dataset.from_pandas(df_eval[[text_col, y_col]]).map(tok_fn, batched=True).rename_column(y_col,"labels")

    cols = ["input_ids","attention_mask","labels"] + (["token_type_ids"] if "token_type_ids" in ds_train.features else [])
    ds_train.set_format(type="torch", columns=cols); ds_eval.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=num_labels)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.get("batch_size",32),
        per_device_eval_batch_size=cfg.get("batch_size",32),
        num_train_epochs=cfg.get("epochs",3),
        learning_rate=float(cfg.get("learning_rate",5e-5)),
        weight_decay=cfg.get("weight_decay",0.01),
        warmup_ratio=cfg.get("warmup_ratio",0.1),
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=cfg.get("logging_steps",10),
        fp16=bool(cfg.get("fp16",True)) and torch.cuda.is_available(),
        bf16=bool(cfg.get("bf16",False)),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

    trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_eval, tokenizer=tok)
    trainer.train(); eval_metrics = trainer.evaluate()
    trainer.save_model(str(out_dir)); tok.save_pretrained(str(out_dir))

    from transformers import AutoConfig
    cfgm = AutoConfig.from_pretrained(str(out_dir))
    cfgm.id2label = {0:"non_disaster",1:"disaster"}
    cfgm.label2id = {"non_disaster":0,"disaster":1}
    cfgm.save_pretrained(str(out_dir))

    preds = np.argmax(trainer.predict(ds_eval).predictions, axis=-1)
    cm = confusion_matrix(df_eval[y_col].values, preds, labels=[0,1])
    fig = plt.figure(figsize=(4,4)); ax = plt.gca(); ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion matrix"); ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["non_disaster","disaster"]); ax.set_yticklabels(["non_disaster","disaster"])
    for (i,j), v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); cm_path = art_dir / "confusion_matrix.png"; fig.tight_layout(); fig.savefig(cm_path, dpi=160); plt.close(fig)

    metrics = {"metrics": eval_metrics, "confusion_matrix": str(cm_path), "model_dir": str(out_dir)}
    (art_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
