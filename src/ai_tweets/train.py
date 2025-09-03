from __future__ import annotations
import json, random, os
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import yaml

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _load_csv_guess_cols(path: str, text_col: Optional[str]):
    df = pd.read_csv(path)
    if text_col is None:
        text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "target" if "target" in df.columns else ("label" if "label" in df.columns else None)
    if label_col is None:
        raise ValueError(f"No 'target' or 'label' column found in {path}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "labels"})
    return df, "text", "labels"

def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def _make_datasets(df_train: pd.DataFrame, df_eval: pd.DataFrame, tok, max_length: int):
    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    ds_train = Dataset.from_pandas(df_train)
    ds_eval = Dataset.from_pandas(df_eval)
    ds_train = ds_train.map(tokenize, batched=True)
    ds_eval = ds_eval.map(tokenize, batched=True)
    ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c not in ["input_ids","attention_mask","labels","text"]])
    ds_eval = ds_eval.remove_columns([c for c in ds_eval.column_names if c not in ["input_ids","attention_mask","labels","text"]])
    ds_train.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    ds_eval.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    return ds_train, ds_eval

def _metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

def _plot_confusion(y_true, y_pred, out_path: Path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks([0,1], ["non_disaster","disaster"])
    plt.yticks([0,1], ["non_disaster","disaster"])
    for (i,j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def train(config: str, train_csv: str, eval_csv: Optional[str]=None, text_col: Optional[str]=None) -> Dict[str, Any]:
    cfg = _load_config(config)
    art_dir = Path("/app/artifacts")
    _ensure_dir(art_dir)

    df_train, _, _ = _load_csv_guess_cols(train_csv, text_col)
    if eval_csv:
        df_eval, _, _ = _load_csv_guess_cols(eval_csv, text_col)
    else:
        # quick split if no eval provided
        from sklearn.model_selection import train_test_split
        df_train, df_eval = train_test_split(df_train, test_size=0.2, random_state=cfg.get("seed", 42), stratify=df_train["labels"])

    id2label = {0:"non_disaster", 1:"disaster"}
    label2id = {"non_disaster":0, "disaster":1}

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2, id2label=id2label, label2id=label2id)

    ds_train, ds_eval = _make_datasets(df_train, df_eval, tok, cfg.get("max_length", 192))

    out_dir = art_dir / "checkpoints" / "final"
    _ensure_dir(out_dir)

    args_kwargs = dict(
        output_dir=str(art_dir / "checkpoints" / "tmp"),
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 32),
        per_device_eval_batch_size=min(64, cfg.get("batch_size", 32)*2),
        learning_rate=float(cfg.get("learning_rate", 5e-5)),
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        fp16=bool(cfg.get("fp16", True)),
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to=[],
    )
    try:
        args = TrainingArguments(evaluation_strategy="epoch", save_strategy="no", **args_kwargs)
    except TypeError:
        args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tok,
        compute_metrics=_metrics_fn,
    )
    trainer.train()
    eval_out = trainer.evaluate(ds_eval)

    import numpy as np
    y_true = np.array(ds_eval["labels"])
    preds = np.argmax(trainer.predict(ds_eval).predictions, axis=-1)
    _plot_confusion(y_true, preds, art_dir / "confusion_matrix.png")

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    return {
        "metrics": eval_out,
        "confusion_matrix": str(art_dir / "confusion_matrix.png"),
        "model_dir": str(out_dir),
    }
