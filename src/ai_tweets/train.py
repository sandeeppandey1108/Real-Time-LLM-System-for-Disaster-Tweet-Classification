from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import json, pandas as pd

from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from sklearn.metrics import accuracy_score, f1_score

from .utils import load_config, set_seed, ensure_parent

def train(config_path: str | None = None, train_csv: str = "data/train.csv", text_col: str | None = None) -> Dict[str, Any]:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    df = pd.read_csv(train_csv)
    text_col = text_col or ("text" if "text" in df.columns else ("tweet" if "tweet" in df.columns else None))
    assert text_col is not None, "CSV must include 'text' or 'tweet'"
    assert "target" in df.columns, "CSV must include 'target' 0/1"
    df[text_col] = df[text_col].astype(str)

    n_eval = max(1, int(len(df) * 0.1))
    df_train = df.iloc[:-n_eval].reset_index(drop=True)
    df_eval  = df.iloc[-n_eval:].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    def tok_fn(batch): return tok(batch[text_col], truncation=True, max_length=int(cfg["max_length"]))
    collator = DataCollatorWithPadding(tokenizer=tok)

    ds_train = Dataset.from_pandas(df_train[[text_col, "target"]]).map(tok_fn, batched=True)
    ds_eval  = Dataset.from_pandas(df_eval[[text_col, "target"]]).map(tok_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2)

    args = TrainingArguments(
        output_dir=str(cfg["output_dir"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["batch_size"]),
        num_train_epochs=int(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

    tr = Trainer(
        model=model, args=args,
        train_dataset=ds_train, eval_dataset=ds_eval,
        tokenizer=tok, data_collator=collator, compute_metrics=compute_metrics
    )

    tr.train()
    metrics = tr.evaluate()

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    tr.save_model(str(cfg["output_dir"])); tok.save_pretrained(str(cfg["output_dir"]))

    ensure_parent(cfg["metrics_path"]).write_text(json.dumps(metrics, indent=2))
    return metrics
