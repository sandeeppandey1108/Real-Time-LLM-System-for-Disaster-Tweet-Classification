from pathlib import Path
from typing import Optional, Dict, Any
import json, yaml
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _load_csv_guess_cols(path: str, text_col: Optional[str]):
    df = pd.read_csv(path)
    if text_col is None:
        text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "target" if "target" in df.columns else ("label" if "label" in df.columns else None)
    return df, text_col, label_col

def train(config: str, train_csv: str, eval_csv: Optional[str]=None, text_col: Optional[str]=None) -> Dict[str, Any]:
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    art_dir = Path("/app/artifacts")
    _ensure_dir(art_dir)

    df_train, text_col, y_col = _load_csv_guess_cols(train_csv, text_col)
    if eval_csv:
        df_eval, _, _ = _load_csv_guess_cols(eval_csv, text_col)
    else:
        # if no eval provided, do a quick 80/20 split
        n_eval = max(1, int(0.2*len(df_train)))
        df_eval = df_train.iloc[-n_eval:].reset_index(drop=True)
        df_train = df_train.iloc[:-n_eval].reset_index(drop=True)

    # Tokenizer & model
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    def tok_fn(batch):
        return tok(batch[text_col], truncation=True, max_length=cfg.get("max_length", 192))

    label2id = {"non_disaster":0, "disaster":1}
    id2label = {0:"non_disaster", 1:"disaster"}
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2, id2label=id2label, label2id=label2id
    )

    # Datasets
    train_ds = Dataset.from_pandas(df_train[[text_col, y_col]].rename(columns={text_col:"text", y_col:"label"}))
    eval_ds  = Dataset.from_pandas(df_eval[[text_col, y_col]].rename(columns={text_col:"text", y_col:"label"}))
    train_ds = train_ds.map(tok_fn, batched=True)
    eval_ds  = eval_ds.map(tok_fn, batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    eval_ds  = eval_ds.rename_column("label", "labels")
    train_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    eval_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    # Training
    out_dir = art_dir / "checkpoints" / "final"
    _ensure_dir(out_dir)

    args = TrainingArguments(
        output_dir=str(art_dir / "checkpoints" / "tmp"),
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 32),
        per_device_eval_batch_size=cfg.get("batch_size", 32),
        learning_rate=float(cfg.get("learning_rate", 5e-5)),
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        fp16=bool(cfg.get("fp16", False)),
        bf16=bool(cfg.get("bf16", False)),
        evaluation_strategy="epoch",
        logging_strategy="steps",
        save_strategy="no",
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="binary")
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # final metrics
    eval_metrics = trainer.evaluate()

    # Save model
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # Confusion matrix
    preds = np.argmax(trainer.predict(eval_ds).predictions, axis=-1)
    cm = confusion_matrix(eval_ds["labels"], preds, labels=[0,1])
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks([0,1], ["non_disaster", "disaster"])
    plt.yticks([0,1], ["non_disaster", "disaster"])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True")
    (art_dir / "confusion_matrix.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(art_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # Metrics JSON
    metrics = {
        "metrics": eval_metrics,
        "confusion_matrix": str(art_dir / "confusion_matrix.png"),
        "model_dir": str(out_dir),
    }
    with open(art_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
