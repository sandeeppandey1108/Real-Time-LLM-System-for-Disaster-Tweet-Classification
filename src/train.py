import os, json, math, argparse, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, set_seed)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def df_to_ds(df, text_col="text", label_col="target"):
    return Dataset.from_pandas(df[[text_col, label_col]].rename(columns={text_col:"text", label_col:"labels"}))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }

def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig = plt.figure(figsize=(3.5,3.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpu.yaml")
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--eval-csv", default="data/processed/val.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = cfg["model_name"]
    max_length = int(cfg.get("max_length", 192))

    # Load data
    df_train = pd.read_csv(args.train_csv)
    df_eval = pd.read_csv(args.eval_csv)

    tok = AutoTokenizer.from_pretrained(model_name)
    def tok_batch(examples):
        return tok(examples["text"], truncation=True, max_length=max_length)

    ds_train = df_to_ds(df_train).map(tok_batch, batched=True)
    ds_eval  = df_to_ds(df_eval ).map(tok_batch, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    out_dir = Path(args.artifacts) / "checkpoints" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(Path(args.artifacts) / "checkpoints" / "tmp"),
        per_device_train_batch_size=int(cfg.get("batch_size", 32)),
        per_device_eval_batch_size=int(cfg.get("batch_size", 32)),
        num_train_epochs=float(cfg.get("epochs", 3)),
        learning_rate=float(cfg.get("learning_rate", 5e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        fp16=bool(cfg.get("fp16", True)) and torch.cuda.is_available(),
        report_to=[],
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    # Predictions for CM
    preds = np.argmax(trainer.predict(ds_eval).predictions, axis=-1)
    cm_path = Path(args.artifacts) / "confusion_matrix.png"
    plot_confusion_matrix(df_eval["target"].to_numpy(), preds, cm_path)

    # Save final
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # Save metrics
    (Path(args.artifacts) / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps({
        "metrics": metrics,
        "confusion_matrix": str(cm_path),
        "model_dir": str(out_dir)
    }, indent=2))

if __name__ == "__main__":
    main()
