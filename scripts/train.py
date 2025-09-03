import argparse, os, json, numpy as np, pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import yaml
import torch

def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_df(path):
    df = pd.read_csv(path)
    # enforce expected columns
    if "text" not in df.columns or "target" not in df.columns:
        raise ValueError(f"{path} must have 'text' and 'target' columns")
    df = df.dropna(subset=["text","target"])
    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds))
    }

def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0,1]); ax.set_xticklabels(["non_disaster","disaster"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["non_disaster","disaster"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--eval-csv", required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    save_dir = Path(cfg.get("save_dir", "/app/artifacts/checkpoints/final"))
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_df(args.train_csv)
    eval_df  = load_df(args.eval_csv)

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=int(cfg.get("max_length", 192)))

    train_ds = Dataset.from_pandas(train_df)[["text","target"]].rename_column("target","labels").map(tok_fn, batched=True)
    eval_ds  = Dataset.from_pandas(eval_df)[["text","target"]].rename_column("target","labels").map(tok_fn, batched=True)

    collate = DataCollatorWithPadding(tokenizer=tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    targs = TrainingArguments(
        output_dir=str(save_dir),
        learning_rate=float(cfg.get("learning_rate", 5e-5)),
        per_device_train_batch_size=int(cfg.get("batch_size", 32)),
        per_device_eval_batch_size=int(cfg.get("batch_size", 32)),
        num_train_epochs=int(cfg.get("epochs", 3)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        fp16=bool(cfg.get("fp16", True)),
        bf16=bool(cfg.get("bf16", False)),
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collate,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()

    # predictions for confusion matrix
    preds = trainer.predict(eval_ds)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(-1)

    # save artifacts
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cm_path = save_dir.parent.parent / "confusion_matrix.png"  # /app/artifacts/confusion_matrix.png
    plot_confusion_matrix(y_true, y_pred, str(cm_path))

    # save model + tokenizer
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    # stamp readable labels
    from transformers import AutoConfig
    cfgm = AutoConfig.from_pretrained(str(save_dir))
    cfgm.id2label = {0:"non_disaster", 1:"disaster"}
    cfgm.label2id = {"non_disaster":0, "disaster":1}
    cfgm.save_pretrained(str(save_dir))

    out = {
        "metrics": metrics,
        "confusion_matrix": str(cm_path),
        "model_dir": str(save_dir)
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
