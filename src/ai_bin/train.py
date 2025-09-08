
import os, json, numpy as np, pandas as pd, torch, inspect, warnings
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
try:
    from transformers import EarlyStoppingCallback
    HAVE_ES = True
except Exception:
    HAVE_ES = False

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

@dataclass
class TrainCfg:
    seed:int=42
    model_name:str="microsoft/Multilingual-MiniLM-L12-H384"
    max_length:int=128
    batch_size:int=16
    epochs:int=3
    learning_rate:float=5.0e-5
    weight_decay:float=0.01
    warmup_ratio:float=0.06
    early_stopping_patience:int=10
    early_stopping_threshold:float=1e-4
    monitor_metric:str="f1"
    greater_is_better:bool=True

def load_cfg(yaml_path:str)->TrainCfg:
    import yaml
    with open(yaml_path,"r",encoding="utf-8") as f:
        y=yaml.safe_load(f)
    return TrainCfg(**y)

def _read(csv):
    df=pd.read_csv(csv)
    if not {'text','label'}.issubset(df.columns):
        raise ValueError(f"{csv} must have text,label")
    return df

def _plot_cm(y_true,y_pred,out_path, labels=("fake","real")):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    fig,ax=plt.subplots(figsize=(5.2,4.4))
    im=ax.imshow(cm,cmap="viridis")
    for (i,j),z in np.ndenumerate(cm):
        ax.text(j, i, str(z), ha='center', va='center', color='white' if z>cm.max()/2 else 'black')
    ax.set_xticks([0,1], labels)
    ax.set_yticks([0,1], labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix (fake vs real)")
    fig.tight_layout(); fig.savefig(out_path, dpi=170); plt.close(fig)

class DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc=enc; self.lab=labels.values.astype(int)
    def __len__(self): return len(self.lab)
    def __getitem__(self, i):
        d={k:torch.tensor(v[i]) for k,v in self.enc.items()}
        d['labels']=torch.tensor(self.lab[i]).long()
        return d

def _build_training_args(cfg, out_dir):
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    kwargs = dict(
        output_dir=os.path.join(out_dir,"runs"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=50,
    )
    # Grouped handling: only set eval/save/best if ALL are supported by this version.
    supports_eval = "evaluation_strategy" in allowed
    supports_save = "save_strategy" in allowed
    supports_best = "load_best_model_at_end" in allowed and "metric_for_best_model" in allowed and "greater_is_better" in allowed
    if supports_eval and supports_save and supports_best:
        kwargs["evaluation_strategy"] = "epoch"
        kwargs["save_strategy"] = "epoch"
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = cfg.monitor_metric
        kwargs["greater_is_better"] = cfg.greater_is_better
        if "save_total_limit" in allowed: kwargs["save_total_limit"]=2
    # reporting can be empty safely
    if "report_to" in allowed: kwargs["report_to"]=[]
    # Optional: gradient clipping if supported
    if "max_grad_norm" in allowed: kwargs["max_grad_norm"]=1.0
    kwargs = {k:v for k,v in kwargs.items() if k in allowed}
    return TrainingArguments(**kwargs)

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p,r,f1,_ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def train_main(cfg_path:str, train_csv:str, eval_csv:str, out_dir:str="artifacts"):
    cfg=load_cfg(cfg_path)
    if cfg.learning_rate >= 1e-3:
        warnings.warn(f"Learning rate {cfg.learning_rate} is very high for transformer fine-tuning; monitor loss.", RuntimeWarning)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    tr=_read(train_csv); ev=_read(eval_csv)
    num_labels=2

    local_dir=os.environ.get("MODEL_DIR","")
    local_only=bool(local_dir and os.path.isdir(local_dir))
    src = local_dir if local_only else cfg.model_name
    tok=AutoTokenizer.from_pretrained(src, local_files_only=local_only)
    model=AutoModelForSequenceClassification.from_pretrained(src, num_labels=num_labels, local_files_only=local_only)

    def enc(df): return tok(df['text'].tolist(), padding=True, truncation=True, max_length=cfg.max_length)
    tr_enc, ev_enc = enc(tr), enc(ev)

    args=_build_training_args(cfg, out_dir)
    callbacks=[]
    # Only attach early stopping if evaluation happens during training
    if HAVE_ES and getattr(args, "evaluation_strategy", "no") != "no":
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience,
                                               early_stopping_threshold=cfg.early_stopping_threshold))

    trainer=Trainer(model=model, args=args,
                    train_dataset=DS(tr_enc, tr['label']),
                    eval_dataset=DS(ev_enc, ev['label']),
                    tokenizer=tok,
                    compute_metrics=_compute_metrics,
                    callbacks=callbacks)
    trainer.train()
    metrics = trainer.evaluate()
    preds = trainer.predict(DS(ev_enc, ev['label'])).predictions.argmax(-1)

    os.makedirs(out_dir, exist_ok=True)
    cm_path = os.path.join(out_dir, "confusion_matrix_bin.png")
    _plot_cm(ev['label'].values, preds, cm_path, labels=("fake","real"))

    ckpt = os.path.join(out_dir, "checkpoints", "final")
    os.makedirs(ckpt, exist_ok=True)
    tok.save_pretrained(ckpt)
    trainer.model.save_pretrained(ckpt)

    rep={"metrics":metrics, "confusion_matrix":cm_path, "model_dir":ckpt, "num_labels":num_labels}
    with open(os.path.join(out_dir,"train_report.json"),"w",encoding="utf-8") as f:
        json.dump(rep,f,indent=2)
    print(json.dumps(rep, indent=2))
    return rep

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    a = ap.parse_args()
    train_main(a.config, a.train_csv, a.eval_csv, a.out_dir)
