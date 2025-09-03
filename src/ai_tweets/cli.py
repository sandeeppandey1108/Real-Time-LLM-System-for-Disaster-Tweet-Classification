import typer
from rich import print as rprint
from typing import Optional
from .prepare import prepare as prepare_fn
from .train import train as train_fn

app = typer.Typer(help="Crisis LLM Pro CLI")

@app.command(help="Prepare data: merge/clean/split")
def prepare(
    raw_dir: str = typer.Option("data/raw", help="Raw CSV folder"),
    out_dir: str = typer.Option("data/processed", help="Output folder for processed CSVs"),
    text_col: Optional[str] = typer.Option(None, help="Name of text column if not 'text'"),
    label_col: Optional[str] = typer.Option(None, help="Name of label column if not 'target'/'label'"),
    val_size: float = typer.Option(0.2, help="Validation split size"),
    seed: int = typer.Option(42, help="Random seed"),
):
    paths = prepare_fn(raw_dir, out_dir, text_col=text_col, label_col=label_col, val_size=val_size, seed=seed)
    rprint({"train": paths[0], "val": paths[1]})

@app.command(help="Train + evaluate")
def train(
    config: str = typer.Option("configs/gpu.yaml", help="YAML config"),
    train_csv: str = typer.Option("data/train.csv"),
    eval_csv: Optional[str] = typer.Option(None),
    text_col: Optional[str] = typer.Option(None, help="Text column (auto-detect)"),
):
    metrics = train_fn(config, train_csv, eval_csv=eval_csv, text_col=text_col)
    rprint(metrics)

if __name__ == "__main__":
    app()
