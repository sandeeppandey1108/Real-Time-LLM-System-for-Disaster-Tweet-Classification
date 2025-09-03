import typer
from typing import Optional
from .prepare import prepare as prepare_fn
from .train import train as train_fn
from rich import print as rprint

cli = typer.Typer(help="AI Tweets — Disaster tweet classification CLI")

@cli.command(help="Merge/clean CSVs and split into train/val")
def prepare(
    raw_dir: str = typer.Option("data/raw", help="Folder with raw CSVs"),
    out_dir: str = typer.Option("data/processed", help="Output folder for processed CSVs"),
    val_ratio: float = typer.Option(0.2, help="Validation ratio")
):
    out = prepare_fn(raw_dir, out_dir, val_ratio=val_ratio)
    rprint(out)

@cli.command(help="Train and evaluate")
def train(
    config: str = typer.Option("configs/gpu.yaml", help="Config YAML"),
    train_csv: str = typer.Option("data/processed/train.csv"),
    eval_csv: Optional[str] = typer.Option(None),
    text_col: Optional[str] = typer.Option(None, help="Text column (auto-detect if omitted)")
):
    metrics = train_fn(config, train_csv, eval_csv=eval_csv, text_col=text_col)
    rprint(metrics)

if __name__ == "__main__":
    cli()
