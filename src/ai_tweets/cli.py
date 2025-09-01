from __future__ import annotations
import typer, uvicorn
from typing import Optional
from .train import train as train_fn
from .serve import create_app

app = typer.Typer(add_completion=False, help="Disaster Tweets LLM CLI")

@app.command(help="Train a sequence classifier")
def train(config: Optional[str] = typer.Option(None, "--config", "-c"), train_csv: str = typer.Option("data/train.csv"), text_col: Optional[str] = None):
    metrics = train_fn(config, train_csv, text_col=text_col)
    typer.echo(metrics)

@app.command(help="Serve the model (HTTP + WebSocket + /metrics)")
def serve(
    host: str = typer.Option("0.0.0.0", help="Host"),
    port: int = typer.Option(8000, help="Port"),
    model_dir: str = typer.Option("artifacts/model", help="Saved model dir"),
    model_name: Optional[str] = typer.Option(None, help="HF model if model_dir missing"),
    reload: bool = typer.Option(False, help="Auto-reload (dev)"),
):
    uvicorn.run(create_app(model_dir, model_name), host=host, port=port, reload=reload)

@app.command(help="Quick one-off prediction")
def predict(text: str):
    from transformers import pipeline
    clf = pipeline("text-classification", model="artifacts/model", tokenizer="artifacts/model")
    print(clf([text])[0])

def main():
    app()

if __name__ == "__main__":
    main()
