
import typer, json, os, subprocess, sys
from ai_tweets.data_prep import prepare as prepare_fn
from ai_tweets.train import train as train_fn

cli = typer.Typer(help="Crisis LLM Pro CLI")

@cli.command(help="Merge/clean CSVs in data/raw and write stratified train/val to data/processed")
def prepare(raw_dir: str = typer.Option("data/raw"), out_dir: str = typer.Option("data/processed"),
            val_size: float = typer.Option(0.2), seed: int = typer.Option(42)):
    train_p, val_p = prepare_fn(raw_dir, out_dir, val_size, seed)
    typer.echo(json.dumps({"train": train_p, "val": val_p}, indent=2))

@cli.command(help="Train + evaluate")
def train(config: str = typer.Option(..., "--config"), train_csv: str = typer.Option(..., "--train-csv"),
          eval_csv: str = typer.Option(..., "--eval-csv"), text_col: str = typer.Option(None, "--text-col")):
    metrics = train_fn(config, train_csv, eval_csv, text_col)
    typer.echo(json.dumps(metrics, indent=2))

@cli.command(help="Serve FastAPI (GPU if available)")
def serve(model_dir: str = typer.Option("/app/artifacts/checkpoints/final", "--model-dir"),
          host: str = typer.Option("0.0.0.0", "--host"), port: int = typer.Option(8000, "--port")):
    env = os.environ.copy(); env["MODEL_DIR"] = model_dir
    cmd = [sys.executable, "-m", "uvicorn", "ai_tweets.serve_app:app", "--host", host, "--port", str(port)]
    subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__": cli()
