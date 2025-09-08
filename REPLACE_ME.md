
# How to replace your current project

1) Unzip this package.
2) Copy all folders into your repo root, overwriting existing files:
   - `src/ai_bin/` (contains the **compatibility-fixed** `train.py`)
   - `configs/bin_es.yaml` (**safe**: lr=3e-5, epochs=1000, patience=10)
   - `configs/bin_es_highlr.yaml` (**aggressive**: lr=1e-3, epochs=1000, patience=10)
   - `scripts/run_all_bin_mounted.ps1` (PowerShell-safe; `${Var}` in volume bindings)
3) Re-run:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all_bin_mounted.ps1 -Full -ProjectRoot .
   ```
   - To use the aggressive LR: add `-Config configs\bin_es_highlr.yaml`

