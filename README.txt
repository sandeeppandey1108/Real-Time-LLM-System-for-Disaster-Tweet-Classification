This patch contains:
- scripts/run_all.ps1  (PowerShell-safe; no heredocs; uses temp file for Python stamp step)
- Dockerfile.gpu       (adds pip timeout/retries to fix intermittent ReadTimeout during build)

How to apply:
1) Replace your repo files with these two:
   - copy scripts/run_all.ps1 → <repo_root>\scripts\run_all.ps1
   - copy Dockerfile.gpu      → <repo_root>\Dockerfile.gpu

2) Rebuild:
   powershell
   $Image = "sandeep_pandey/crisis-llm:gpu-latest"
   docker build -f Dockerfile.gpu -t $Image .

3) Run end-to-end:
   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_all.ps1" -Image $Image
