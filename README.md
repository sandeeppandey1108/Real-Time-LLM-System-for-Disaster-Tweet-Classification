# Crisis LLM Pro (GPU)

End-to-end: prepare data → train → stamp labels → serve API + Streamlit UI

## Quickstart (Windows PowerShell)

```powershell
cd .\crisis-llm-pro

# Build
$Image = "sandeep_pandey/crisis-llm:gpu-latest"
docker build -f Dockerfile.gpu -t $Image .

# Run all
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_all.ps1" -Image $Image
```

API: http://localhost:8000/docs  
UI:  http://localhost:8501
