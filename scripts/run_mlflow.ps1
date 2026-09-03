# Launch the MLflow UI against the local file store (Windows PowerShell).
#
#   .\scripts\run_mlflow.ps1     ->  http://localhost:5000
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui --backend-store-uri "file:///$((Get-Location).Path -replace '\','/')/mlruns" --host 127.0.0.1 --port 5000
