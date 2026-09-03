#!/usr/bin/env bash
# Launch the MLflow UI against the local file store.
#
#   ./scripts/run_mlflow.sh      ->  http://localhost:5000
#
# MLFLOW_ALLOW_FILE_STORE keeps tracking database-free, which is a deliberate
# project constraint: MLflow 3.x otherwise insists on a SQL backend.
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_ALLOW_FILE_STORE=true
exec mlflow ui --backend-store-uri "file://$(pwd)/mlruns" --host 127.0.0.1 --port 5000
