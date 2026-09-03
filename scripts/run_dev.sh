#!/usr/bin/env bash
# Start the PhishGuard AI API and frontend for local development.
#
#   ./scripts/run_dev.sh
#
# The API runs on :8000 and the Vite dev server on :5173. Both are stopped
# together on Ctrl-C.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f models/phishing_model.joblib ]; then
  echo "WARNING: no model artifact found. The API will start in a degraded"
  echo "         state until you run:  python -m ml.train"
  echo
fi

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "Starting API on http://localhost:8000 (docs at /docs)"
uvicorn backend.app.main:app --reload --port 8000 &

echo "Starting frontend on http://localhost:5173"
(cd frontend && npm run dev) &

wait
