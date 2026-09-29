#!/usr/bin/env bash
# Linux counterpart of start_mlflow.ps1. The UI listens on localhost only, since the server
# is shared; from Windows, reach it through a tunnel and open http://localhost:5000:
#   ssh -L 5000:localhost:5000 <host>
# A different port is the first argument. Stop it with Ctrl+C, which also stops the
# uvicorn workers; a worker left behind keeps mlflow.db open and the sync refuses (ADR 0009).
set -euo pipefail
cd "$(dirname "$0")"
exec uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port "${1:-5000}"
