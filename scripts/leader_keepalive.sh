#!/usr/bin/env bash
# ADR 0033, interim: on the leader (gorgona8), cron starts the MLflow server when it is not
# running, so medusa1's runs can log to this store through their tunnel. Log: sync/mlflow_server.log.
. ~/.bashrc  # cron's environment is bare; the exports sit above the bashrc guard
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
PORT=47731  # unusual on purpose: 5000 is the default of many tools
pgrep -u "$USER" -f "[m]lflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port $PORT" > /dev/null && exit 0
echo "$(date -Is) starting the MLflow server" >> sync/mlflow_server.log
setsid nohup ./start_mlflow.sh "$PORT" >> sync/mlflow_server.log 2>&1 < /dev/null &
