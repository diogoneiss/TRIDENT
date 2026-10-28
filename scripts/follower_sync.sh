#!/usr/bin/env bash
# ADR 0033, interim: on the follower (medusa1), cron keeps the tunnel to the leader's MLflow
# server (runs log through MLFLOW_TRACKING_URI=http://127.0.0.1:47731) and copies this checkout's
# outputs to the leader, additively; files it replaces there go to sync/backups/outputs-<day>/.
# shortcut: no store replica, roles or fork yet; leadership.py of ADR 0033 replaces this script.
. ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
LEADER=gorgona8
LEADER_ROOT=/scratch2/diogoneiss/TRIDENT
PORT=47731  # the same number as the leader's server
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
mkdir -p sync mlruns results metrics
exec 9> sync/follower.lock
flock -n 9 || exit 0
if ! ss -ltn | grep -q "127.0.0.1:$PORT "; then
    $SSH -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -L "127.0.0.1:$PORT:127.0.0.1:$PORT" "$LEADER" && echo "$(date -Is) tunnel up"
fi
rsync -a -e "$SSH" --backup --backup-dir="sync/backups/outputs-$(date +%F)" \
    mlruns results metrics "$LEADER:$LEADER_ROOT/"
echo "$(date -Is) outputs sent, rsync exit $?"
