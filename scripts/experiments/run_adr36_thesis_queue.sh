#!/usr/bin/env bash
# ADR 0036 thesis cells (docs/tickets/imputation-protocol/43-adr36-thesis-cells.md), one GPU queue of <queues> on this machine
# (cells from adr36_thesis_runs.py --todo <n> <queues>), from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_adr36_thesis_queue.sh <n> <queues> > /dev/null 2>&1 &
# MLflow is on (the leader, through MLFLOW_TRACKING_URI); finished cells are skipped, so relaunching resumes. Stop between cells: touch $S/STOP.
set -u
N=$1; Q=$2
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:47731}
# ADR37=1 pre-computes ADR 0037's confirmation cells (adr36_thesis_runs.py --adr37) in their own state.
if [ -n "${ADR37:-}" ]; then D=trident-2026-11-03-adr37-thesis-pre; X=" --adr37"; else D=trident-2026-11-01-adr36-thesis; X=""; fi
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/$D; else S=/scratch/diogoneiss/$D; fi
P=$S/progress.log
mkdir -p $S/logs
PY="uv run --python 3.11 python -B scripts/experiments/adr36_thesis_runs.py$X"
ITEMS=$($PY --todo $N $Q) || { echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1; }
while read -r v s; do
  [ -z "$v" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  c=${v}_s$s
  echo "$(date '+%F %T %z') queue $N: cell start $c" >> $P
  if $PY train $v $s < /dev/null > $S/logs/$c.log 2>&1 && grep -q CELL_DONE $S/logs/$c.log; then
    echo "$(date '+%F %T %z') queue $N: cell done $c" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED $c" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
