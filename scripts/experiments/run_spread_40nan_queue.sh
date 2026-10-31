#!/usr/bin/env bash
# E105 (docs/tickets/imputation-protocol/42-spread-40nan.md), one GPU queue of <queues>
# (cells from spread_40nan.py --todo <n> <queues>), from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_spread_40nan_queue.sh <n> <queues> > /dev/null 2>&1 &
# MLflow is off; finished cells are skipped, so relaunching resumes. Stop between cells: touch $S/STOP.
set -u
N=$1; Q=$2
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/trident-2026-10-31-spread-40nan; else S=/scratch/diogoneiss/trident-2026-10-31-spread-40nan; fi
P=$S/progress.log
mkdir -p $S/logs
PY="uv run --python 3.11 python -B scripts/experiments/spread_40nan.py"
ITEMS=$($PY --todo $N $Q) || { echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1; }
while read -r v s; do
  [ -z "$v" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  echo "$(date '+%F %T %z') queue $N: cell start ${v}_s$s" >> $P
  if $PY train $v $s < /dev/null > $S/logs/${v}_s$s.log 2>&1 && grep -q CELL_DONE $S/logs/${v}_s$s.log; then
    echo "$(date '+%F %T %z') queue $N: cell done ${v}_s$s" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED ${v}_s$s" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
