#!/usr/bin/env bash
# E98 (docs/tickets/classification-step-zero/07-step-zero.md), TRIDENT queue <n> of <queues>, from the
# checkout root (queues 1-3 on gorgona8, 4-6 on medusa1):
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_stage4_queue.sh <n> <queues> > /dev/null 2>&1 &
# Each queue runs its variants' studies and evaluations; a finished item is skipped, so relaunching
# resumes. No item starts after 2026-10-31 08:00 GMT-3 (the cap). Stop between items: touch $S/STOP.
set -u
N=$1; Q=$2
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:47731}
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/trident-2026-10-30-step-zero; else S=/scratch/diogoneiss/trident-2026-10-30-step-zero; fi
DEADLINE=$(date -d '2026-10-31 08:00' +%s)
P=$S/progress.log
mkdir -p $S/logs
PY="uv run --python 3.11 python -B scripts/experiments/stage4_trident.py"
ITEMS=$($PY --todo $N $Q) || { echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1; }
while read -r kind v s; do
  [ -z "$kind" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "$(date '+%F %T %z') queue $N: deadline, no new item" >> $P; exit 0; }
  c=${kind}_${v}${s:+_s$s}
  echo "$(date '+%F %T %z') queue $N: cell start $c" >> $P
  if $PY $kind $v $s < /dev/null > $S/logs/$c.log 2>&1 && grep -q CELL_DONE $S/logs/$c.log; then
    echo "$(date '+%F %T %z') queue $N: cell done $c" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED $c" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
