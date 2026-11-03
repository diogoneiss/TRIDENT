#!/usr/bin/env bash
# E112 (docs/tickets/classification-step-zero/12-ema-target.md), one GPU queue of <queues> on this machine
# (cells from cls_ema.py --todo <n> <queues>), from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_cls_ema_queue.sh <n> <queues> > /dev/null 2>&1 &
# MLflow is on (the leader, through MLFLOW_TRACKING_URI); finished cells are skipped, so relaunching resumes. Stop between cells: touch $S/STOP.
set -u
N=$1; Q=$2
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:47731}
STUDY=${STUDY:-}
if [ "$STUDY" = e114 ]; then D=trident-2026-11-03-cls-ema-e114; else D=trident-2026-11-02-cls-ema; fi
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/$D; else S=/scratch/diogoneiss/$D; fi
P=$S/progress.log
mkdir -p $S/logs
PY="uv run --python 3.11 python -B scripts/experiments/cls_ema.py${STUDY:+ --study $STUDY}"
ITEMS=$($PY --todo $N $Q) || { echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1; }
while read -r v s a; do
  [ -z "$a" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  c=${a}_${v}_s$s
  echo "$(date '+%F %T %z') queue $N: cell start $c" >> $P
  if $PY train $v $s $a < /dev/null > $S/logs/$c.log 2>&1 && grep -q CELL_DONE $S/logs/$c.log; then
    echo "$(date '+%F %T %z') queue $N: cell done $c" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED $c" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
