#!/usr/bin/env bash
# E87 (docs/tickets/classification-step-zero/04-tenfold-pretraining.md), one of eight GPU queues on
# medusa1 (cells from cls_tenfold.py --todo <n>), from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_cls_tenfold_queue.sh <n> > /dev/null 2>&1 &
# Finished cells are skipped, so relaunching resumes. Stop between cells: touch $S/STOP.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# A shell started by ssh or setsid has not read ~/.bashrc: log through the leader's server (ADR 0033).
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:47731}
S=/scratch/diogoneiss/trident-2026-10-28-cls-tenfold
P=$S/progress.log
mkdir -p $S/logs
if ! ITEMS=$(uv run --python 3.11 python -B scripts/experiments/cls_tenfold.py --todo $N); then
  echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1
fi
while read -r v s a; do
  [ -z "$v" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  c=${v}_s${s}_$a
  echo "$(date '+%F %T %z') queue $N: cell start $c" >> $P
  if uv run --python 3.11 python -B scripts/experiments/cls_tenfold.py train $v $s $a < /dev/null > $S/logs/$c.log 2>&1 && grep -q CELL_DONE $S/logs/$c.log; then
    echo "$(date '+%F %T %z') queue $N: cell done $c" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED $c" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
