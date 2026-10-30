#!/usr/bin/env bash
# E100 (docs/tickets/imputation-protocol/37-steps-rule.md), one GPU queue of <queues> on this machine
# (cells from steps_rule.py --todo <n> <queues>), from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_steps_rule_queue.sh <n> <queues> > /dev/null 2>&1 &
# MLflow is off; finished cells are skipped, so relaunching resumes. Stop between cells: touch $S/STOP.
set -u
N=$1; Q=$2
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/trident-2026-10-30-steps-rule; else S=/scratch/diogoneiss/trident-2026-10-30-steps-rule; fi
P=$S/progress.log
mkdir -p $S/logs
PY="uv run --python 3.11 python -B scripts/experiments/steps_rule.py"
ITEMS=$($PY --todo $N $Q) || { echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1; }
while read -r a v s; do
  [ -z "$a" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  c=${a}_${v}_s$s
  echo "$(date '+%F %T %z') queue $N: cell start $c" >> $P
  if $PY train $a $v $s < /dev/null > $S/logs/$c.log 2>&1 && grep -q CELL_DONE $S/logs/$c.log; then
    echo "$(date '+%F %T %z') queue $N: cell done $c" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED $c" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
