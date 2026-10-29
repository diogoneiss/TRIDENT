#!/usr/bin/env bash
# E92 (docs/tickets/classification-step-zero/05-tabpfn-tenfold.md), one of two GPU queues on medusa1
# (pairs from tabpfn_classification.py --study e92 --todo <n>), from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_tabpfn_tenfold_queue.sh <n> > /dev/null 2>&1 &
# It waits for E87's queues to end, so the GPUs are E87's until then. Finished pairs are skipped, so
# relaunching resumes. Stop between pairs: touch $S/STOP.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
S=/scratch/diogoneiss/trident-2026-10-28-tabpfn-tenfold
P=$S/progress.log
mkdir -p $S/logs
echo "$(date '+%F %T %z') queue $N: waiting for E87's queues" >> $P
while pgrep -f run_cls_tenfold_queue.sh > /dev/null; do sleep 120; done
echo "$(date '+%F %T %z') queue $N: E87's queues ended, starting" >> $P
if ! ITEMS=$(uv run --python 3.11 --extra tabpfn python -B scripts/experiments/tabpfn_classification.py --study e92 --todo $N); then
  echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1
fi
while read -r v s; do
  [ -z "$v" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  c=${v}_s${s}
  echo "$(date '+%F %T %z') queue $N: pair start $c" >> $P
  if uv run --python 3.11 --extra tabpfn python -B scripts/experiments/tabpfn_classification.py --study e92 run $v $s < /dev/null > $S/logs/$c.log 2>&1 && [ -f $S/csv/$c.csv ]; then
    echo "$(date '+%F %T %z') queue $N: pair done $c" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: pair FAILED $c" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
