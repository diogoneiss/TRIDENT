#!/usr/bin/env bash
# E98 sweep (amendment 2026-10-30), once this machine's studies are done, from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_stage4_sweep.sh <k> <lanes> > /dev/null 2>&1 &
# Runs every evaluation still missing for a variant whose study finished on this machine, lane k of
# <lanes>; claims keep any evaluation from running twice.
set -u
K=$1; L=$2
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:47731}
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/trident-2026-10-30-step-zero; else S=/scratch/diogoneiss/trident-2026-10-30-step-zero; fi
P=$S/progress.log
ITEMS=$(uv run --python 3.11 python -B -c "
import sys; sys.path.insert(0, '.')
from scripts.experiments import stage4_trident as t
from scripts.experiments.cls_stage2_run import fold_metrics
work = [(v, s) for v in t.VARIANTS for s in t.EVAL_SEEDS if t.retrained(v) and fold_metrics(t.eval_dir(v, s)) is None]
for k, (v, s) in enumerate(work):
    if k % $L == $K - 1: print(v, s)
")
while read -r v s; do
  [ -z "$v" ] && continue
  c=eval_${v}_s$s
  uv run --python 3.11 python -B scripts/experiments/stage4_trident.py eval $v $s < /dev/null > $S/logs/sweep_$c.log 2>&1
  if grep -q "CELL_DONE.*skipped" $S/logs/sweep_$c.log; then echo "$(date '+%F %T %z') sweep: cell skipped $c" >> $P
  elif grep -q CELL_DONE $S/logs/sweep_$c.log; then echo "$(date '+%F %T %z') sweep: cell done $c" >> $P
  else echo "$(date '+%F %T %z') sweep: cell FAILED $c" >> $P; fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') sweep $K: complete" >> $P
