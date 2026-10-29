#!/usr/bin/env bash
# E89 (docs/tickets/imputation-protocol/33-hybrid-selection-rules.md), one of two GPU lanes on gorgona8,
# from the checkout root:
#   setsid nohup bash scripts/experiments/run_hybrid_selection_rules_queue.sh <1|2> > /dev/null 2>&1 &
# Cells go to the lanes greedily by table size, the largest first; a finished cell is skipped.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
S=/scratch2/diogoneiss/trident-2026-10-28-hybrid-selection-rules
P=$S/progress.log
mkdir -p $S/logs
ITEMS=$(uv run --python 3.11 python -B -c "
import sys; sys.path.insert(0, '.')
from scripts.experiments import deployable_hybrid as e82
work = sorted(((e82.ROWS[v.split('_')[0]], v, s) for v in e82.VARIANTS for s in e82.SEEDS), key=lambda w: (-w[0], w[1], w[2]))
loads = [0.0, 0.0]
for cost, v, s in work:
    lane = loads.index(min(loads)); loads[lane] += cost
    if lane == $N - 1: print(v, s)
") || { echo "$(date '+%F %T %z') lane $N: queue read FAILED" >> $P; exit 1; }
while read -r v s; do
  [ -z "$v" ] && continue
  [ -f "$S/cells/${v}_s$s/selection_fold_5.csv" ] && [ -f "$S/cells/${v}_s$s/scale_fold_5.json" ] && continue
  echo "$(date '+%F %T %z') lane $N: start ${v}_s$s" >> $P
  if uv run --python 3.11 --extra tabpfn python -B scripts/experiments/hybrid_selection_rules.py cells $v $s < /dev/null > $S/logs/${v}_s$s.log 2>&1; then
    echo "$(date '+%F %T %z') lane $N: done ${v}_s$s :: $(grep -o 'DONE.*' $S/logs/${v}_s$s.log)" >> $P
  else
    echo "$(date '+%F %T %z') lane $N: FAILED ${v}_s$s" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') lane $N: lane complete" >> $P
