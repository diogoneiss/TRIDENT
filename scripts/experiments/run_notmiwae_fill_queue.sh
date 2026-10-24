#!/usr/bin/env bash
# E73's CPU lane <n> (docs/tickets/imputation-protocol/26-notmiwae-fill.md), from the checkout root.
# Reads "<e45|e46> <variant> <seed>" lines from $S/items<n>.txt; metric_panel.py skips a finished CSV.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
S=/scratch2/diogoneiss/trident-2026-10-24-notmiwae-fill; P=$S/progress.log
declare -A STATE=([e45]=/scratch2/diogoneiss/trident-2026-10-17-mnar-smoke [e46]=/scratch2/diogoneiss/trident-2026-10-17-mnar-family)
mkdir -p $S/logs
while read -r tag v s; do
  [ -z "$v" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') lane $N STOP file" >> $P; exit 0; }
  [ -f "$S/$tag/${v}_s$s.csv" ] && continue
  echo "$(date '+%F %T %z') lane $N start $tag $v s$s" >> $P
  if uv run --python 3.11 python -B scripts/experiments/metric_panel.py ${STATE[$tag]} $S/$tag $v $s --baselines --fitters notmiwae_fills --cell_suffix _G > $S/logs/${v}_s$s.log 2>&1; then
    echo "$(date '+%F %T %z') lane $N done $tag $v s$s :: $(grep -o 'DONE.*' $S/logs/${v}_s$s.log)" >> $P
  else
    echo "$(date '+%F %T %z') lane $N FAILED $tag $v s$s" >> $P
  fi
done < $S/items$N.txt
echo "$(date '+%F %T %z') lane $N complete" >> $P
