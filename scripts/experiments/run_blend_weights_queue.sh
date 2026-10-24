#!/usr/bin/env bash
# E69's GPU lane <n> (docs/tickets/imputation-protocol/24-blend-weights.md), from the checkout root.
# Reads "<set> <variant> <seed>" lines from $S/items<n>.txt; a pair whose csv exists is skipped.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TABPFN_MODEL_CACHE_DIR=/scratch2/diogoneiss/tabpfn-weights
S=/scratch2/diogoneiss/trident-2026-10-23-blend-weights; P=$S/progress.log
mkdir -p $S/logs
while read -r set v s; do
  [ -z "$v" ] && continue
  [ -f "$S/csv/${v}_s$s.csv" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') lane $N STOP file" >> $P; exit 0; }
  echo "$(date '+%F %T %z') lane $N start $v s$s" >> $P
  if uv run --python 3.11 --extra tabpfn python -B scripts/experiments/blend_weights.py $set $v $s $S > $S/logs/${v}_s$s.log 2>&1; then
    echo "$(date '+%F %T %z') lane $N done $v s$s :: $(grep -o 'DONE.*' $S/logs/${v}_s$s.log)" >> $P
  else
    echo "$(date '+%F %T %z') lane $N FAILED $v s$s" >> $P
  fi
done < $S/items$N.txt
echo "$(date '+%F %T %z') lane $N complete" >> $P
