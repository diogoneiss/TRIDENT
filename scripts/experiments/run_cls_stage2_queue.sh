#!/usr/bin/env bash
# E71's GPU queue <n> (docs/tickets/classification-step-zero/02-no-pretraining.md), from the checkout root.
# Reads its cells from cls_stage2_run.py --todo <n>, so a relaunch resumes; a STOP file ends it.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
S=/scratch2/diogoneiss/trident-2026-10-24-cls-stage2; P=$S/progress.log
mkdir -p $S/logs
echo "$(date '+%F %T %z') queue $N up" >> $P
if ! ITEMS=$(uv run --python 3.11 python -B scripts/experiments/cls_stage2_run.py --todo $N $S); then
  echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1
fi
while read -r v s a; do
  [ -z "$v" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  echo "$(date '+%F %T %z') queue $N: cell start ${v}_s${s}_$a" >> $P
  if uv run --python 3.11 python -B scripts/experiments/cls_stage2_run.py $v $s $a $S > $S/logs/${v}_s${s}_$a.log 2>&1; then
    echo "$(date '+%F %T %z') queue $N: cell done ${v}_s${s}_$a :: $(grep -o 'CELL_DONE.*' $S/logs/${v}_s${s}_$a.log)" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED ${v}_s${s}_$a" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
