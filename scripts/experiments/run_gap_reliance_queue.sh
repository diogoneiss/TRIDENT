#!/usr/bin/env bash
# E81 (docs/tickets/imputation-interpretability/04-gap-reliance.md), one of four GPU training queues
# (cells from gap_reliance.py --todo <n>), from the checkout root:
#   setsid nohup bash scripts/experiments/run_gap_reliance_queue.sh <n> > /dev/null 2>&1 &
# Finished cells are skipped, so relaunching resumes. Stop between cells: touch $S/STOP.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
S=/scratch2/diogoneiss/trident-2026-10-27-gap-reliance
P=$S/progress.log
mkdir -p $S/logs
if ! ITEMS=$(uv run --python 3.11 python -B scripts/experiments/gap_reliance.py --todo $N); then
  echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1
fi
while read -r v s a; do
  [ -z "$v" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  echo "$(date '+%F %T %z') queue $N: cell start ${v}_s${s}_$a" >> $P
  if uv run --python 3.11 python -B scripts/experiments/gap_reliance.py train $v $s $a < /dev/null > $S/logs/${v}_s${s}_$a.log 2>&1 && grep -q CELL_DONE $S/logs/${v}_s${s}_$a.log; then
    echo "$(date '+%F %T %z') queue $N: cell done ${v}_s${s}_$a :: $(grep -o 'CELL_DONE.*' $S/logs/${v}_s${s}_$a.log)" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED ${v}_s${s}_$a" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
