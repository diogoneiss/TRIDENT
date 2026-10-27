#!/usr/bin/env bash
# E84 (docs/tickets/imputation-protocol/32-spread-mechanisms.md), one of four GPU training queues
# (cells from spread_mechanisms.py --todo <n>), from the checkout root:
#   setsid nohup bash scripts/experiments/run_spread_mechanisms_queue.sh <n> > /dev/null 2>&1 &
# Finished cells are skipped, so relaunching resumes. Stop between cells: touch $S/STOP.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
S=/scratch2/diogoneiss/trident-2026-10-27-spread-mechanisms
P=$S/progress.log
mkdir -p $S/logs
if ! ITEMS=$(uv run --python 3.11 python -B scripts/experiments/spread_mechanisms.py --todo $N); then
  echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1
fi
while read -r v s; do
  [ -z "$v" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  echo "$(date '+%F %T %z') queue $N: cell start ${v}_s$s" >> $P
  if uv run --python 3.11 python -B scripts/experiments/spread_mechanisms.py train $v $s < /dev/null > $S/logs/${v}_s$s.log 2>&1 && grep -q CELL_DONE $S/logs/${v}_s$s.log; then
    echo "$(date '+%F %T %z') queue $N: cell done ${v}_s$s :: $(grep -o 'CELL_DONE.*' $S/logs/${v}_s$s.log)" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED ${v}_s$s" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
