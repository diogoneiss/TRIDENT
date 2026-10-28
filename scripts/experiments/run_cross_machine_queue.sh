#!/usr/bin/env bash
# E86 (docs/tickets/cross-machine/01-gorgona8-medusa1.md), one of two GPU queues per machine
# (cells from cross_machine.py --todo <n>), from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_cross_machine_queue.sh <n> > /dev/null 2>&1 &
# Finished cells are skipped, so relaunching resumes. Stop between cells: touch $S/STOP.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/trident-2026-10-28-cross-machine; else S=/scratch/diogoneiss/trident-2026-10-28-cross-machine; fi
P=$S/progress.log
mkdir -p $S/logs
if ! ITEMS=$(uv run --python 3.11 python -B scripts/experiments/cross_machine.py --todo $N); then
  echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1
fi
while read -r t v s d; do
  [ -z "$t" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  c=${t}_${v}_s$s${d:+_$d}
  echo "$(date '+%F %T %z') queue $N: cell start $c" >> $P
  if uv run --python 3.11 python -B scripts/experiments/cross_machine.py train $t $v $s $d < /dev/null > $S/logs/$c.log 2>&1 && grep -q CELL_DONE $S/logs/$c.log; then
    echo "$(date '+%F %T %z') queue $N: cell done $c" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED $c" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
