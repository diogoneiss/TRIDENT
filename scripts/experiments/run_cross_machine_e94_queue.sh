#!/usr/bin/env bash
# E94 (docs/tickets/cross-machine/02-equivalence.md) or, with STUDY=e96, E96 (03), one of four GPU
# queues per machine (cells from cross_machine.py --study $STUDY --todo <n>), from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_cross_machine_e94_queue.sh <n> > /dev/null 2>&1 &
# MLflow is off; finished cells are skipped, so relaunching resumes. Stop between cells: touch $S/STOP.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
STUDY=${STUDY:-e94}
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/trident-2026-10-29-cross-machine-$STUDY; else S=/scratch/diogoneiss/trident-2026-10-29-cross-machine-$STUDY; fi
DEADLINE=$(date -d '2026-10-30 02:00' +%s)
P=$S/progress.log
mkdir -p $S/logs
PY="uv run --python 3.11 python -B scripts/experiments/cross_machine.py --study $STUDY"
ITEMS=$($PY --todo $N) || { echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1; }
while read -r t v s d; do
  [ -z "$t" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "$(date '+%F %T %z') queue $N: deadline, no new cell" >> $P; exit 0; }
  c=${t}_${v}_s$s
  echo "$(date '+%F %T %z') queue $N: cell start $c" >> $P
  if $PY train $t $v $s < /dev/null > $S/logs/$c.log 2>&1 && grep -q CELL_DONE $S/logs/$c.log; then
    echo "$(date '+%F %T %z') queue $N: cell done $c" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED $c" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
