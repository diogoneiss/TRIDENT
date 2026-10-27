#!/usr/bin/env bash
# E78 (docs/tickets/imputation-interpretability/01-occlusion-map.md), one of four GPU training queues
# (cells from occlusion_map.py --todo <n>), from the checkout root:
#   setsid nohup bash scripts/experiments/run_occlusion_queue.sh <n> > /dev/null 2>&1 &
# Finished cells are skipped, so relaunching resumes. Stop between cells: touch $S/STOP.
# STUDY=e79 runs E79 (imputation-interpretability/02), STUDY=e85 E85 (05), each in its own state directory; DEADLINE (epoch
# seconds) starts no cell after that time.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
STUDY=${STUDY:-e78}
DEADLINE=${DEADLINE:-0}
if [ "$STUDY" = e79 ]; then S=/scratch2/diogoneiss/trident-2026-10-27-occlusion-60; ARGS="--study e79"
elif [ "$STUDY" = e85 ]; then S=/scratch2/diogoneiss/trident-2026-10-27-occlusion-grid; ARGS="--study e85"
else S=/scratch2/diogoneiss/trident-2026-10-26-occlusion; ARGS=""; fi
P=$S/progress.log
mkdir -p $S/logs
if ! ITEMS=$(uv run --python 3.11 python -B scripts/experiments/occlusion_map.py $ARGS --todo $N); then
  echo "$(date '+%F %T %z') queue $N: queue read FAILED" >> $P; exit 1
fi
while read -r v s; do
  [ -z "$v" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') queue $N: STOP file" >> $P; exit 0; }
  [ "$DEADLINE" -gt 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "$(date '+%F %T %z') queue $N: deadline, no new cell" >> $P; exit 0; }
  echo "$(date '+%F %T %z') queue $N: cell start ${v}_s$s" >> $P
  if uv run --python 3.11 python -B scripts/experiments/occlusion_map.py $ARGS train $v $s < /dev/null > $S/logs/${v}_s$s.log 2>&1 && grep -q CELL_DONE $S/logs/${v}_s$s.log; then
    echo "$(date '+%F %T %z') queue $N: cell done ${v}_s$s :: $(grep -o 'CELL_DONE.*' $S/logs/${v}_s$s.log)" >> $P
  else
    echo "$(date '+%F %T %z') queue $N: cell FAILED ${v}_s$s" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') queue $N: queue complete" >> $P
