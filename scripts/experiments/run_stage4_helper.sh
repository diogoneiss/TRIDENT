#!/usr/bin/env bash
# E98 helper queue <h> (amendment 2026-10-30), on medusa1 GPU0, from the checkout root:
#   CUDA_VISIBLE_DEVICES=0 setsid nohup bash scripts/experiments/run_stage4_helper.sh <h> <helpers> > /dev/null 2>&1 &
# It walks medusa1's items (queues 4-6 of 6) from the end, every <helpers>-th one, and runs each that
# is ready and unclaimed; stage4_trident.py's claims keep any item from running twice. An evaluation
# whose study is not finished is left for its own queue. Same cap as the six queues.
set -u
H=$1; HS=$2
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:47731}
S=/scratch/diogoneiss/trident-2026-10-30-step-zero
DEADLINE=$(date -d '2026-10-31 08:00' +%s)
P=$S/progress.log
PY="uv run --python 3.11 python -B scripts/experiments/stage4_trident.py"
ITEMS=$(for n in 4 5 6; do $PY --todo $n 6; done | tac | awk -v h=$H -v hs=$HS '(NR - 1) % hs == h - 1')
while read -r kind v s; do
  [ -z "$kind" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') helper $H: STOP file" >> $P; exit 0; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "$(date '+%F %T %z') helper $H: deadline, no new item" >> $P; exit 0; }
  c=${kind}_${v}${s:+_s$s}
  [ -e "$S/claims/$c" ] && continue
  $PY $kind $v $s < /dev/null > $S/logs/helper_$c.log 2>&1
  if grep -q "CELL_DONE.*skipped" $S/logs/helper_$c.log; then continue
  elif grep -q CELL_DONE $S/logs/helper_$c.log; then echo "$(date '+%F %T %z') helper $H: cell done $c" >> $P
  elif grep -q "no finished study yet" $S/logs/helper_$c.log; then continue
  else echo "$(date '+%F %T %z') helper $H: cell FAILED $c" >> $P; fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') helper $H: complete" >> $P
