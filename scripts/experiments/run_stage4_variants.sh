#!/usr/bin/env bash
# E98 variant runner (amendment 2026-10-30), from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_stage4_variants.sh <list> > /dev/null 2>&1 &
# Walks the variants in <list>, one per line, takes each whose variant lock it wins, and runs its
# study, then its two evaluations, in that order. stage4_trident.py skips a finished study and
# claims each item, so a queue that reaches the same item skips it. Same cap as the queues.
set -u
LIST=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:47731}
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/trident-2026-10-30-step-zero; else S=/scratch/diogoneiss/trident-2026-10-30-step-zero; fi
DEADLINE=$(date -d '2026-10-31 08:00' +%s)
P=$S/progress.log
mkdir -p $S/vlocks $S/logs
PY="uv run --python 3.11 python -B scripts/experiments/stage4_trident.py"
while read -r v; do
  [ -z "$v" ] && continue
  mkdir "$S/vlocks/$v" 2>/dev/null || continue
  for item in "study $v" "eval $v 7" "eval $v 13"; do
    [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') runner: STOP file" >> $P; exit 0; }
    [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "$(date '+%F %T %z') runner: deadline, no new item" >> $P; exit 0; }
    set -- $item; c=${1}_${2}${3:+_s$3}
    echo "$(date '+%F %T %z') runner: cell start $c" >> $P
    $PY $item < /dev/null > $S/logs/runner_$c.log 2>&1
    if grep -q "CELL_DONE.*skipped" $S/logs/runner_$c.log; then echo "$(date '+%F %T %z') runner: cell skipped $c" >> $P
    elif grep -q CELL_DONE $S/logs/runner_$c.log; then echo "$(date '+%F %T %z') runner: cell done $c" >> $P
    else echo "$(date '+%F %T %z') runner: cell FAILED $c" >> $P; fi
  done
done < "$LIST"
echo "$(date '+%F %T %z') runner: list complete" >> $P
