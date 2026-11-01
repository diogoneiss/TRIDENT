#!/usr/bin/env bash
# E99 (docs/tickets/classification-step-zero/08-deep-baselines.md), or E106/E107 with STUDY=e106|e107, one lane of this machine's units
# (deep_baselines.py --deal <machine>, dearest first), from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_deep_baselines_queue.sh <n> > /dev/null 2>&1 &
# Every lane walks the same list and takes a unit whose lock it wins, so no unit runs twice. A finished
# unit (its csv) is skipped, so relaunching resumes. MLflow is off. Stop between units: touch $S/STOP.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
STUDY=${STUDY:-}
if [ -n "$STUDY" ]; then D=trident-2026-11-01-deep-$STUDY; else D=trident-2026-10-31-deep-baselines; fi
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/$D; M=gorgona8; else S=/scratch/diogoneiss/$D; M=medusa; fi
P=$S/progress.log
mkdir -p $S/logs $S/locks
PY="uv run --python 3.11 python -B scripts/experiments/deep_baselines.py${STUDY:+ --study $STUDY}"
ITEMS=$($PY --deal $M) || { echo "$(date '+%F %T %z') lane $N: deal FAILED" >> $P; exit 1; }
while read -r m v; do
  [ -z "$m" ] && continue
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') lane $N: STOP file" >> $P; exit 0; }
  [ -f "$S/csv/${m}_$v.csv" ] && continue
  mkdir "$S/locks/${m}_$v" 2>/dev/null || continue
  echo "$(date '+%F %T %z') lane $N: unit start ${m}_$v" >> $P
  if $PY run $m $v < /dev/null > $S/logs/${m}_$v.log 2>&1 && [ -f "$S/csv/${m}_$v.csv" ]; then
    echo "$(date '+%F %T %z') lane $N: unit done ${m}_$v" >> $P
  else
    echo "$(date '+%F %T %z') lane $N: unit FAILED ${m}_$v" >> $P
  fi
done <<< "$ITEMS"
echo "$(date '+%F %T %z') lane $N: lane complete" >> $P
