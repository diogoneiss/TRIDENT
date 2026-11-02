#!/usr/bin/env bash
# E111 (docs/tickets/classification-step-zero/11-deep-baselines-fresh-seeds.md): tuned TRIDENT (E98's
# configurations) evaluated at fresh seeds, one lane, from the checkout root:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_e111_trident_queue.sh <n> > /dev/null 2>&1 &
# A variant's evaluations run where its E98 study lives: electricity_40nan on gorgona8, the rest on
# medusa1. Lanes share the list by lock; a finished evaluation is skipped. Stop between items: touch $S/STOP.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:47731}
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/trident-2026-11-02-e111; else S=/scratch/diogoneiss/trident-2026-11-02-e111; fi
P=$S/progress.log
mkdir -p $S/logs $S/locks
if [ "$(hostname)" = gorgona8 ]; then VARIANTS="electricity_40nan"; else
  VARIANTS="electricity_60nan electricity_80nan letter_40nan letter_60nan letter_80nan pendigits_40nan pendigits_60nan pendigits_80nan kr-vs-kp_40nan kr-vs-kp_60nan kr-vs-kp_80nan spambase_40nan spambase_60nan spambase_80nan biodeg_40nan biodeg_60nan biodeg_80nan credit-g_40nan credit-g_60nan credit-g_80nan vehicle_40nan vehicle_60nan vehicle_80nan kc2_40nan kc2_60nan kc2_80nan"; fi
for v in $VARIANTS; do for s in 5151 6262 7373; do
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') lane $N: STOP file" >> $P; exit 0; }
  mkdir "$S/locks/${v}_s$s" 2>/dev/null || continue
  c=eval_${v}_s$s
  echo "$(date '+%F %T %z') lane $N: cell start $c" >> $P
  if uv run --python 3.11 python -B scripts/experiments/stage4_trident.py eval $v $s step-zero-e111-2026-11-02 < /dev/null > $S/logs/$c.log 2>&1 && grep -q "CELL_DONE eval" $S/logs/$c.log; then
    echo "$(date '+%F %T %z') lane $N: cell done $c" >> $P
  else echo "$(date '+%F %T %z') lane $N: cell FAILED $c" >> $P; fi
done; done
echo "$(date '+%F %T %z') lane $N: lane complete" >> $P
