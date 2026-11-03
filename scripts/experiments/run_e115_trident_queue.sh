#!/usr/bin/env bash
# E115 (docs/tickets/classification-step-zero/14-deep-baselines-mar-mnar.md): tuned TRIDENT on the MAR and
# MNAR variants at 60nan, each with its MCAR sibling's E98 configuration, one lane on medusa1:
#   CUDA_VISIBLE_DEVICES=<gpu> setsid nohup bash scripts/experiments/run_e115_trident_queue.sh <n> > /dev/null 2>&1 &
# E120 (classification-step-zero/16): STUDY=e120 before bash, fresh seeds 5151/6262/7373, own state and tag.
# Lanes share the list by lock; a finished evaluation is skipped. Stop between items: touch $S/STOP.
set -u
N=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:47731}
if [ "${STUDY:-}" = e120 ]; then
  S=/scratch/diogoneiss/trident-2026-11-03-e120; EXP=step-zero-e120-2026-11-03; SEEDS="5151 6262 7373"
else
  S=/scratch/diogoneiss/trident-2026-11-03-e115; EXP=step-zero-e115-2026-11-03; SEEDS="42 7 13"
fi
P=$S/progress.log
mkdir -p $S/logs $S/locks
for v in spambase_probit_60nan spambase_marf_60nan kr-vs-kp_marf_60nan biodeg_probit_60nan biodeg_marf_60nan credit-g_marf_60nan vehicle_probit_60nan vehicle_marf_60nan kc2_probit_60nan kc2_marf_60nan; do
  base=${v%%_*}
  for s in $SEEDS; do
    [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') lane $N: STOP file" >> $P; exit 0; }
    mkdir "$S/locks/${v}_s$s" 2>/dev/null || continue
    c=eval_${v}_s$s
    echo "$(date '+%F %T %z') lane $N: cell start $c" >> $P
    if uv run --python 3.11 python -B scripts/experiments/stage4_trident.py eval $v $s $EXP ${base}_60nan < /dev/null > $S/logs/$c.log 2>&1 && grep -q "CELL_DONE eval" $S/logs/$c.log; then
      echo "$(date '+%F %T %z') lane $N: cell done $c" >> $P
    else echo "$(date '+%F %T %z') lane $N: cell FAILED $c" >> $P; fi
  done
done
echo "$(date '+%F %T %z') lane $N: lane complete" >> $P
