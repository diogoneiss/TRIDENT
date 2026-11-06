#!/usr/bin/env bash
# T50 on gorgona8: E149's two queues (docs/tickets/mimic/04-pattern-transplant.md) start when E148's two have ended,
# so the study never holds more than two trainers on a GPU it shares with another project. From the checkout root:
#   TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data setsid nohup bash scripts/experiments/run_mimic_chain.sh > /dev/null 2>&1 &
# Stop between cells: touch <state>/STOP in the running study's state directory; one in E148's before its queues end
# also keeps E149 from starting. Finished cells are skipped, so relaunching resumes.
set -u
export TZ=America/Sao_Paulo
: "${TRIDENT_DATASETS_ROOT:?set TRIDENT_DATASETS_ROOT}"
A=/scratch2/diogoneiss/trident-2026-11-06-mimic-e148
B=/scratch2/diogoneiss/trident-2026-11-06-mimic-e149
Q=2
mkdir -p $B/logs
say() { echo "$(date '+%F %T %z') chain: $*" >> $B/progress.log; }
say "waiting for E148's $Q queues"
for i in $(seq 1 720); do  # at most 12 hours
  [ "$(grep -c 'queue complete' $A/progress.log 2>/dev/null)" -ge $Q ] && break
  if [ -f $A/STOP ] || [ -f $B/STOP ]; then say "STOP file; E149 not started"; exit 0; fi
  sleep 60
done
if [ "$(grep -c 'queue complete' $A/progress.log 2>/dev/null)" -lt $Q ] || grep -q 'cell FAILED' $A/progress.log; then
  say "E148 did not end clean; E149 not started"; exit 1
fi
say "E148's queues ended; starting E149's"
for q in $(seq 1 $Q); do
  QUEUE_STATE=$B QUEUE_RUNNER=scripts/experiments/mimic_transplant_run.py CUDA_VISIBLE_DEVICES=0 \
    setsid nohup bash scripts/experiments/c15_queue.sh $q $Q > /dev/null 2>&1 < /dev/null &
done
