#!/usr/bin/env bash
# C1's cuts in sequence on this machine (E134 then E135): each cut's queues through c15_queue.sh, the next cut
# once this machine's queues are done.  setsid nohup bash scripts/experiments/c1_chain.sh "<cuts>" <queues> "<gpus>" > $S/chain.log 2>&1 &
set -u
CUTS=($1); Q=$2; GPUS=($3)
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/trident-2026-11-04-c1; else S=/scratch/diogoneiss/trident-2026-11-04-c1; fi
for cut in "${CUTS[@]}"; do
  echo "$(date '+%F %T %z') starting cut $cut with $Q queues"
  for q in $(seq 1 $Q); do
    g=${GPUS[$(( (q - 1) % ${#GPUS[@]} ))]}
    C1_CUT=$cut QUEUE_STATE=$S QUEUE_RUNNER=scripts/experiments/c1_run.py CUDA_VISIBLE_DEVICES=$g setsid nohup bash scripts/experiments/c15_queue.sh $q $Q > /dev/null 2>&1 < /dev/null &
  done
  sleep 30; while pgrep -f "[c]15_queue.sh" > /dev/null; do sleep 60; done
  echo "$(date '+%F %T %z') cut $cut queues done"
done
