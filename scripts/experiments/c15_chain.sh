#!/usr/bin/env bash
# C15's overnight chain on one machine: when this machine's Stage 1 queues are done, and once G1's reading has
# left $S/G1_PASSED (or $S/G1_FAILED, which ends the chain), start Stage 2's queues; on medusa1, then Stage 3's.
#   setsid nohup bash scripts/experiments/c15_chain.sh <queues> <gpus: "0 0 0 0" or "0 1"> > $S/chain.log 2>&1 &
set -u
Q=$1; GPUS=($2)
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/trident-2026-11-04-c15-stage1; STAGES="2"; else S=/scratch/diogoneiss/trident-2026-11-04-c15-stage1; STAGES="2 3"; fi
wait_queues() { while pgrep -f "[c]15_queue.sh" > /dev/null; do sleep 60; done; }
wait_queues
until [ -f $S/G1_PASSED ] || [ -f $S/G1_FAILED ]; do sleep 60; done
[ -f $S/G1_FAILED ] && { echo "$(date '+%F %T %z') G1 failed: chain ends"; exit 0; }
for stage in $STAGES; do
  echo "$(date '+%F %T %z') starting stage $stage with $Q queues"
  for q in $(seq 1 $Q); do
    g=${GPUS[$(( (q - 1) % ${#GPUS[@]} ))]}
    C15_STAGE=$stage CUDA_VISIBLE_DEVICES=$g setsid nohup bash scripts/experiments/c15_queue.sh $q $Q > /dev/null 2>&1 < /dev/null &
  done
  sleep 30; wait_queues
  echo "$(date '+%F %T %z') stage $stage queues done"
done
