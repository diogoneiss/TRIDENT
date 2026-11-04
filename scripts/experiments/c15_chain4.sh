#!/usr/bin/env bash
# C15 Stage 4 (ticket 06) on idle GPUs: once this machine's overnight chain (c15_chain.sh) has finished its
# stages and no C15 queue runs, and G1 passed, start Stage 4's queues.
#   setsid nohup bash scripts/experiments/c15_chain4.sh <queues> <gpus> > $S/chain4.log 2>&1 &
set -u
Q=$1; GPUS=($2)
if [ "$(hostname)" = gorgona8 ]; then S=/scratch2/diogoneiss/trident-2026-11-04-c15-stage1; else S=/scratch/diogoneiss/trident-2026-11-04-c15-stage1; fi
while pgrep -f "[c]15_chain.sh" > /dev/null || pgrep -f "[c]15_queue.sh" > /dev/null; do sleep 60; done
[ -f $S/G1_PASSED ] || { echo "$(date '+%F %T %z') no G1_PASSED: stage 4 not started"; exit 0; }
echo "$(date '+%F %T %z') starting stage 4 with $Q queues"
for q in $(seq 1 $Q); do
  g=${GPUS[$(( (q - 1) % ${#GPUS[@]} ))]}
  C15_STAGE=4 CUDA_VISIBLE_DEVICES=$g setsid nohup bash scripts/experiments/c15_queue.sh $q $Q > /dev/null 2>&1 < /dev/null &
done
