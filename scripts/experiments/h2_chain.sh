#!/usr/bin/env bash
# E139 (c15-mnar-prediction/09): this machine's stage-h2 A1 cells through c15_queue.sh, then the TabPFN pairs that
# now have an A1 cell here.  setsid nohup bash scripts/experiments/h2_chain.sh <queues> "<gpus>" > $S/h2_chain.log 2>&1 &
set -u
Q=$1; GPUS=($2)
echo "$(date '+%F %T %z') starting stage h2 with $Q queues"
for q in $(seq 1 $Q); do
  g=${GPUS[$(( (q - 1) % ${#GPUS[@]} ))]}
  C15_STAGE=h2 CUDA_VISIBLE_DEVICES=$g setsid nohup bash scripts/experiments/c15_queue.sh $q $Q > /dev/null 2>&1 < /dev/null &
done
sleep 30; while pgrep -f "[c]15_queue.sh" > /dev/null; do sleep 60; done
echo "$(date '+%F %T %z') stage h2 queues done; TabPFN pairs"
CUDA_VISIBLE_DEVICES=${GPUS[0]} uv run --python 3.11 --extra tabpfn python -B scripts/experiments/tabpfn_mnar.py run 2>&1 | grep -E "DONE|Error|Traceback|not run|not on this machine"
echo "$(date '+%F %T %z') h2 chain done"
