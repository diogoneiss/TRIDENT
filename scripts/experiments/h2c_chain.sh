#!/usr/bin/env bash
# E142 (c15-mnar-prediction/10): this machine's stage-h2c A1 cells in the foreground (one queue), then the TabPFN
# pairs that now have an A1 cell here.  setsid nohup bash scripts/experiments/h2c_chain.sh <gpu> > $S/h2c_chain.log 2>&1 &
set -u
echo "$(date '+%F %T %z') starting stage h2c"
C15_STAGE=h2c CUDA_VISIBLE_DEVICES=$1 bash scripts/experiments/c15_queue.sh 1 1
echo "$(date '+%F %T %z') stage h2c queue done; TabPFN pairs"
CUDA_VISIBLE_DEVICES=$1 uv run --python 3.11 --extra tabpfn python -B scripts/experiments/tabpfn_mnar.py run 2>&1 | grep -E "DONE|Error|Traceback|not run|not on this machine"
echo "$(date '+%F %T %z') h2c chain done"
