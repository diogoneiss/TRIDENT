#!/usr/bin/env bash
# E74 (docs/tickets/imputation-protocol/27-hybrid-output.md), from the checkout root.
#   run_hybrid_queue.sh train <n>   one of four GPU training queues (cells from hybrid_blend.py --todo <n>)
#   run_hybrid_queue.sh blend <n>   one of two GPU blend lanes, over $S/blend_items<n>.txt
#   run_hybrid_queue.sh chain       waits for the four training queues, then starts both blend lanes
set -u
MODE=$1
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TABPFN_MODEL_CACHE_DIR=/scratch2/diogoneiss/tabpfn-weights
S=/scratch2/diogoneiss/trident-2026-10-24-hybrid-blend; P=$S/progress.log
mkdir -p $S/logs
if [ "$MODE" = train ]; then
  N=$2
  if ! ITEMS=$(uv run --python 3.11 python -B scripts/experiments/hybrid_blend.py --todo $N); then
    echo "$(date '+%F %T %z') train $N: queue read FAILED" >> $P; exit 1
  fi
  while read -r v s; do
    [ -z "$v" ] && continue
    [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') train $N: STOP file" >> $P; exit 0; }
    echo "$(date '+%F %T %z') train $N: cell start ${v}_s$s" >> $P
    if uv run --python 3.11 python -B scripts/experiments/hybrid_blend.py train $v $s < /dev/null > $S/logs/${v}_s$s.log 2>&1 && grep -q CELL_DONE $S/logs/${v}_s$s.log; then
      echo "$(date '+%F %T %z') train $N: cell done ${v}_s$s :: $(grep -o 'CELL_DONE.*' $S/logs/${v}_s$s.log)" >> $P
    else
      echo "$(date '+%F %T %z') train $N: cell FAILED ${v}_s$s" >> $P
    fi
  done <<< "$ITEMS"
  echo "$(date '+%F %T %z') train $N: queue complete" >> $P
elif [ "$MODE" = blend ]; then
  N=$2
  while read -r v s; do
    [ -z "$v" ] && continue
    [ -f "$S/blend/csv/${v}_s$s.csv" ] && continue
    echo "$(date '+%F %T %z') blend $N: start ${v}_s$s" >> $P
    if uv run --python 3.11 --extra tabpfn python -B scripts/experiments/hybrid_blend.py blend $v $s < /dev/null > $S/logs/blend_${v}_s$s.log 2>&1; then
      echo "$(date '+%F %T %z') blend $N: done ${v}_s$s :: $(grep -o 'DONE.*' $S/logs/blend_${v}_s$s.log)" >> $P
    else
      echo "$(date '+%F %T %z') blend $N: FAILED ${v}_s$s" >> $P
    fi
  done < $S/blend_items$N.txt
  echo "$(date '+%F %T %z') blend $N: lane complete" >> $P
elif [ "$MODE" = chain ]; then
  until [ "$(grep -cE 'train [0-9]: (queue complete|STOP file|queue read FAILED)' $P 2>/dev/null)" -ge 4 ]; do sleep 60; done
  [ -f "$S/STOP" ] && exit 0
  echo "$(date '+%F %T %z') chain: training ended, starting the blend lanes" >> $P
  for n in 1 2; do setsid bash scripts/experiments/run_hybrid_queue.sh blend $n < /dev/null > $S/blend$n.out 2>&1 & done
  wait
fi
