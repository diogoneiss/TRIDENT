#!/usr/bin/env bash
# E152 (docs/tickets/mimic/06-sofa-positive-control.md): wait for E149's cells to end on gorgona8, then run the 9 cells
# on three queues, one seed each (every arm of a seed in one queue). A STOP file in the state dir stops the queues
# between cells. Run from the checkout root: setsid nohup bash scripts/experiments/run_sofa_g1_chain.sh &
set -u
STATE=/scratch2/diogoneiss/trident-2026-11-06-sofa-g1
LOG=$STATE/progress.log
QUEUES=3
mkdir -p "$STATE/logs"
say() { echo "$(date '+%F %T %z') $*" >> "$LOG"; }

say "chain: waiting for E149's cells (mimic_transplant_run) to end"
while pgrep -f '[m]imic_transplant_run.py train' > /dev/null; do sleep 60; done
say "chain: no E149 cell running; starting $QUEUES queues"

queue() {
  local n=$1 arm seed
  uv run --python 3.11 python -B scripts/experiments/sofa_g1.py --todo "$n" "$QUEUES" 2>/dev/null | while read -r arm seed; do
    if [ -e "$STATE/STOP" ]; then say "queue $n: STOP seen"; break; fi
    say "queue $n: cell start ${arm}_s${seed}"
    if uv run --python 3.11 python -B scripts/experiments/sofa_g1.py train "$arm" "$seed" > "$STATE/logs/${arm}_s${seed}.log" 2>&1 < /dev/null; then
      say "queue $n: cell done ${arm}_s${seed}"
    else
      say "queue $n: cell FAILED ${arm}_s${seed}"
    fi
  done
  say "queue $n: ended"
}

for n in $(seq 1 $QUEUES); do queue "$n" & done
wait
say "chain: all queues ended"
