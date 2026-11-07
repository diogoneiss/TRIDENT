#!/usr/bin/env bash
# E156 (docs/tickets/c15-mnar-prediction/12-mechanism-split.md): 36 cells on gorgona8, three queues, every arm of a
# (file, seed) pair in one queue (mechanism_split.py).
# A STOP file in the state dir stops the queues between cells.
# Run from the checkout root, from a copy: setsid nohup bash <copy> &
set -u
STATE=/scratch2/diogoneiss/trident-2026-11-06-mechanism-split
LOG=$STATE/progress.log
QUEUES=3
mkdir -p "$STATE/logs"
say() { echo "$(date '+%F %T %z') $*" >> "$LOG"; }

cells() {  # the queue's remaining cells, as "<runner> <arm> <variant> <seed>"
  uv run --python 3.11 python -B scripts/experiments/mechanism_split.py --todo "$1" "$QUEUES" 2>/dev/null | sed 's/^/mechanism_split /'
}

queue() {
  local n=$1 runner arm variant seed
  cells "$n" | while read -r runner arm variant seed; do
    if [ -e "$STATE/STOP" ]; then say "queue $n: STOP seen"; break; fi
    say "queue $n: cell start ${arm}_${variant}_s${seed}"
    if uv run --python 3.11 python -B "scripts/experiments/${runner}.py" train "$arm" "$variant" "$seed" \
         > "$STATE/logs/${arm}_${variant}_s${seed}.log" 2>&1 < /dev/null; then
      say "queue $n: cell done ${arm}_${variant}_s${seed}"
    else
      say "queue $n: cell FAILED ${arm}_${variant}_s${seed}"
    fi
  done
  say "queue $n: ended"
}

say "chain: starting $QUEUES queues"
for n in $(seq 1 $QUEUES); do queue "$n" & done
wait
say "chain: all queues ended"
