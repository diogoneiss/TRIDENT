#!/usr/bin/env bash
# One of the four concurrent queues of the default refresh at 20nan on fresh seeds
# (docs/tickets/imputation-protocol/08-default-refresh.md, E51), one cell at a time. Run from the
# checkout root, detached:
#   setsid nohup bash scripts/experiments/run_default_refresh_queue.sh <1|2|3|4> <state_dir> > /dev/null 2>&1 &
# The queue is read from the store at start (finished cells are skipped), so relaunching resumes.
# Stop between cells: touch <state_dir>/STOP. Every cell must end by the pre-registered deadline,
# 08:00 GMT-3 on 2026-10-20: a cell starts only if now + 1.3 x its estimated minutes is before it.
set -u
export TZ=America/Sao_Paulo
# Two trainers fit the baseline imputers at once; at 32 threads each they oversubscribe the
# 32 cores and stall. Their fills are identical at 1, 4 and 32 threads (ticket 02, note).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
QUEUE="$1"
STATE="$2"
ROOT="$(pwd)"
[ -f "$ROOT/train.py" ] || { echo "run from the checkout root" >&2; exit 2; }
DEADLINE="${DEADLINE:-2026-10-20 08:00:00 -0300}"
DEADLINE_EPOCH="$(date -d "$DEADLINE" +%s)"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$STATE/logs"
P="$STATE/progress.log"
stamp() { date '+%F %T %z'; }
if ! CELLS="$(uv run --python 3.11 python "$HERE/default_refresh_report.py" --todo "$QUEUE")"; then
  echo "$(stamp) queue $QUEUE: queue read FAILED" >> "$P"
  exit 1
fi
echo "$(stamp) launcher up (queue $QUEUE); $(printf '%s\n' "$CELLS" | grep -c .) cells" >> "$P"
while read -r ds seed minutes; do
  [ -z "$ds" ] && continue
  if [ -f "$STATE/STOP" ]; then echo "$(stamp) queue $QUEUE: STOP file found; exiting" >> "$P"; exit 0; fi
  cell="${ds}_s${seed}"
  if [ $(( $(date +%s) + minutes * 78 )) -ge "$DEADLINE_EPOCH" ]; then
    echo "$(stamp) queue $QUEUE: cell skipped $cell (would not end by the deadline)" >> "$P"
    continue
  fi
  echo "$(stamp) queue $QUEUE: cell start $cell" >> "$P"
  uv run --python 3.11 python "$HERE/default_refresh_run.py" "$ds" "$seed" \
    "$STATE/results/$cell" "$STATE/metrics/$cell" < /dev/null > "$STATE/logs/$cell.log" 2>&1
  if grep -q CELL_DONE "$STATE/logs/$cell.log"; then
    echo "$(stamp) queue $QUEUE: cell done $cell :: $(grep -o 'CELL_DONE .*' "$STATE/logs/$cell.log")" >> "$P"
  else
    echo "$(stamp) queue $QUEUE: cell FAILED $cell" >> "$P"
  fi
done <<< "$CELLS"
echo "$(stamp) queue $QUEUE: queue complete" >> "$P"
