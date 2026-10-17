#!/usr/bin/env bash
# One of the four concurrent queues of the deployable-checkpoint study
# (docs/tickets/imputation-protocol/03-deployable-checkpoint.md, E43), one cell at a
# time. Run from the checkout root, detached:
#   setsid nohup bash scripts/experiments/run_checkpoint_queue.sh <1|2|3|4> <state_dir> > /dev/null 2>&1 &
# The queue is read from the store at start (finished cells are skipped), so relaunching
# resumes. Stop between cells: touch <state_dir>/STOP. No cell starts after the pre-registered
# cutoff, 08:00 GMT-3 on 2026-10-17.
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
CUTOFF="${CUTOFF:-2026-10-17 08:00:00 -0300}"
CUTOFF_EPOCH="$(date -d "$CUTOFF" +%s)"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$STATE/logs"
P="$STATE/progress.log"
stamp() { date '+%F %T %z'; }
if ! CELLS="$(uv run --python 3.11 python "$HERE/checkpoint_report.py" --todo "$QUEUE")"; then
  echo "$(stamp) queue $QUEUE: queue read FAILED" >> "$P"
  exit 1
fi
echo "$(stamp) launcher up (queue $QUEUE); $(printf '%s\n' "$CELLS" | grep -c .) cells" >> "$P"
while read -r ds seed; do
  [ -z "$ds" ] && continue
  if [ -f "$STATE/STOP" ]; then echo "$(stamp) queue $QUEUE: STOP file found; exiting" >> "$P"; exit 0; fi
  if [ "$(date +%s)" -ge "$CUTOFF_EPOCH" ]; then echo "$(stamp) queue $QUEUE: cutoff reached; exiting" >> "$P"; exit 0; fi
  cell="${ds}_s${seed}"
  echo "$(stamp) queue $QUEUE: cell start $cell" >> "$P"
  uv run --python 3.11 python "$HERE/checkpoint_run.py" "$ds" "$seed" \
    "$STATE/results/$cell" "$STATE/metrics/$cell" < /dev/null > "$STATE/logs/$cell.log" 2>&1
  if grep -q CELL_DONE "$STATE/logs/$cell.log"; then
    echo "$(stamp) queue $QUEUE: cell done $cell :: $(grep -o 'CELL_DONE .*' "$STATE/logs/$cell.log")" >> "$P"
  else
    echo "$(stamp) queue $QUEUE: cell FAILED $cell" >> "$P"
  fi
done <<< "$CELLS"
echo "$(stamp) queue $QUEUE: queue complete" >> "$P"
