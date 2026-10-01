#!/usr/bin/env bash
# The queue of the batched-heads check
# (docs/tickets/imputation-pretraining/06-batched-decoder-heads.md), one cell at a time.
# Run from the checkout root, detached:
#   setsid nohup bash scripts/experiments/run_pretrain_objective_queue.sh h <state_dir> > /dev/null 2>&1 &
# The queue is read from the store at start (finished cells are skipped), so relaunching
# resumes. Stop between cells: touch <state_dir>/STOP. No cell starts after the
# pre-registered cutoff, 06:30 GMT-3 on 2026-10-02 (09:30 UTC).
set -u
# Two trainers fit the baseline imputers at once; at 32 threads each they oversubscribe the
# 32 cores and stall. Their fills are identical at 1, 4 and 32 threads (ticket 02, note).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
QUEUE="$1"
STATE="$2"
ROOT="$(pwd)"
[ -f "$ROOT/train.py" ] || { echo "run from the checkout root" >&2; exit 2; }
# Override for a later run, e.g. CUTOFF_UTC="2026-10-02 09:30:00"; past it no cell starts.
CUTOFF_UTC="${CUTOFF_UTC:-2026-10-02 09:30:00}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$STATE/logs"
P="$STATE/progress.log"
if ! CELLS="$(uv run --python 3.11 python "$HERE/batched_heads_report.py" --todo "$QUEUE")"; then
  echo "$(date -u '+%F %T') queue read FAILED" >> "$P"
  exit 1
fi
echo "$(date -u '+%F %T') launcher up (queue $QUEUE); $(printf '%s\n' "$CELLS" | grep -c .) cells" >> "$P"
while read -r ds seed arm; do
  [ -z "$ds" ] && continue
  if [ -f "$STATE/STOP" ]; then echo "$(date -u '+%F %T') STOP file found; exiting" >> "$P"; exit 0; fi
  if [ "$(date -u '+%F %T')" \> "$CUTOFF_UTC" ]; then echo "$(date -u '+%F %T') cutoff reached; exiting" >> "$P"; exit 0; fi
  cell="${ds}_s${seed}_${arm}"
  echo "$(date -u '+%F %T') cell start $cell" >> "$P"
  uv run --python 3.11 python "$HERE/batched_heads_run.py" "$ds" "$seed" "$arm" \
    "$STATE/results/$cell" "$STATE/metrics/$cell" < /dev/null > "$STATE/logs/$cell.log" 2>&1
  if grep -q CELL_DONE "$STATE/logs/$cell.log"; then
    echo "$(date -u '+%F %T') cell done $cell :: $(grep -o 'CELL_DONE .*' "$STATE/logs/$cell.log")" >> "$P"
  else
    echo "$(date -u '+%F %T') cell FAILED $cell" >> "$P"
  fi
done <<< "$CELLS"
echo "$(date -u '+%F %T') queue complete" >> "$P"
