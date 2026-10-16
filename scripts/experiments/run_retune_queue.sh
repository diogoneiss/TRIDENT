#!/usr/bin/env bash
# One of the four concurrent workers of the re-tuning of the promoted imputation files
# (docs/tickets/imputation-retune/01-retune-promoted-files.md, E39). The workers share one job
# list, <state_dir>/jobs.txt (written once by retune_report.py --jobs), and each takes the first
# job nobody has claimed, under a lock. A T cell waits until its variant's study has written
# best_hyperparameters.json. Run from the checkout root, detached:
#   setsid nohup bash scripts/experiments/run_retune_queue.sh <1|2|3|4> <state_dir> > /dev/null 2>&1 &
# Stop between jobs: touch <state_dir>/STOP. No job starts after the pre-registered cutoff,
# 10:00 GMT-3 on 2026-10-16. Relaunching after the workers died: delete <state_dir>/claims/<job>
# for each job that has no done/ or failed/ marker; a killed study restarts from its first trial,
# because opt.py opens a new study directory on every launch.
set -u
export TZ=America/Sao_Paulo
# Two trainers fit the baseline imputers at once; at 32 threads each they oversubscribe the
# 32 cores and stall. Their fills are identical at 1, 4 and 32 threads (ticket 02, note).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
WORKER="$1"
STATE="$2"
ROOT="$(pwd)"
[ -f "$ROOT/train.py" ] || { echo "run from the checkout root" >&2; exit 2; }
CUTOFF="${CUTOFF:-2026-10-16 10:00:00 -0300}"
CUTOFF_EPOCH="$(date -d "$CUTOFF" +%s)"
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS="$STATE/jobs.txt"
mkdir -p "$STATE/logs" "$STATE/claims" "$STATE/done" "$STATE/failed"
P="$STATE/progress.log"
stamp() { date '+%F %T %z'; }
[ -f "$JOBS" ] || { echo "$(stamp) worker $WORKER: no $JOBS" >> "$P"; exit 1; }

best_of() { ls "$STATE/studies/$1"/optuna_*/best_hyperparameters.json 2>/dev/null | head -1; }

# Prints the job it claimed, or WAIT (only T cells left unclaimed, their studies still running),
# or NONE. Runs in the command substitution's subshell, so the lock goes with it.
pick() {
  exec 9>"$STATE/pool.lock"
  flock 9
  local waiting=0 kind ds seed arm id
  while read -r kind ds seed arm; do
    [ -z "$kind" ] && continue
    id="${kind}_${ds}_${seed}_${arm}"
    [ -e "$STATE/claims/$id" ] && continue
    if [ "$kind" = cell ] && [ "$arm" = T ] && [ -z "$(best_of "$ds")" ]; then
      if [ -e "$STATE/failed/study_${ds}_-_-" ]; then
        : > "$STATE/claims/$id"
        echo "$(stamp) worker $WORKER: skip $id, its study failed" >> "$P"
        continue
      fi
      waiting=1
      continue
    fi
    : > "$STATE/claims/$id"
    echo "$kind $ds $seed $arm"
    return
  done < "$JOBS"
  if [ "$waiting" = 1 ]; then echo WAIT; else echo NONE; fi
}

echo "$(stamp) worker $WORKER up" >> "$P"
while true; do
  if [ -f "$STATE/STOP" ]; then echo "$(stamp) worker $WORKER: STOP file found; exiting" >> "$P"; exit 0; fi
  if [ "$(date +%s)" -ge "$CUTOFF_EPOCH" ]; then echo "$(stamp) worker $WORKER: cutoff reached; exiting" >> "$P"; exit 0; fi
  job="$(pick)"
  case "$job" in
    NONE) echo "$(stamp) worker $WORKER: no job left; exiting" >> "$P"; exit 0 ;;
    WAIT) sleep 60; continue ;;
  esac
  read -r kind ds seed arm <<< "$job"
  id="${kind}_${ds}_${seed}_${arm}"
  echo "$(stamp) worker $WORKER: start $id" >> "$P"
  if [ "$kind" = study ]; then
    uv run --python 3.11 python opt.py --dataset_name "$ds" --task imputation --n_trials 40 --seed 42 \
      --search_space reduced --search_objective induced \
      --output_dir "$STATE/studies" --metrics_dir "$STATE/metrics/$id" < /dev/null > "$STATE/logs/$id.log" 2>&1
    best="$(best_of "$ds")"
    if [ -n "$best" ]; then
      : > "$STATE/done/$id"
      echo "$(stamp) worker $WORKER: done $id :: $(tr -d '\n ' < "$best")" >> "$P"
    else
      : > "$STATE/failed/$id"
      echo "$(stamp) worker $WORKER: FAILED $id" >> "$P"
    fi
  else
    extra=()
    [ "$arm" = T ] && extra=(--config "$(best_of "$ds")")
    uv run --python 3.11 python "$HERE/retune_run.py" "$ds" "$seed" "$arm" \
      "$STATE/results/$id" "$STATE/metrics/$id" "${extra[@]}" < /dev/null > "$STATE/logs/$id.log" 2>&1
    if grep -q CELL_DONE "$STATE/logs/$id.log"; then
      : > "$STATE/done/$id"
      echo "$(stamp) worker $WORKER: done $id :: $(grep -o 'CELL_DONE .*' "$STATE/logs/$id.log")" >> "$P"
    else
      : > "$STATE/failed/$id"
      echo "$(stamp) worker $WORKER: FAILED $id" >> "$P"
    fi
  fi
done
