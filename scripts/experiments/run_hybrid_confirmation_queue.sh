#!/usr/bin/env bash
# E90/E91 (docs/tickets/imputation-protocol/34-hybrid-confirmation.md) and E93 (35) on gorgona8, from the checkout root:
#   setsid nohup bash scripts/experiments/run_hybrid_confirmation_queue.sh train <1-4> > /dev/null 2>&1 &
#   setsid nohup bash scripts/experiments/run_hybrid_confirmation_queue.sh cells <1-2> > /dev/null 2>&1 &
# A training queue runs its cells from --todo; no cell starts after TRAIN_DEADLINE. A cells lane takes,
# in turn, every cell whose run the store copy holds and whose cells are not written, until all are
# written or CELLS_DEADLINE passes. Stop between cells: touch $S/STOP. S, TRAIN_AT and CELLS_AT override the
# state directory and the two deadlines (E93); STUDY limits both modes to one study (E95, on medusa1).
set -u
MODE=$1; N=$2
export TZ=America/Sao_Paulo OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# A shell started by ssh or setsid has not read ~/.bashrc: log through the leader's server (ADR 0033).
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:47731}
TRAIN_DEADLINE=$(date -d "${TRAIN_AT:-2026-10-29 07:00}" +%s); CELLS_DEADLINE=$(date -d "${CELLS_AT:-2026-10-29 08:40}" +%s)
S=${S:-/scratch2/diogoneiss/trident-2026-10-28-hybrid-confirmation}
P=$S/progress.log
mkdir -p $S/logs
PY="uv run --python 3.11 python -B scripts/experiments/hybrid_confirmation.py"
if [ "$MODE" = train ]; then
  ITEMS=$($PY --todo $N ${STUDY:-}) || { echo "$(date '+%F %T %z') train $N: queue read FAILED" >> $P; exit 1; }
  while read -r st v s; do
    [ -z "$st" ] && continue
    [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') train $N: STOP file" >> $P; exit 0; }
    [ "$(date +%s)" -ge "$TRAIN_DEADLINE" ] && { echo "$(date '+%F %T %z') train $N: deadline, no new cell" >> $P; exit 0; }
    echo "$(date '+%F %T %z') train $N: cell start ${st}_${v}_s$s" >> $P
    if $PY --study $st train $v $s < /dev/null > $S/logs/train_${st}_${v}_s$s.log 2>&1 && grep -q CELL_DONE $S/logs/train_${st}_${v}_s$s.log; then
      echo "$(date '+%F %T %z') train $N: cell done ${st}_${v}_s$s" >> $P
    else
      echo "$(date '+%F %T %z') train $N: cell FAILED ${st}_${v}_s$s" >> $P
    fi
  done <<< "$ITEMS"
  echo "$(date '+%F %T %z') train $N: queue complete" >> $P
  exit 0
fi
# cells lane N of 2: E90's cells, then E91's, alternating between the lanes, the longest first.
ALL=$(uv run --python 3.11 python -B -c "
import sys; sys.path.insert(0, '.')
from scripts.experiments import hybrid_confirmation as h
work = sorted(((h.MINUTES[v.split('_')[0]], st, v, s) for st, (_, _, vs) in h.STUDIES.items() if '${STUDY:-}' in ('', st) for v in vs for s in h.SEEDS), key=lambda w: (w[1], -w[0], w[2], w[3]))
for k, (_, st, v, s) in enumerate(work):
    if k % 2 == $N - 1: print(st, v, s)
")
while true; do
  [ -f "$S/STOP" ] && { echo "$(date '+%F %T %z') cells $N: STOP file" >> $P; exit 0; }
  [ "$(date +%s)" -ge "$CELLS_DEADLINE" ] && { echo "$(date '+%F %T %z') cells $N: deadline" >> $P; exit 0; }
  READY=$(uv run --python 3.11 python -B -c "
import sys; sys.path.insert(0, '.')
from scripts.experiments import hybrid_confirmation as h, report_common as c
runs = {st: c.parents(e) for st, (e, _, _) in h.STUDIES.items() if '${STUDY:-}' in ('', st)}
pending = 0
for line in '''$ALL'''.split(chr(10)):
    st, v, s = line.split()
    if (h.STUDIES[st][1] / 'cells' / f'{v}_s{s}' / 'scale_fold_5.json').exists():
        continue
    pending += 1
    if (v, int(s)) in runs[st]:
        print(st, v, s); break
else:
    print('DONE' if pending == 0 else 'WAIT')
")
  [ "$READY" = DONE ] && { echo "$(date '+%F %T %z') cells $N: lane complete" >> $P; exit 0; }
  [ "$READY" = WAIT ] && { sleep 120; continue; }
  read -r st v s <<< "$READY"
  echo "$(date '+%F %T %z') cells $N: start ${st}_${v}_s$s" >> $P
  if uv run --python 3.11 --extra tabpfn python -B scripts/experiments/hybrid_confirmation.py --study $st cells $v $s < /dev/null > $S/logs/cells_${st}_${v}_s$s.log 2>&1; then
    echo "$(date '+%F %T %z') cells $N: done ${st}_${v}_s$s :: $(grep -o 'DONE.*' $S/logs/cells_${st}_${v}_s$s.log)" >> $P
  else
    echo "$(date '+%F %T %z') cells $N: FAILED ${st}_${v}_s$s" >> $P; sleep 60
  fi
done
