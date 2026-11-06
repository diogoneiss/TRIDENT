#!/usr/bin/env bash
# E147 on the record (ticket imputation-protocol/53); pre-registered and run as E149. Kept as it ran: the paths point at
# the state directory's copies of the scripts that now live in scripts/experiments/ (imputer_utility.py, and
# imputer_utility_cells.py there as mech_run.py).
# Superseded at 22:39 GMT-3 by run_imputer_utility_chain_gorgona8.sh before any cell ran (the ticket's amendment).
# E149's night chain (ticket 55), run detached on gorgona8; it drives medusa1 by ssh.
#   1. the 60 MAR/MNAR cells: on medusa1 if its GPUs free up before 01:00 GMT-3, else on gorgona8
#   2. their ledgers copied to the other machine
#   3. the CPU imputers and the downstream fits on medusa1; TabPFN on gorgona8's GPU
#   4. every CSV gathered on gorgona8 and the tally written
set -u
L=/scratch2/diogoneiss/trident-2026-11-05-downstream-utility
D=/scratch/diogoneiss/trident-2026-11-05-downstream-utility
SG=/scratch2/diogoneiss/trident-2026-11-05-mechanisms
SM=/scratch/diogoneiss/trident-2026-11-05-mechanisms
MECH="kr-vs-kp_marf_60nan spambase_marf_60nan credit-g_marf_60nan vehicle_marf_60nan biodeg_marf_60nan kc2_marf_60nan spambase_probit_60nan vehicle_probit_60nan biodeg_probit_60nan kc2_probit_60nan"
export TZ=America/Sao_Paulo
say() { echo "$(date '+%F %H:%M %Z') $*" >> $L/chain_e149.log; }
cd /scratch2/diogoneiss/TRIDENT
say "chain started"

while true; do
  set -- $(ssh medusa1 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' 2>/dev/null | tr '\n' ' ')
  if [ $# -eq 2 ] && [ "$1" -lt 4000 ] && [ "$2" -lt 4000 ]; then HOSTB=medusa1; break; fi
  now=$((10#$(date +%H%M)))
  if [ $now -ge 100 ] && [ $now -lt 1200 ]; then HOSTB=gorgona8; break; fi
  sleep 300
done
say "GPU tier on $HOSTB (medusa1 GPUs: $*)"

if [ $HOSTB = medusa1 ]; then
  Q=8
  ssh medusa1 "cd /scratch/diogoneiss/TRIDENT && mkdir -p $SM/logs && for q in 1 2 3 4 5 6 7 8; do g=\$(( (q - 1) % 2 )); DECODE_METRIC_STATE=$SM QUEUE_STATE=$SM QUEUE_RUNNER=$D/mech_run.py CUDA_VISIBLE_DEVICES=\$g setsid nohup bash scripts/experiments/c15_queue.sh \$q 8 > /dev/null 2>&1 < /dev/null & done; sleep 3"
  progress() { ssh medusa1 "cat $SM/progress.log" 2>/dev/null; }
else
  Q=3
  mkdir -p $SG/logs
  for q in 1 2 3; do DECODE_METRIC_STATE=$SG QUEUE_STATE=$SG QUEUE_RUNNER=$L/mech_run.py CUDA_VISIBLE_DEVICES=0 setsid nohup bash scripts/experiments/c15_queue.sh $q 3 > /dev/null 2>&1 < /dev/null & done
  progress() { cat $SG/progress.log 2>/dev/null; }
fi
for i in $(seq 1 144); do  # at most 12 hours
  sleep 300
  p=$(progress)
  ended=$(grep -c "queue complete\|queue read FAILED\|STOP file" <<< "$p")
  [ $((i % 6)) -eq 0 ] && say "cells done $(grep -c 'cell done' <<< "$p")/60 failed $(grep -c 'cell FAILED' <<< "$p") queues ended $ended/$Q"
  [ "$ended" -ge $Q ] && break
done
p=$(progress)
say "GPU tier ended: cells done $(grep -c 'cell done' <<< "$p")/60 failed $(grep -c 'cell FAILED' <<< "$p")"

mkdir -p $SG/results; ssh medusa1 "mkdir -p $SM/results"
if [ $HOSTB = medusa1 ]; then
  rsync -a --prune-empty-dirs --include='*/' --include='*_cells.csv' --include='raw_fold_metrics.csv' --exclude='*' medusa1:$SM/results/ $SG/results/
else
  rsync -a --prune-empty-dirs --include='*/' --include='*_cells.csv' --include='raw_fold_metrics.csv' --exclude='*' $SG/results/ medusa1:$SM/results/
fi
say "ledgers copied"

ssh medusa1 "cd /scratch/diogoneiss/TRIDENT && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 && (for v in $MECH; do for s in 42 7 13; do for g in trident adr0007 lgbm missforest linear; do echo \"$SM \$v \$s \$g\"; done; done; done | setsid nohup xargs -P 14 -L 1 sh -c '.venv/bin/python -B $D/imputer_utility.py run \$0 \$1 \$2 \$3 > $D/imp_log_\$1_s\$2_\$3.txt 2>&1 || echo FAILED \$1 \$2 \$3 >> $D/imp_failed.txt' > /dev/null 2>&1 &) ; sleep 2"
say "CPU imputers started on medusa1 for the MAR/MNAR variants"

while pgrep -u $(id -u) -f "imputer_utility.py run .* tabpfn" > /dev/null; do sleep 120; done  # the MCAR TabPFN lane, if still running
for v in $MECH; do for s in 42 7 13; do
  now=$((10#$(date +%H%M))); if [ $now -ge 730 ] && [ $now -lt 1200 ]; then say "TabPFN lane stopped at the 07:30 limit before $v s$s"; break 2; fi
  DOWNSTREAM_OUT=$L OMP_NUM_THREADS=4 uv run --python 3.11 --extra tabpfn python -B $L/imputer_utility.py run $SG $v $s tabpfn > $L/imp_log_${v}_s${s}_tabpfn.txt 2>&1 || say "TabPFN FAILED $v s$s"
done; done
say "TabPFN lane for the MAR/MNAR variants ended"

for i in $(seq 1 96); do  # the CPU passes on medusa1, at most 8 more hours
  left=$(ssh medusa1 "pgrep -u \$(id -u) -f '[i]mputer_utility.py ru[n]' | wc -l" 2>/dev/null)  # bracketed, or it matches its own shell
  [ "${left:-1}" = 0 ] && break
  sleep 300
done
rsync -a medusa1:$D/csv_imputers/ $L/csv_imputers/
DOWNSTREAM_OUT=$L uv run --python 3.11 python -B $L/imputer_utility.py tally 2>/dev/null | grep -v 'Warning: Categorical\|Label Mapping\|^[0-9] -> \|^====\|Loaded categorical' > $L/tally_imputers.txt
say "CHAIN DONE: $(ls $L/csv_imputers | wc -l) files, tally in $L/tally_imputers.txt"
