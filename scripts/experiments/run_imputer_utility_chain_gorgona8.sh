#!/usr/bin/env bash
# E147 on the record (ticket imputation-protocol/53); pre-registered and run as E149. Kept as it ran: the paths point at
# the state directory's copies of the scripts that now live in scripts/experiments/ (imputer_utility.py, and
# imputer_utility_cells.py there as mech_run.py).
# E149's night chain after the amendment of 22:39 GMT-3 (the user: "rode tudo na gorgona mesmo"): everything from
# here on gorgona8. The CPU imputers of the 12 MCAR variants had already finished on medusa1 (180 files).
#   1. the 60 MAR/MNAR cells, four queues on gorgona8's GPU
#   2. the CPU imputers and the downstream fits of those variants, and TabPFN after its MCAR lane, on gorgona8
#   3. medusa1's MCAR CSVs copied over and the tally written
set -u
L=/scratch2/diogoneiss/trident-2026-11-05-downstream-utility
D=/scratch/diogoneiss/trident-2026-11-05-downstream-utility
SG=/scratch2/diogoneiss/trident-2026-11-05-mechanisms
MECH="kr-vs-kp_marf_60nan spambase_marf_60nan credit-g_marf_60nan vehicle_marf_60nan biodeg_marf_60nan kc2_marf_60nan spambase_probit_60nan vehicle_probit_60nan biodeg_probit_60nan kc2_probit_60nan"
Q=4
export TZ=America/Sao_Paulo
say() { echo "$(date '+%F %H:%M %Z') $*" >> $L/chain_e149.log; }
cd /scratch2/diogoneiss/TRIDENT
say "chain restarted on gorgona8 alone (amendment)"

mkdir -p $SG/logs
for q in $(seq 1 $Q); do DECODE_METRIC_STATE=$SG QUEUE_STATE=$SG QUEUE_RUNNER=$L/mech_run.py CUDA_VISIBLE_DEVICES=0 setsid nohup bash scripts/experiments/c15_queue.sh $q $Q > /dev/null 2>&1 < /dev/null & done
for i in $(seq 1 168); do  # at most 14 hours
  sleep 300
  ended=$(grep -c "queue complete\|queue read FAILED\|STOP file" $SG/progress.log 2>/dev/null)
  [ $((i % 6)) -eq 0 ] && say "cells done $(grep -c 'cell done' $SG/progress.log)/60 failed $(grep -c 'cell FAILED' $SG/progress.log) queues ended $ended/$Q"
  [ "${ended:-0}" -ge $Q ] && break
done
say "GPU tier ended: cells done $(grep -c 'cell done' $SG/progress.log)/60 failed $(grep -c 'cell FAILED' $SG/progress.log)"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
(for g in linear missforest lgbm adr0007 trident; do for v in $MECH; do for s in 42 7 13; do echo "$SG $v $s $g"; done; done; done \
  | DOWNSTREAM_OUT=$L xargs -P 5 -L 1 sh -c 'uv run --python 3.11 python -B '$L'/imputer_utility.py run $0 $1 $2 $3 > '$L'/imp_log_$1_s$2_$3.txt 2>&1 || echo FAILED $1 $2 $3 >> '$L'/imp_failed.txt') &
CPU=$!
say "CPU imputers started on gorgona8 for the MAR/MNAR variants"

while pgrep -u $(id -u) -f "imputer_utility.py run .* tabpfn" > /dev/null; do sleep 120; done  # the MCAR TabPFN lane, if still running
for v in $MECH; do for s in 42 7 13; do
  now=$((10#$(date +%H%M))); if [ $now -ge 730 ] && [ $now -lt 1200 ]; then say "TabPFN lane stopped at the 07:30 limit before $v s$s"; break 2; fi
  DOWNSTREAM_OUT=$L uv run --python 3.11 --extra tabpfn python -B $L/imputer_utility.py run $SG $v $s tabpfn > $L/imp_log_${v}_s${s}_tabpfn.txt 2>&1 || say "TabPFN FAILED $v s$s"
done; done
say "TabPFN lane for the MAR/MNAR variants ended"
wait $CPU
say "CPU imputers ended; failures listed: $(cat $L/imp_failed.txt 2>/dev/null | wc -l)"

rsync -a medusa1:$D/csv_imputers/ $L/csv_imputers/
DOWNSTREAM_OUT=$L uv run --python 3.11 python -B $L/imputer_utility.py tally 2>/dev/null | grep -v 'Warning: Categorical\|Label Mapping\|^[0-9] -> \|^====\|Loaded categorical' > $L/tally_imputers.txt
say "CHAIN DONE: $(ls $L/csv_imputers | wc -l) files, tally in $L/tally_imputers.txt"
