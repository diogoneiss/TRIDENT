# Proposal: the calibration as the imputation output at 40% missing and above

**Status: adopted (2026-10-17)** by [ADR 0027](../adr/0027-calibrated-imputation-output-at-40-percent-missing.md),
after E44's confirmation. It was parked on 2026-10-16. The open decisions below were settled there:

- threshold 40%;
- the measured missing rate with a cut at 0.3;
- the masked population left uncalibrated.

Task T20 in [TASKS.md](../TASKS.md).

## The idea

The decode stage keeps the checkpoint chosen on the validation rows' own gaps (K, ADR 0015). The
calibration is then fitted on those same validation gaps (`src/training/calibration.py`):

- each numerical column's prediction is shrunk toward the training mean by a fitted α (a prior of
  30 cells);
- a categorical answer whose confidence falls below a fitted threshold (quantiles 0.1, 0.2, 0.3,
  0.5 of the validation confidences; at least 30 cells) is replaced by the mode.

Today it exists only as a diagnostic readout. `--score_calibrated` logs
`impute/induced/calibrated/*` beside the headline and changes nothing else.

**The proposal:** the imputation output is the calibrated one on tables with 40% missing or more,
and today's output on the others.

## Evidence

**KC − K across three seed sets.** KC is the calibrated readout and K is the same checkpoint
without it. The table shows the induced `impute_score` over 15 within-run fold pairs per cell, as
mean and verdict. A verdict needs the 95% interval to exclude zero and the three seeds to agree;
lower is better, so a negative difference favours KC. The table was built by script from the store
(tags `experiment=selection-2026-10-03`, `selection-replication-2026-10-04` and
`calibration-2026-10-16`).

- In E34 and E35, credit-g_40nan, kr-vs-kp_40nan and spambase_40nan still read their old promoted
  files, which ADR 0019 removed.
- E40 ran only the variants at 40nan or more.

| variant | E34 (seeds 101/202/303) | E35 (seeds 42/7/13) | E40 (seeds 404/505/606) |
|---|---|---|---|
| `pendigits_20nan` | −0.0016 KC better | −0.0015 KC better | not run (20nan) |
| `letter_20nan` | −0.0015 KC better | −0.0018 KC better | not run (20nan) |
| `electricity_20nan` | +0.0003 none | +0.0003 none | not run (20nan) |
| `kr-vs-kp_20nan` | +0.0076 none | +0.0115 K better | not run (20nan) |
| `kr-vs-kp_40nan` | −0.0105 none | −0.0049 none | +0.0001 none |
| `kr-vs-kp_60nan` | −0.0101 KC better | −0.0032 none | −0.0029 none |
| `kr-vs-kp_80nan` | −0.0083 KC better | −0.0067 KC better | −0.0083 KC better |
| `spambase_20nan` | +0.0050 K better | +0.0033 K better | not run (20nan) |
| `spambase_40nan` | −0.0043 KC better | −0.0001 none | −0.0012 none |
| `spambase_60nan` | −0.0024 KC better | −0.0029 KC better | −0.0020 KC better |
| `spambase_80nan` | −0.0060 KC better | −0.0059 KC better | −0.0054 KC better |
| `credit-g_20nan` | −0.0013 KC better | −0.0025 KC better | not run (20nan) |
| `credit-g_40nan` | −0.0084 KC better | −0.0055 none | −0.0016 none |
| `credit-g_60nan` | −0.0083 KC better | −0.0114 KC better | −0.0061 KC better |
| `credit-g_80nan` | −0.0056 KC better | −0.0056 KC better | −0.0040 KC better |
| `vehicle_20nan` | −0.0004 none | −0.0003 none | not run (20nan) |
| `vehicle_60nan` | −0.0083 KC better | −0.0096 KC better | −0.0081 KC better |
| `biodeg_20nan` | +0.0020 none | +0.0006 none | not run (20nan) |
| `biodeg_60nan` | −0.0034 none | −0.0042 none | −0.0047 KC better |
| `kc2_20nan` | −0.0013 none | +0.0032 none | not run (20nan) |
| `kc2_60nan` | −0.0034 none | −0.0033 none | +0.0018 none |

**What the three sets say:**

- **At 40nan and above, "K better" never appears** across the 36 (variant, seed set) readings.
- **Six variants replicate on all three sets:** kr-vs-kp_80nan, spambase_60nan, spambase_80nan,
  credit-g_60nan, credit-g_80nan and vehicle_60nan. biodeg_60nan joins them in E40.
- **At 20nan the calibration fails.** It is "K better" on spambase_20nan in both sets that ran it,
  and on kr-vs-kp_20nan once. That is why the rule leaves 20nan alone, even though pendigits,
  letter and credit-g_20nan gain about 0.15% there.
- **The gain grows with missingness.** In E40 the mean KC − K is −0.0009 at 40nan, −0.0037 at 60nan
  and −0.0059 at 80nan. No 40nan variant earned a verdict in E40.

**E40 in full** (seeds 404/505/606, today's defaults; ★ marks the six variants named in advance;
from [imputation-calibration/01](../tickets/imputation-calibration/01-calibration-at-higher-missingness.md)):

| variant | K | KC | change | best baseline | KC − K [95% interval] (seeds 404, 505, 606) | verdict | mean α | minutes a cell |
|---|---|---|---|---|---|---|---|---|
| `kr-vs-kp_40nan` | 0.7238 | 0.7239 | +0.01% | 0.7667 | +0.0001 [−0.0041, +0.0043] (−0.0038, +0.0019, +0.0021) | none | — | 22.5 |
| `kr-vs-kp_60nan` | 0.8673 | 0.8644 | −0.33% | 0.9105 | −0.0029 [−0.0076, +0.0018] (+0.0009, −0.0068, −0.0029) | none | — | 19.5 |
| `kr-vs-kp_80nan` ★ | 0.9790 | 0.9706 | −0.85% | 1.0000 | −0.0083 [−0.0120, −0.0046] (−0.0081, −0.0057, −0.0112) | KC better | — | 16.0 |
| `spambase_40nan` | 0.9038 | 0.9026 | −0.13% | 0.9374 | −0.0012 [−0.0031, +0.0008] (−0.0012, −0.0015, −0.0007) | none | 0.79 | 44.0 |
| `spambase_60nan` ★ | 0.9391 | 0.9371 | −0.21% | 0.9826 | −0.0020 [−0.0033, −0.0006] (−0.0021, −0.0005, −0.0033) | KC better | 0.72 | 39.6 |
| `spambase_80nan` ★ | 0.9863 | 0.9809 | −0.55% | 1.0000 | −0.0054 [−0.0073, −0.0034] (−0.0063, −0.0059, −0.0041) | KC better | 0.60 | 43.9 |
| `credit-g_40nan` | 0.9239 | 0.9223 | −0.17% | 0.9613 | −0.0016 [−0.0055, +0.0023] (−0.0021, +0.0005, −0.0031) | none | 0.85 | 5.6 |
| `credit-g_60nan` ★ | 0.9899 | 0.9838 | −0.62% | 1.0000 | −0.0061 [−0.0099, −0.0023] (−0.0114, −0.0056, −0.0012) | KC better | 0.72 | 5.8 |
| `credit-g_80nan` ★ | 1.0112 | 1.0073 | −0.39% | 1.0000 | −0.0040 [−0.0055, −0.0024] (−0.0035, −0.0028, −0.0055) | KC better | 0.60 | 5.7 |
| `vehicle_60nan` ★ | 0.6378 | 0.6297 | −1.27% | 0.7021 | −0.0081 [−0.0110, −0.0052] (−0.0087, −0.0065, −0.0091) | KC better | 0.93 | 4.5 |
| `biodeg_60nan` | 0.7738 | 0.7691 | −0.61% | 0.8630 | −0.0047 [−0.0084, −0.0011] (−0.0041, −0.0046, −0.0055) | KC better | 0.88 | 9.9 |
| `kc2_60nan` | 0.5502 | 0.5520 | +0.34% | 0.7402 | +0.0018 [−0.0065, +0.0102] (+0.0040, +0.0023, −0.0008) | none | 0.94 | 3.6 |

The effect is consistent but small, from −0.2% to −1.3% where it earns a verdict. It does not move
any variant across its best baseline: credit-g_80nan stays above the mean/mode fill (1.0112 →
1.0073).

## What adopting it takes

1. **Code, behind the usual gate** (flag + MLflow tag + backfill, old behaviour reachable):
   - a hyperparameter such as `DECODE_CALIBRATION` with values `none`, `gaps` and `auto`, and the
     flag `--decode_calibration`;
   - `auto` calibrates when the table's measured missing rate reaches a threshold;
   - the imputation task default becomes `auto` through `task_defaults` (ADR 0013's layer);
   - classification is untouched;
   - a backfill stamps `decode_calibration=none` on every earlier imputation run.

   The calibrated output must then be what the headline `impute/*` keys score, and what the
   imputed table written by the run contains.
2. **Decisions for the user:**
   - **Threshold:** 40%, as pre-registered and never worse, or 60%, which is where every verdict
     falls. Using 60% was suggested by E40's data, so it would need its own check.
   - **How a run knows its missing level:** the variant's name (`_40nan`) works only on the
     benchmark. The measured missing rate of the training table also works on real data; a cut at
     0.3 separates the 20nan and 40nan+ variants of the 21. Measured rate with a cut at 0.3 is the
     suggestion.
   - **The masked population:** the calibration is fitted on the induced validation gaps and has
     only been scored on the induced test cells. Whether the masked headline should be calibrated
     too is open. The masked cells are a different population, so it would need a check.
3. **Verification:** an `auto` run on a 60nan variant equals E40's KC readout to the last digit,
   and a 20nan run equals today's default. Both regression fixtures stay unedited: credit-g_20nan
   is 20nan, and vehicle_00nan is a classification run.
4. **An ADR,** amending ADR 0015.

## Sources

- [imputation-token-shape/04](../tickets/imputation-token-shape/04-checkpoint-and-calibration.md) (E34);
- [imputation-token-shape/05](../tickets/imputation-token-shape/05-checkpoint-replication.md) (E35);
- [imputation-calibration/01](../tickets/imputation-calibration/01-calibration-at-higher-missingness.md) (E40);
- `scripts/experiments/calibration_report.py`;
- the experimentation log, E34, E35 and E40.
