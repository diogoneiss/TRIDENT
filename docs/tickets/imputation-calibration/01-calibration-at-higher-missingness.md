# Does the calibration help at 40% missing and above? (pre-registered 2026-10-16)

This ticket was written and committed before the first cell (`5634858`, 10:55 GMT-3 on
2026-10-16). The queues started at 10:55:52 GMT-3, so about 13:30 GMT-3 was the expected end. The user chose this
study on 2026-10-16 ("vamos no 1, pode iniciar agora") from the threads left open after ADR 0019.

## Why

The calibration was studied in E34 and E35 (tickets
[imputation-token-shape/04](../imputation-token-shape/04-checkpoint-and-calibration.md) and
[05](../imputation-token-shape/05-checkpoint-replication.md)). After training, it is fitted on the
validation rows' own gaps:

- each numerical column is shrunk toward the training mean by a fitted α;
- low-confidence categories are answered with the mode.

On the checkpoint chosen on the validation gaps (K, today's default, ADR 0015), the calibrated
readout KC behaved differently by missingness:

- **At 20nan it fails:** "K better" on spambase_20nan in both seed sets, and on kr-vs-kp_20nan in
  E35. There the model is already nearly calibrated (mean α 0.85 to 0.98).
- **At 40nan and above it was never worse.** Of those 12 variants, both earlier seed sets found
  "KC better" on 6, one set on 3, and neither on 3. None was "K better" in either set:

  | | variants |
  |---|---|
  | **KC better in both sets** | kr-vs-kp_80nan, spambase_60nan, spambase_80nan, credit-g_60nan, credit-g_80nan, vehicle_60nan |
  | **KC better in one set** | kr-vs-kp_60nan, spambase_40nan, credit-g_40nan |
  | **in neither set** (all leaning toward KC) | kr-vs-kp_40nan, biodeg_60nan, kc2_60nan |

  The gains are about 0.3% to 1.1%. They fall where the model is closest to the mean/mode fill:
  credit-g_80nan is still above 1.0 under today's defaults.

The candidate rule, chosen from E34 and E35: **calibrate when the variant has 40% missing or
more, and leave the 20nan variants alone.** This study tests the rule on seeds neither earlier
study used.

## Design (fixed before launch)

- **Cells:** one run per (variant, seed), in today's imputation configuration as a default run
  reads it (ADRs 0013 to 0015 and 0019; none of these 12 variants has a promoted file), plus
  `--score_calibrated`. Runner: `scripts/experiments/calibration_run.py`.
- **Two arms in each run:** K is the run's headline `impute/induced/*`. KC is the calibrated
  readout `impute/induced/calibrated/*` on the same checkpoint. The pairs are within-run.
- **Variants:** the 12 of ADR 0007's 21 at 40nan or more: kr-vs-kp, spambase and credit-g at
  40/60/80nan, and vehicle_60nan, biodeg_60nan and kc2_60nan.
  - The 9 variants at 20nan are not run, because the rule leaves them unchanged by construction.
  - Two named sets are fixed now: **the six named variants** are those where both earlier sets
    found KC better (the first row of the table above).
- **Seeds:** 404, 505 and 606. The store holds no imputation run on any of them.
- **Size:** 36 cells, tagged `experiment=calibration-2026-10-16`.
- **Running it:** four queues (`scripts/experiments/run_calibration_queue.sh 1..4`), one trainer
  each, with CPU threads capped at 4.
  - Cells are assigned longest first by E35's cell times: about 144 estimated minutes a queue, so
    about 2.5 h of wall time.
  - No cell starts after 18:00 GMT-3 on 2026-10-16. A cell not started by then is reported
    missing.
  - Nothing is rerun or extended on its result. A crash is fixed, its cell rerun from the start,
    and the rerun noted here.
- **Check before launch:** the runner on credit-g_40nan at seed 101 (`--no-mlflow`) must equal
  E39's D cell of that seed on every headline model metric and every baseline column. That shows
  the calibrated readout leaves the headline untouched, and that the runner is a default run.
  **Passed:** fold by fold, 15 headline model metrics and 60 baseline columns are equal. Seed 101
  is not a study seed.

## Measures

- **Primary:** per variant, KC − K on the induced `impute_score`, paired by seed and fold
  (15 pairs), with a t-interval. A verdict needs the interval to exclude zero and all three seeds'
  mean differences to agree in sign.
- **Reading, fixed now:**
  1. **Supported:** "KC better" on at least 4 of the 6 named variants, and "K better" on none of
     the 12. The recommendation is then to make the calibration the imputation output at 40%
     missing or more, by an ADR and the user's decision.

     Today the calibration exists only as a diagnostic readout. Making it the output needs a code
     change behind a flag, a tag and a backfill, which needs its own authorization. It also needs
     to decide how a run knows its missing level: the variant's name, or the measured missing rate
     of its table.
  2. **Refuted:** "K better" on any of the 12.
  3. **Mixed:** anything else.
- **Secondary, descriptive:**
  - the verdict counts over all 12;
  - the mean KC − K per level (40, 60, 80);
  - the mean α and the thresholded categorical columns;
  - how many variants each arm puts below the best ADR 0007 baseline (credit-g_80nan crossing
    1.0 or not);
  - minutes a cell.
- **Analysis:** `scripts/experiments/calibration_report.py`. `--tally` prints the reading and the
  secondary counts.

## Known limits, stated in advance

- **Within-run pairs,** as in E34 and E35: the intervals are narrow, so a small effect can earn a
  verdict.
- **The 40% threshold and the six named variants come from E34 and E35.** That is the point of a
  confirmation, so they are not tested anew. The 20nan half of the rule rests on those two studies
  alone.
- **The validation gaps do double duty.** They choose the checkpoint (ADR 0015) and fit the
  calibration, so both use the same cells. The test split stays clean.
- **The bar is ADR 0007's four baselines.** The lgbm baseline of ADR 0017 is not in the run's
  fold files.
- **Multiplicity:** 12 primary tests, with the seed-agreement rule.

## Outcome (2026-10-16, 17:50 GMT-3)

**Run.** All 36 cells ran from 10:55 to 13:54 GMT-3: 2 h 58 min against the 2.5 h planned
(1.19x). None failed and none is missing.

**Table.** Numbers come from `scripts/experiments/calibration_report.py`; the table was built from
it by script.

- Lower is better; a negative difference favours KC.
- Each row is the induced `impute_score` over 15 within-run fold pairs.
- ★ marks the six named variants.
- "Mean α" is the numerical shrinkage. kr-vs-kp has no numerical column, so its calibration only
  thresholds categories.

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

**Reading, by the rule fixed in advance: supported.** KC is better on all 6 named variants and K
is better on none of the 12. The recommendation is to make the calibration the imputation output at
40% missing or more, by an ADR and the user's decision. That needs a code change, which needs its
own authorization.

**Secondary measures:**

- **Counts:** KC is better on 7 of the 12 (the six named and biodeg_60nan), with no detectable
  difference on 5 and K better on none.
- **By level, the gain grows with missingness:**

  | level | mean KC − K | fold pairs |
  |---|---|---|
  | 40nan | −0.0009 | 45 |
  | 60nan | −0.0037 | 90 |
  | 80nan | −0.0059 | 45 |

  **No 40nan variant earns a verdict.** kr-vs-kp_40nan moves +0.01%, spambase_40nan −0.13% and
  credit-g_40nan −0.17%. Every verdict falls at 60nan or 80nan. The rule's threshold of 40 is
  therefore safe (never worse) but buys nothing measurable at 40. A threshold of 60 would capture
  every gain seen here. That observation is made after the data, so it would need its own check
  before it replaced the pre-registered threshold.
- **Size:** the changes run from −1.27% (vehicle_60nan) to +0.34% (kc2_60nan, no verdict).
- **Below the best baseline:** 11 of 12 under both arms. The calibration narrows credit-g_80nan's
  gap without closing it (1.0112 → 1.0073 against the mean/mode fill's 1.0).
- **Mean α** falls with missingness, from about 0.80 to 0.85 at 40nan to 0.60 at 80nan on credit-g
  and spambase. vehicle_60nan, biodeg_60nan and kc2_60nan stay at 0.88 to 0.94.
- **Agreement with E34 and E35:** all six named variants replicate on a third seed set.
  biodeg_60nan, which neither earlier set called, is now "KC better". kc2_60nan leans K on two
  seeds without a verdict.
