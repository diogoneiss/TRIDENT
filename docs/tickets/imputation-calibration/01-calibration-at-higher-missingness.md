# Does the calibration help at 40% missing and above? (pre-registered 2026-10-16)

This ticket was written and committed before the first cell, on the morning of 2026-10-16 GMT-3.
The queues start right after the commit; about 13:30 GMT-3 is the expected end. The user chose this
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
