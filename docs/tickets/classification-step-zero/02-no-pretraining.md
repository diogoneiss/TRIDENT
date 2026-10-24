# Stage 2: what does pre-training buy classification under the fork's defaults? Arms A, B and C on gorgona8 (E71; pre-registered 2026-10-23)

This ticket was written and committed before the study's first cell. Two smoke cells ran before it
with MLflow off, written to a scratch directory and not counted (see "Launch").

- **Why:** Stage 2 of classification step zero
  ([plan](../../plans/2026-10-23-classification-step-zero.md); review item 16). The paper's Table 4
  reports a pre-training gain of about 1.7 macro-F1 points; the fork's default sweep sits within
  0.017 of Table 4's no-pre-training arm at every level (review item 13). No run has measured the
  gain under the fork's defaults.
- **The user's go:** "sim aprovo" (2026-10-23, 23:44 GMT-3), to running Stage 2 after E69.
- **Why arm A is rerun:** Stage 0 (E70, [01](01-determinism.md)) found that HEAD on gorgona8 does
  not reproduce the old sweep (fold differences up to 0.076 macro-F1), so B cannot be paired with
  the sweep. All three arms run here, on one machine and one commit.
- **Code:** `scripts/experiments/cls_stage2_run.py` and `run_cls_stage2_queue.sh`. No `src/` change.

## Design (fixed before the first cell)

- **Every arm** takes the old sweep's `hyperparameters.json` for the variant as
  `hyperparams_override` (one configuration, identical on all 45: DIM 128, 2 layers,
  `EPOCHS_PRE 300`, `EPOCH_FINE 150`, batch 256), with `cosine_legacy` and 3-fold StratifiedKFold.
  - **A:** as is, the fork's default classification run.
  - **B:** `EPOCHS_PRE 0`, no pre-training.
  - **C:** `EPOCHS_PRE 0`, `EPOCH_FINE 450`: as many epochs as A, all of them fine-tuning (as E24
    did for imputation).
- **Cells:** all 45 variants (9 tables × 0/20/40/60/80% MCAR), seeds 42, 7 and 13, so 405 cells.
  - **Order:** every arm of seed 42 first, then seed 7, then seed 13; within a seed, the largest
    tables first, dealt over four GPU queues.
  - **A fixed stop:** the GPU is the user's until 10:00 GMT-3 on 2026-10-24. A `STOP` file at
    09:30 GMT-3 ends the queues after their running cells. The reading uses only the seeds whose
    135 cells are all complete, so the stop depends on the clock, never on results.
  - **Expected:** seed 42 in about 4–5 h on four queues (a rough estimate from the Stage 0 runs'
    time per row), so seeds 7 and 13 may not complete.
- **MLflow on,** tags `experiment=cls-stage2-2026-10-24` and `arm`, with private output and metrics
  directories under `/scratch2/diogoneiss/trident-2026-10-24-cls-stage2/`.
- **The determinism gate (descriptive, not a stop):** before launch, vehicle_40nan, seed 42, arm A,
  MLflow off, against Stage 0's rerun of the same cell on gorgona8. It says whether a fold-paired
  B − A difference on this machine carries run-to-run noise.

## Measures and reading, fixed now (`cls_stage2_run.py --tally`)

On test macro-F1, fold-paired (the arms share their folds), over the complete seeds:

- **Per level:** B − A and C − A, pooled over the 9 tables × 3 folds per seed, with the verdict rule
  (95% t-interval, every seed agreeing in sign), plus the range of the per-table means.
- **Descriptive:** the 27 fold pairs of a level are not independent (3 folds of each table), and
  one seed gives no seed agreement. The interval is a summary, not a test.
- **What each reading means (no decision follows; step zero's Stage 4 is the test):**
  - **A higher at a level:** pre-training helps classification under the fork's defaults there.
  - **No verdict or A lower:** the fork's default gains nothing from pre-training at that level,
    as E24 found for imputation.
  - **C against B** separates "pre-training" from "more epochs".

## Launch

- **Smoke cells (not counted), MLflow off, to a scratch directory:** kc2_00nan, seed 42, arms B and
  C, 23:51 GMT-3. Both ran to the end (`EPOCHS_PRE 0` works under `cosine_legacy`); their fold
  macro-F1 was seen.
- **The determinism gate (23:53 GMT-3):** vehicle_40nan, seed 42, arm A, MLflow off, equals
  Stage 0's rerun of the same cell on every fold, with a largest difference of 0 on accuracy and
  macro-F1, while E69 shared the GPU. So gorgona8 is deterministic run to run, fold-paired
  differences here carry no run noise, and E70's mismatch with the old sweep comes from the
  machine or the environment.
- **Launched** on the commit that records this gate: four GPU queues (`run_cls_stage2_queue.sh`),
  beside E69's two lanes.

## Amendment (2026-10-24, 00:39 GMT-3)

- **A defect in the runner, found and fixed mid-run.** `cls_determinism_run.sweep_run` picked the
  latest classification run with seed 42, 3 folds and `cosine_legacy`, and from 00:15 GMT-3 E71's
  own runs matched it. Seven seed-42 cells started between 00:34 and 00:38 GMT-3 failed before
  training (no MLflow run), reading a newer run's missing artifacts: electricity 40nan A, 60nan A,
  80nan C and 20nan B, and letter 00nan C, 20nan C and 60nan C.
- **The fix (00:38 GMT-3):** the lookup now takes only runs started before 2026-09-12 with no
  `experiment` tag. All 45 variants resolve to the sweep's runs again, and E70's two runs keep the
  same ids.
- **Every cell that trained used the sweep's configuration.** Each cell log prints the run its
  override came from; all of them name the sweep's run of their variant.
- **The seven reruns** go on a repair queue (`repair.sh` in the state directory) beside the four,
  since the queues' lists were fixed at launch. The design and the reading do not change.

## Outcome (2026-10-24, 09:36 GMT-3)

**The run:** 23:54 GMT-3 on 2026-10-23 to 09:36 GMT-3 on 2026-10-24, four GPU queues and the
repair queue. Seeds 42 and 7 complete (135 cells each), seed 13 stopped at 89 of 135 and is not
read. 359 cells done; the 7 failures are the amendment's, all rerun and done.
- **The stop came at 09:34, not 09:30:** the waiter checked the clock every 5 minutes. The last
  running cells ended by 09:36, before the GPU window closed at 10:00. The stop depended on the
  clock only.
- **Two queues ran dry early** (queue 2 at 07:56, queue 4 at 08:11), since the queues' cost
  weights balanced each seed's cells imperfectly; this cost seed 13 time, not correctness.

**The printout** (`cls_stage2_run.py --tally`, unedited):

```text
Complete seeds: [42, 7] of [42, 7, 13]

## B − A, test macro-F1
- 00nan (A's mean 0.8533): B − A +0.0033 [-0.0005, +0.0071] no verdict; tables from credit-g -0.0042 to letter +0.0099
- 20nan (A's mean 0.7942): B − A +0.0033 [-0.0016, +0.0082] no verdict; tables from electricity -0.0037 to kc2 +0.0076
- 40nan (A's mean 0.7337): B − A +0.0036 [-0.0044, +0.0116] no verdict; tables from letter -0.0076 to vehicle +0.0220
- 60nan (A's mean 0.6557): B − A -0.0051 [-0.0118, +0.0017] no verdict; tables from credit-g -0.0117 to biodeg +0.0046
- 80nan (A's mean 0.5414): B − A -0.0050 [-0.0190, +0.0091] no verdict; tables from vehicle -0.0416 to credit-g +0.0415

## C − A, test macro-F1
- 00nan (A's mean 0.8533): C − A +0.0026 [-0.0028, +0.0081] no verdict; tables from biodeg -0.0127 to letter +0.0161
- 20nan (A's mean 0.7942): C − A -0.0042 [-0.0100, +0.0016] no verdict; tables from biodeg -0.0162 to kc2 +0.0112
- 40nan (A's mean 0.7337): C − A +0.0012 [-0.0073, +0.0097] no verdict; tables from letter -0.0102 to vehicle +0.0104
- 60nan (A's mean 0.6557): C − A -0.0042 [-0.0122, +0.0038] no verdict; tables from credit-g -0.0212 to biodeg +0.0243
- 80nan (A's mean 0.5414): C − A -0.0036 [-0.0171, +0.0098] no verdict; tables from vehicle -0.0303 to credit-g +0.0270
```

**Verdict: no detectable pre-training gain at any level.** B − A and C − A have no verdict at every
rate, with means between −0.005 and +0.004 macro-F1.

**Reading:**

- **Pre-training buys nothing measurable under the fork's defaults.** Without pre-training (B) the
  model is within half a point of A at every level, and giving the dropped epochs to fine-tuning
  (C) changes nothing either. This is E24's imputation finding, now on classification.
- **The paper's Table 4 gain of about 1.7 points is outside the intervals at 0–60nan:** a gain that
  size would read B − A ≈ −0.017, and the lower bounds are −0.001, −0.002, −0.004 and −0.012. At
  80nan the interval (−0.019 to +0.009) still includes it. The intervals are summaries, not tests
  (the 54 fold pairs of a level come from 9 tables × 3 folds × 2 seeds).
- **Table by table the sign varies,** most at 80nan (vehicle −0.042, credit-g +0.042 on B − A),
  where fold noise is largest.
- **A reproduces the old sweep's level:** A's means (0.853 / 0.794 / 0.734 / 0.656 / 0.541) are
  within 0.011 of the sweep's (0.858 / 0.799 / 0.742 / 0.661 / 0.531), though not fold by fold
  (E70).
- **What it leads to:** step zero's Stage 4 can keep TRIDENT's pre-training arm or drop it at no
  measured cost; that, with its tuning, is in the plan's decisions. No default changes.

