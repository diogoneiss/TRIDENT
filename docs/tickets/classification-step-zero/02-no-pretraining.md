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
