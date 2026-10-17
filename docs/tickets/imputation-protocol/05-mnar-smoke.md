# An MNAR smoke family: value-dependent missingness at the MCAR counts (E45; pre-registered 2026-10-17)

This ticket was written and committed before the first cell.

- **Plan:** [2026-10-17-experiment-improvements](../../plans/2026-10-17-experiment-improvements.md)
  (§2f; review items 8–9; D18). The user authorized free experiment runs until lunch on 2026-10-17.
- **When:** it runs after E44 on the same four GPU queues, never beside it.
- **Code:** `datasets/generate_mnar.py` and the scripts below. No source change.

## Why

Every number so far comes from MCAR holes, where a gap says nothing about the hidden value. RQ1's
MNAR arm and RQ4 ask whether the model's lead survives value-dependent missingness, which is what
clinical data look like. This is the first, smallest such family. Its job is to show the
machinery works and to give a first, descriptive reading; the full family would follow. It also
gives two side readings:

- **t2's positive control:** tied against untied gap tokens, `[MASK]` against `[NULL]`;
- **the oracle's optimism:** K reads the true values of MNAR-hidden validation cells; H does not.

## Design (fixed before launch)

### Data (`datasets/generate_mnar.py`, seed 1060)

- **Variants:** `biodeg_probit_60nan`, `kc2_probit_60nan` and `vehicle_probit_60nan`, from the
  complete tables. All three tables are numerical.
- **How cells are hidden:** every feature column loses exactly `round(0.6 n)` cells, the MCAR
  sibling's count, drawn without replacement with weights Phi(0.5 z − 0) on the standardised
  column, so large values are hidden more often.
  - At a fixed count, weighted sampling is not exactly a probit mechanism: the inclusion
    probabilities are not Phi. This is stated.
- **Checked before launch:**
  - per-column counts equal the MCAR `_60nan` files';
  - every observed cell equals the complete table's;
  - on every column the hidden values' mean is above the observed values' mean;
  - no context-free row on biodeg or kc2 (vehicle has 0, against 1 in its MCAR file).
- **Unit tests:** `tests/unit/test_generate_mnar.py`.
- **Name:** these are not called "self-masked" (CONTEXT.md reserves the term).

### Arms (runner `scripts/experiments/mnar_smoke_run.py`, tag `experiment=mnar-smoke-2026-10-17` plus `arm`)

- **G:** today's default with `--score_checkpoints`.
  - **The headline read is H** (E43 named it the deployable one). On these numerical tables M
    equals H by construction (E43).
  - K is logged and read only for the oracle contrast.
- **Mx:** the gaps shown as `[NULL]` in the decode stage, the loss checkpoint and
  `--score_column_wise`: each gap column asked with the other gaps as `[NULL]`. This is E31's
  shape, plus `--score_checkpoints`.

### Cells, seeds and runs

- **Cells:** both arms on the three MNAR variants and on the three MCAR `_60nan` variants.
- **Seeds:** 42, 7 and 13. 36 cells.
  - **Why these seeds:** G on the MCAR variants must reproduce E35's cells (seeds 42/7/13,
    `decode_checkpoint=loss`) fold by fold, K as E35's `induced_checkpoint` and H as its headline.
    That is the integrity check (`--integrity`).
  - The two variants of a table share their folds (`build_folds` reads only the labels), so
    mechanism contrasts are fold-paired.
- **Running it:** four queues (`scripts/experiments/run_mnar_smoke_queue.sh`), CPU threads capped at
  4, about 70 minutes of wall time.
  - It is launched after E44's last cell.
  - No cell starts after 12:00 GMT-3 on 2026-10-17; unstarted cells are reported missing.
- **Checked before launch, with `--no-mlflow`, passed (2026-10-17, about 02:00 GMT-3):**
  - **G on vehicle_60nan at seed 42** equals E35's cell fold by fold: K equals E35's
    `induced_checkpoint` and H equals E35's headline, on both populations (largest difference
    0.0).
  - **Mx on vehicle_probit_60nan at seed 99** (not a study seed) ran and logged its column-wise
    family. Its scores were not read.

## Measures

**All readings are on the induced `impute_score` over 15 fold pairs,** with the verdict rule,
through `scripts/experiments/mnar_smoke_report.py --tally`:

1. **Per mechanism:** G's H against the weak bar, against missForest and against the strong tier.
2. **The mechanism effect:** (H − bar) under MNAR minus the same under MCAR, fold-paired.
3. **The oracle's optimism:** (K − H) under MNAR minus the same under MCAR.
4. **t2's positive control:** (Mx column-wise − G's H) per mechanism, and its MNAR-minus-MCAR
   difference. If the `[NULL]`/`[MASK]` split matters when missingness is informative, the
   untied arm gains relative to the tied one under MNAR.
5. **Signed bias per column,** offline with `scripts/experiments/metric_panel.py` on the fold
   ledgers (model) and refitted baselines. Under MNAR every imputer is expected to under-predict
   the hidden large values; the question is which degrades least.

**Reading, fixed now:** this is a smoke test, so everything is descriptive.

- **Go for the full family** (spambase plus the 20nan rung, about 48 cells, and a tilt dial on
  biodeg) if all three hold:
  - the generator's checks pass;
  - all 36 cells finish;
  - the integrity check holds.
- **Otherwise:** fix the problem first.

**A known limit, which is why only within-variant comparisons are read:** under MNAR the
mean/mode denominator comes from the training fold's observed values, which the mechanism biases.
So `impute_score` is compared across mechanisms only as a difference of within-variant gaps.

## Launch

- **The watcher:** `launch_after_e44.sh` in `/scratch2/diogoneiss/trident-2026-10-17-mnar-smoke/`,
  started at 02:05 GMT-3. It starts the four queues once E44's four queues have ended, and checks
  only E44's progress log. The launch time is logged in that directory's `progress.log`.
- **The cutoff:** E44 should end at about 10:30–11:30 GMT-3, so the 12:00 GMT-3 cutoff may leave
  some cells unstarted. Those are reported missing, as fixed above.
