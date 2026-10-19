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

## Outcome (2026-10-17, 12:24 GMT-3)

**Run.** The four queues started at 10:24 GMT-3, after E44's last cell, on 4915515, and ended
between 11:25 and 11:29 GMT-3 (about 65 minutes against about 70 planned). All 36 cells finished
before the 12:00 GMT-3 cutoff; none failed and none is missing.

**Reading, by the rule fixed in advance: go for the full family.** All three conditions hold:

- the generator's checks passed before launch;
- all 36 cells finished;
- the integrity check holds: G on the MCAR variants equals E35 fold by fold on all 9 cells
  (largest difference 0, `mnar_smoke_report.py --integrity`).

**Tally,** printed by `scripts/experiments/mnar_smoke_report.py --tally`. Each line is a mean
difference over the 15 fold pairs of the induced `impute_score` (lower is better), its 95%
t-interval and the verdict.

```text
=== biodeg
  MNAR G/H - bar: -0.0165 [-0.0232, -0.0097] -> better
  MNAR G/H - missforest: +0.0204 [+0.0111, +0.0296] -> worse
  MNAR G/H - strong: +0.0204 [+0.0111, +0.0296] -> worse
  MCAR G/H - bar: -0.0973 [-0.1112, -0.0835] -> better
  MCAR G/H - missforest: +0.0059 [-0.0126, +0.0244] -> none
  MCAR G/H - strong: +0.0059 [-0.0126, +0.0244] -> none
  (H - bar), MNAR minus MCAR: +0.0809 [+0.0665, +0.0953] -> worse
  (K - H), MNAR minus MCAR (the oracle's optimism): +0.0063 [-0.0007, +0.0132] -> none
  MNAR Mx column-wise - G/H: +0.0097 [+0.0034, +0.0161] -> worse
  MCAR Mx column-wise - G/H: +0.0205 [+0.0130, +0.0281] -> worse
  t2, (Mx column-wise - G/H), MNAR minus MCAR: -0.0108 [-0.0213, -0.0003] -> better
=== kc2
  MNAR G/H - bar: -0.1159 [-0.1420, -0.0898] -> better
  MNAR G/H - missforest: -0.0107 [-0.0460, +0.0245] -> none
  MNAR G/H - strong: -0.0107 [-0.0460, +0.0245] -> none
  MCAR G/H - bar: -0.0961 [-0.1732, -0.0191] -> better
  MCAR G/H - missforest: -0.0581 [-0.1087, -0.0075] -> better
  MCAR G/H - strong: -0.0581 [-0.1087, -0.0075] -> better
  (H - bar), MNAR minus MCAR: -0.0198 [-0.1066, +0.0670] -> none
  (K - H), MNAR minus MCAR (the oracle's optimism): +0.0311 [-0.0073, +0.0695] -> none
  MNAR Mx column-wise - G/H: +0.0118 [-0.0030, +0.0266] -> none
  MCAR Mx column-wise - G/H: +0.0249 [-0.0100, +0.0598] -> none
  t2, (Mx column-wise - G/H), MNAR minus MCAR: -0.0131 [-0.0521, +0.0259] -> none
=== vehicle
  MNAR G/H - bar: -0.0403 [-0.0510, -0.0295] -> better
  MNAR G/H - missforest: +0.0040 [-0.0043, +0.0122] -> none
  MNAR G/H - strong: +0.0040 [-0.0043, +0.0122] -> none
  MCAR G/H - bar: -0.0542 [-0.0685, -0.0399] -> better
  MCAR G/H - missforest: -0.0008 [-0.0186, +0.0169] -> none
  MCAR G/H - strong: -0.0008 [-0.0186, +0.0169] -> none
  (H - bar), MNAR minus MCAR: +0.0140 [-0.0024, +0.0303] -> none
  (K - H), MNAR minus MCAR (the oracle's optimism): -0.0020 [-0.0118, +0.0077] -> none
  MNAR Mx column-wise - G/H: +0.0112 [+0.0051, +0.0173] -> worse
  MCAR Mx column-wise - G/H: +0.0230 [+0.0042, +0.0417] -> worse
  t2, (Mx column-wise - G/H), MNAR minus MCAR: -0.0118 [-0.0335, +0.0100] -> none
```

- **"strong" equals missForest here.** These runs come from the main checkout before ADR 0021's
  implementation, so the linear arms were not fitted on them; the strong tier's best is missForest
  alone.

**What the smoke says, descriptively** (no claim rests on it):

- **The lead over the weak bar survives MNAR on all three tables** (biodeg −0.017, kc2 −0.116,
  vehicle −0.040).
- **It shrinks on biodeg only.** On biodeg the gap to the bar goes from −0.097 under MCAR to
  −0.017 under MNAR, a mechanism effect of +0.081 that excludes zero. Against missForest, biodeg
  goes from no difference to worse (+0.020). On kc2 and vehicle the mechanism effect has no
  verdict.
- **The oracle's optimism does not grow under MNAR.** K − H, MNAR minus MCAR, has no verdict on
  any table. So H stays the right readout here.
- **t2's positive control is weak.** Mx (column-wise, gaps as `[NULL]`) is worse than G's H under
  both mechanisms on biodeg and vehicle, and no different on kc2. Its deficit shrinks under MNAR
  on biodeg (−0.011, excluding zero) and nowhere else. The untied arm does not overtake the tied
  one on any table.
  - Mx changes three things at once (the `[NULL]` gaps, the loss checkpoint and column-wise
    scoring), so this contrast does not isolate the `[NULL]`/`[MASK]` split.
- **The known limit holds:** under MNAR the mean/mode denominator is biased, so only within-variant
  gaps are compared across mechanisms, as above.

**The signed bias per column (measure 5)** was read on 2026-10-19 with E46's, in
[06](06-mnar-family.md) (its last section).

- Every imputer falls further below the hidden values under MNAR.
- The model's bias moves least of the 10 methods on all three of E45's pairs.

**For the user to decide:**

- Whether to run the full family (spambase plus the 20nan rung, about 48 cells, and a tilt dial on
  biodeg), as the rule allows. biodeg is the table worth the dial: the only one where MNAR moves
  the lead.
- Whether the full family should carry the strong tier live, after ADR 0021's merge, so that
  "strong" is more than missForest.

