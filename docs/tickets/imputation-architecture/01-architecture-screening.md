# Does a deeper, wider or normed encoder impute better? A screening (pre-registered 2026-10-05)

This ticket was written and committed before the first run (the commit and the queues' start
time are in the Running notes). It is task T03 in [TASKS.md](../../TASKS.md), which the user chose
on 2026-10-05 after the token-shape line closed (E36).

## Why

Today's imputation encoder has 2 layers, a feed-forward width of 32 against a model width of 128
(the usual ratio is about 4x, not 1/4), and no LayerNorm after its last layer, although every
layer normalises before its sub-layers (a pre-norm stack, which usually ends with one). Every
winner of the full-profile Optuna studies used 4 to 6 layers. The token-shape and checkpoint
changes (ADRs 0014, 0015) are exhausted, and two variants still lose to the best baseline
(credit-g_80nan, electricity_20nan): the encoder itself is the lever left.

## Design (fixed before launch)

- **The new option.** `ENCODER_FINAL_NORM` (`--encoder_final_norm`, `none` | `layer`) adds a
  LayerNorm after the encoder's last layer, in both stages, since pre-training and the decode
  stage share the encoder. Default `none` for both tasks, so classification and every imputation
  default are unchanged; the tag `encoder_final_norm` is on every run and was backfilled `none`.
  `LAYERS` and `DIM_FEED` are existing hyperparameters.
- **Arms**, each today's configuration (ADRs 0013 to 0015, the promoted file or the defaults)
  with one change, through `scripts/experiments/architecture_run.py`:
  - **L4:** `LAYERS` 4 (default 2);
  - **F256:** `DIM_FEED` 256 (default 32);
  - **LN:** `ENCODER_FINAL_NORM layer`;
  - **ALL:** the three together, so that a combination that advances has been screened.
- **Reference R, read rather than rerun:** E35's cells of the same variant and seed, as their K
  readout, which equals a default run to the last digit (verified for this runner's path below).
- **Variants:** credit-g_20nan, credit-g_80nan, kr-vs-kp_40nan and pendigits_20nan (the four of
  E24 to E29), plus biodeg_60nan, since every gain since E31 concentrated where gaps are many.
  **Seeds:** 42, 7, 13. **Size:** 60 cells, tagged `experiment=architecture-2026-10-05`,
  `arm=L4|F256|LN|ALL`.
- **Pairing.** Folds, evaluation masks and the training masks come from the seed before the model
  exists, so every pair shares them. The LN arm also shares the initialisation, dropout and
  shuffles (a LayerNorm draws nothing); L4, F256 and ALL draw more at initialisation, so their
  pairs are two draws, partly shared. The integrity check holds each pair's baseline scores equal.
- **Running it:** four concurrent queues (`scripts/experiments/run_architecture_queue.sh 1..4`),
  one trainer each, CPU threads capped at 4. Cells are assigned longest first by an estimate,
  E30's arm-M cell time times an arm factor (L4 1.7, F256 1.2, LN 1.0, ALL 2.0, rough guesses),
  about 325 estimated minutes per queue: about 5.4 h of wall time. No cell starts after 08:00
  GMT-3 on 2026-10-06 (11:00 UTC); a cell not started by then is reported missing. Nothing is
  rerun or extended on its result; a crash is fixed, its cell rerun from the start, and the rerun
  noted here.
- **Checks before launch:** the runner with no change (arm R0, `--no-mlflow`) on credit-g_20nan
  (defaults) and kr-vs-kp_40nan (promoted file) at seed 42 must equal E35's K readouts on every
  recorded metric, with equal baselines; a smoke of the LN arm at seed 1 must record
  `ENCODER_FINAL_NORM layer` in its `hyperparameters.json` (its scores not read); the unit suite (mypy included) and both
  regression fixtures must pass.

## Measures

- **Primary:** per arm and variant, arm − R on the induced `impute_score`, paired by seed and fold
  (15 pairs), t-interval; a verdict only when the interval excludes zero and all three seeds' mean
  difference agree in sign. A negative difference favours the arm.
- **Reading, fixed now:**
  1. An arm **advances** if no variant shows "R better" and at least one shows "arm better".
     A negative mean alone does not advance an arm: on five variants with two-draw noise it is a
     coin flip under the null.
  2. If several advance, the one with the most "arm better" verdicts goes to the 21-variant
     confirmation (a tie: the most negative mean difference over the five variants). ALL competes
     like any arm.
  3. If none advances, T03 closes: no architecture change helps at screening scale.
  The confirmation on all 21 variants is a separate pre-registration, with the user's go-ahead;
  a default switch would then need its own ADR.
- **Secondary, descriptive:** the masked population; minutes a cell per arm against R's (what a
  bigger encoder costs); means over the five variants; estimated against real wall time.
- **Analysis:** `scripts/experiments/architecture_report.py` (`--tally` prints the reading and the
  integrity check).

## Known limits, stated in advance

- **Five variants screen, they do not decide.** An arm that advances is confirmed on all 21.
- **The promoted files were tuned at 2 layers and a 32-wide feed-forward** (kr-vs-kp_40nan's among
  the five); a bigger encoder may want a different learning rate or dropout, which this screening
  does not tune.
- **Multiplicity:** 20 primary tests (four arms on five variants) with the seed-agreement rule.

## Running notes

- **Checks before launch, done:** the R0 runs of credit-g_20nan and kr-vs-kp_40nan at seed 42
  equal E35's K readouts on every recorded metric (14 and 10), with equal baselines (20 and 12);
  the LN smoke at seed 1 (no MLflow) recorded `ENCODER_FINAL_NORM layer`, `LAYERS` 2 and
  `DIM_FEED` 32 in its `hyperparameters.json` (scores not read); the unit suite passes (312
  tests, mypy included) and both regression fixtures pass.
- **Tag backfill applied** before launch, 5 min 31 s: 2204 earlier runs of both tasks stamped
  `none`; 2192 have a mirror carrying the tag, the rest being `FAILED` sources or under a deleted
  study.
