# Does a deeper, wider or normed encoder impute better? A screening (pre-registered 2026-10-05)

This ticket was written and committed before the first run, at 19:20 GMT-3 on 2026-10-05 (commit
`1d1aff1`; the queues started 19:20:51 GMT-3, so about 00:45 GMT-3 on 2026-10-06 is the expected
end). It is task T03 in [TASKS.md](../../TASKS.md), which the user chose
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

## Outcome (2026-10-06, 00:50 GMT-3)

All 60 cells ran (19:20 GMT-3 on 2026-10-05 to 00:39 GMT-3 on 2026-10-06), none failed, none
missing. The integrity check passed: every pair's baseline scores are equal. Numbers from
`scripts/experiments/architecture_report.py` (kept with `--tally` in the state directory). Arm − R
on the induced `impute_score`, 15 fold pairs per variant; lower is better; a verdict is marked in
bold.

| variant | R | L4 − R | F256 − R | LN − R | ALL − R |
|---|---|---|---|---|---|
| `credit-g_20nan` | 0.8818 | +0.0197 [+0.0105, +0.0289] **R better** | +0.0074 [−0.0040, +0.0188] | −0.0027 [−0.0118, +0.0064] | +0.0133 [−0.0008, +0.0274] |
| `credit-g_80nan` | 1.0079 | +0.0053 [−0.0006, +0.0111] | +0.0028 [−0.0022, +0.0077] | −0.0000 [−0.0046, +0.0046] | −0.0057 [−0.0098, −0.0017] **ALL better** |
| `kr-vs-kp_40nan` | 0.7582 | +0.0052 [−0.0053, +0.0156] | −0.0001 [−0.0106, +0.0105] | −0.0074 [−0.0154, +0.0005] | −0.0120 [−0.0222, −0.0017] **ALL better** |
| `biodeg_60nan` | 0.7605 | +0.0054 [−0.0018, +0.0126] | −0.0019 [−0.0156, +0.0119] | +0.0070 [−0.0021, +0.0161] | +0.0077 [−0.0049, +0.0204] |
| `pendigits_20nan` | 0.3324 | −0.0174 [−0.0197, −0.0151] **L4 better** | −0.0116 [−0.0137, −0.0095] **F256 better** | +0.0142 [+0.0117, +0.0166] **R better** | −0.0133 [−0.0155, −0.0111] **ALL better** |

**Tally.** L4: better on 1 (pendigits), worse on 1 (credit-g_20nan). F256: better on 1
(pendigits), worse on none. LN: worse on 1 (pendigits), better on none. ALL: better on 3
(credit-g_80nan −0.006, kr-vs-kp_40nan −0.012, pendigits −0.013), worse on none.

**Reading, by the rule fixed in advance: F256 and ALL advance; ALL goes to the confirmation.**
Both show no "R better" and at least one "better"; ALL has three such verdicts, F256 one. The
confirmation on all 21 variants is a separate pre-registration and waits on the user's go-ahead.

**Secondary measures and caveats.**
- **ALL nearly fails on credit-g_20nan:** +0.0133 [−0.0008, +0.0274], all three seeds positive,
  one hair short of "R better"; its masked population there is "R better" (+0.0172). The depth
  is the likely cause: L4 alone is "R better" on that variant (+0.0197). A confirmation under the
  "no variant worse" rule may well fail on the small tables.
- **The parts do not add up:** LN alone is worse on pendigits (+0.014) and neutral elsewhere, yet
  ALL, which contains it, is better on pendigits and on two variants where no single change was.
- **Mean difference over the five variants:** L4 +0.0036, F256 −0.0007, LN +0.0022, ALL −0.0020.
  The gain is concentrated in pendigits, the largest table (10,992 rows), where depth and width
  both help (L4 −0.017, F256 −0.012).
- **Cost,** minutes a cell summed over the five variants against R's: L4 1.57x, F256 1.17x, LN
  0.99x, ALL 1.82x (pendigits 71 against 38 min a cell).
- **Time.** Planned about 5.4 h of wall time, ending near 00:45 GMT-3; it took 5 h 19 min and
  ended at 00:39 GMT-3, 0.98x the plan (the arm factors were close: 0.95x of the estimate).
