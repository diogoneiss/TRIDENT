# Overnight full-profile studies against the baseline bar (pre-registered 2026-09-28)

Written and committed before the first trial ran, so the analysis cannot be chosen after
the results are seen.

## Why

With every baseline of ADR 0007 in the store (backfill of 2026-09-28), the decoder is
below the **baseline bar** (the lowest `impute_score` of `mean_mode`, `knn5`, `knn10`,
`hgb`) on only 6 of 21 variants on the induced population, never with disjoint 95%
intervals, while a baseline clears it with disjoint intervals on 3 (`pendigits_20nan`
and `letter_20nan` by `knn10`, `credit-g_80nan` by `mean_mode`). `hgb` sets the bar on
13 of 21.

Twelve `reduced` studies (decode-stage knobs only, default architecture and epochs) did
not beat the defaults (review addendum 2026-09-15, §3 and §8). The `full` profile, which
also samples the architecture and the pre-training knobs, has run on `credit-g_20nan`,
`credit-g_40nan` and `kr-vs-kp_20nan` only, where its best validation objectives were in
the range of the `reduced` studies at a fifth of the epochs. It has never been validated
out of sample, and never on a table where a baseline wins clearly.

## Question

Does a `full`-profile search find a configuration whose five-fold induced
`impute_score` is below the baseline bar of the same run, on the tables where the
decoder loses?

## Design (fixed before launch)

- Profile `full` (as implemented: architecture, pre-training and decode knobs, both
  stages at 20 to 60 epochs), 40 trials, `TPESampler(seed=42)`, `--lr_scheduler cosine`,
  seed 42, the predefined single split, objective `validation/impute/masked/impute_score`
  (ADR 0005). No `--promote_best`: nothing is written under `datasets/hiperparams/`.
- After each study, its best trial's configuration is retrained **five-fold, seed 42**,
  with MLflow on, as an ordinary parent run tagged `optuna_study_run_id` and
  `retrained_from=best_trial`. That run logs every baseline on the same folds and cells.
- Queue, in this order, one process at a time:
  1. `pendigits_20nan`
  2. `letter_20nan`
  3. `electricity_20nan`
  4. `kr-vs-kp_40nan`
  5. `biodeg_20nan`
  6. `spambase_20nan`
  7. `credit-g_80nan`
- Stopping rule: one pass of the queue. Every study whose five-fold retrain finishes is
  reported, in queue order; nothing is rerun, extended or replaced because of its
  result. A study still running when the user resumes the session is reported as
  unfinished.

## Measures

- **Primary:** per retrained run, `cv/test/impute/induced/impute_score/mean` against the
  run's own baseline bar, with both 95% intervals; "beats the bar" only where the
  model's upper bound is below the bar's lower bound.
- **Secondary:** the same against the default-configuration run of 2026-09-24/25 on the
  same variant (same seed, same folds, fold-paired); the masked population the same way.
- Reported alongside: the winner's validation objective, its rank among the study's
  trials once retrained is not known (one retrain per study), and trial time.

## Known limits, stated in advance

- Selection is on one validation split; the addendum measured a median winner rank of
  18/40 on the test side (§8.3), so one retrained winner per study is a noisy draw, and
  a miss does not show the profile cannot win.
- `full` trains 20 to 60 epochs per stage against the defaults' 300 and 150, so a
  difference from the defaults mixes search space and training budget (§8.4).
- One seed. No claim about seed variance.

## Outcome (interim, 2026-09-28 07:41)

The batch ran detached from 00:51. Two studies finished in the night; the queue is still
running (the `letter_20nan` retrain started at 07:10; `electricity_20nan` and the rest
had not started). A study started at 00:47 and stopped at 00:50, while the launch was
rearranged, left `optuna_pendigits_20nan_20260928_004728` and its `optuna_trial_1` as
`RUNNING` in the store; they are not part of this record.

| variant | study | trials, median | best validation objective | 5-fold retrain |
|---|---|---|---|---|
| `pendigits_20nan` | 2.1 h | 40, 198 s | 0.372 | finished |
| `letter_20nan` | 3.7 h | 40, 271 s | 0.499 | running at 07:41 |
| `electricity_20nan` … `credit-g_80nan` | not started | | | |

**`pendigits_20nan`, pre-registered measures** (`cv/test/impute/<population>/impute_score/mean`,
95% intervals; retrain `impute_pendigits_20nan_20260928_025538`):

| population | tuned model | best baseline (same run) | verdict | defaults (2026-09-25) | tuned − defaults, fold-paired |
|---|---|---|---|---|---|
| induced | 0.345 [0.335, 0.355] | `knn10` 0.348 [0.339, 0.357] | inside the bar's interval | 0.399 [0.384, 0.414] | −0.054 [−0.060, −0.048], 5/5 folds |
| masked | 0.393 [0.385, 0.401] | `knn10` 0.407 [0.398, 0.416] | inside the bar's interval | 0.435 [0.417, 0.452] | −0.042 [−0.055, −0.029], 5/5 folds |

By the pre-registered rule the tuned model **does not beat the bar** on either
population: its intervals overlap `knn10`'s. It does move the decoder from clearly losing
to `knn10` (the defaults' 0.399 against 0.348, disjoint) to level with it, and it beats
the defaults on every fold of both populations, by far more than the seed noise measured
before (0.007 to 0.014, addendum §8.2).

Exploratory, not pre-registered: fold-paired against the same run's baselines, which
share folds and cells with the model, the tuned model is below `knn10` by −0.014
[−0.020, −0.008] on the masked cells (5/5 folds) and by −0.003 [−0.009, +0.002] on the
induced ones (4/5), and below `hgb` by −0.020 and −0.048 (5/5 each).

The winning configuration runs the opposite way to the defaults on almost every axis:
`DIM` 192, `HEADS` 8, `LAYERS` 6, `HIDDEN_DIM` 40, `DIM_FEED` 80, `DROPOUT` 0.1, `BATCH` 64,
`PROB_MASCARA` 0.2, `LR_PRE` 1.5e-5, `WEIGHT_DECAY_PRE` 1.9e-5, `LR_DECODE` 4.1e-4,
`WEIGHT_DECAY_DECODE` 1.1e-3, with 40 pre-training and 60 decode epochs against the
defaults' 300 and 150: a larger, deeper model trained less, on smaller batches, with a
lower mask rate. The `letter_20nan` winner points the same way (`DIM` 144, `HEADS` 16,
`LAYERS` 5, `HIDDEN_DIM` 64, `DIM_FEED` 48, `DROPOUT` 0.1, `BATCH` 64, `PROB_MASCARA` 0.4,
`LR_PRE` 1.0e-5, 50 and 60 epochs). One seed, one selected trial per study; the caveats
stated above apply unchanged.
