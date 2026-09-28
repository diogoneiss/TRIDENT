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
