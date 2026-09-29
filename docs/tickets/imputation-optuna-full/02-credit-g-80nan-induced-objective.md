# credit-g_80nan ranked by the induced validation gaps (pre-registered 2026-09-28)

Written and committed before the first trial ran.

## Why

In the overnight full-profile studies
([01](01-overnight-full-profile-studies.md)), the `credit-g_80nan` study ranked its trials
by the masked validation cells and chose a configuration whose five-fold induced test
score was worse than the defaults' on every fold (1.160 [1.083, 1.237] against 1.024,
fold-paired +0.137) and lost to the mean/mode fill with disjoint intervals. ADR 0008 lets
a study rank by the validation rows' own gaps instead.

## Question

On `credit-g_80nan`, does ranking by `validation/impute/induced/impute_score` choose a
configuration that does better on the induced test cells than the one the masked
objective chose?

## Design (fixed before launch)

- Identical to study 01's `credit-g_80nan` study except the objective: `--search_space
  full`, 40 trials, `TPESampler(seed=42)`, `--lr_scheduler cosine`, seed 42, the
  predefined single split, **`--search_objective induced`**. No `--promote_best`.
  With the same seed, the sampler's first trials repeat study 01's draws; they diverge
  once TPE starts modelling a different objective.
- The winner is retrained five-fold, seed 42, MLflow on, tagged
  `retrained_from=best_trial` and `selected_by=validation/impute/induced/impute_score`.
- One study, one retrain; nothing rerun or extended on its result.

## Measures

- **Primary:** the retrained winner's `cv/test/impute/induced/impute_score/mean` with its
  95% interval, against the masked-selected winner's retrain
  (`impute_credit-g_80nan_20260928_200740`) and against the defaults run
  (`impute_credit-g_80nan_20260924_225231`), fold-paired over the same five folds, and
  against the run's own best baseline. "Better" only where the fold-paired interval
  excludes zero; "beats the bar" only with disjoint intervals, as in study 01.
- **Secondary:** the same on the masked test cells.
- **Exploratory:** across the 40 trials, the rank correlation between the masked and the
  induced validation scores (both are logged on every trial), and where study 01's winner
  would have ranked under the induced objective.

## Known limits, stated in advance

One seed, one selected trial, one variant; a single split selects the winner. A better
winner here would show the objective can fix the mis-selection seen in study 01 on this
variant, not that it always does.

## Outcome (2026-09-28, study 22:11 to 22:31, retrain to 22:33)

Study `optuna_credit-g_80nan_20260928_221137`, 40 trials; winner trial 15 (validation
induced 0.985). Retrain `impute_credit-g_80nan_20260928_223114`, tagged
`selected_by=validation/impute/induced/impute_score`.

**Primary, induced test cells** (`cv/test/impute/induced/impute_score/mean`, 95% intervals;
differences fold-paired over the same five folds):

| run | score | − induced winner, fold-paired |
|---|---|---|
| induced winner (this study) | 1.017 [0.999, 1.035] | |
| masked winner (study 01) | 1.160 [1.083, 1.237] | induced winner lower by 0.143 [0.062, 0.225], 5/5 folds |
| defaults | 1.024 [1.008, 1.040] | induced winner lower by 0.007 [−0.017, +0.030], 3/5 folds |
| best baseline (`mean_mode`, same run) | 1.000 [1.000, 1.000] | overlaps the bar |

The induced objective **chose better than the masked one**: its winner beats study 01's on
every fold, by an interval that excludes zero. It is level with the defaults, not better,
and it does not beat the mean/mode fill, which at 80% missingness nothing tried so far has.

**Secondary, masked test cells:** 1.004 [0.977, 1.031], level with the masked winner
(−0.034 [−0.138, +0.069]) and with the defaults (+0.009 [−0.043, +0.060]).

**Exploratory.** Across the 40 trials, the masked and induced validation scores are
unrelated (Spearman −0.145, p = 0.37). Study 01's winner was trial 9, one of the first ten
draws, which a same-seed sampler repeats, so this study trained the same configuration: it
ranks **40th of 40** by the induced validation score (1.389). On this variant the masked
objective did not merely fail to find the best induced configuration; it picked the worst.

Winner: `DIM` 136, `HEADS` 8, `LAYERS` 3, `HIDDEN_DIM` 40, `DIM_FEED` 96, `DROPOUT` 0.4,
`BATCH` 512, `PROB_MASCARA` 0.4, `LR_PRE` 4.3e-4, `LR_DECODE` 5.3e-4, `LAMBDA_NUM` 2.55,
50 and 60 epochs. One seed, one variant, one selected trial, as stated in advance.
