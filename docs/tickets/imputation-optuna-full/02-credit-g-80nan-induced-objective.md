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
