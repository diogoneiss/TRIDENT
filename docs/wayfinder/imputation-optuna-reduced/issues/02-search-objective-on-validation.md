# 02. The search objective is scored on the validation split

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10 (charting session, round 1)
Blocked by: none

## Question

Today a trial's objective is `impute/masked/impute_score` on the **test** half of the
predefined single split: the trial namespace sets no `cv_folds`, the runner returns
`results[0].metrics`, and `decoding.py` scores `fold.test_indices`. Hyperparameters chosen
on the test split leak into every number later reported on that split. Classification's
study has always done this. Keep it for consistency, or score the search on the
validation split?

## Answer

Resolved in one grilling round: **score the imputation search on the validation split**,
gated to the imputation task so the classification study is unchanged.

- The decode stage already encodes a fixed validation mask (`hidden_validation`) and the
  clean validation encoding (`clean_validation`) once per fold for its loss, at the same
  `EVAL_MASK_RATE` and seed the test scoring uses. Scoring the same metrics there is one
  more call to the scorer on tensors that exist; no new masking, no new data path.
- The objective mirrors the fold-ranking metric's *population*: masked cells only. The
  validation split of a `_20nan` variant also holds induced-missing cells, but the
  ranking metric is `impute/masked/impute_score`, so the search objective is its
  validation-split counterpart and nothing else. The induced population stays a test-fold
  report.
- The test split is then unseen by the search and serves `--retrain_best` and the
  comparison in ticket 06 with a clean claim.

**Glossary**: this introduces *search objective* as a term distinct from *fold-ranking
metric*; both are in `CONTEXT.md`. The fold-ranking metric orders CV folds on their test
folds; the search objective orders trials on whatever split the task declares.

**For the plan**: the validation-scored metrics need their own key family so they cannot
collide with the test ones (the metric namespace today has populations under `impute/`
and diagnostics under `impute/masked/rate_NN/`; a validation split is neither). The
`TaskSpec` gains the objective's key and split alongside `ranking_metric`; the objective
wrapper reads that instead of `ranking_metric`. `optuna/objective_value` keeps logging the
objective under one key whatever the task, per ticket 13. Classification's `TaskSpec`
declares its objective as `f1_macro` on the test split, which is today's behaviour spelled
out rather than changed.

## Comments
