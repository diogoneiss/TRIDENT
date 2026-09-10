# 03. Imputation metrics as pure functions

Status: done (this commit) on `feat/imputation-task`
Blocked by: none
Plan task: 3. ADR 0004 decision 6. Wayfinder tickets 01, 05, 14.

## Goal

A pure module that turns per-cell actuals and imputations into the reported metrics and
the fold-ranking scalar, with mean/mode baselines from the training fold, correct on
mixed, all-numerical and all-categorical tables, and safe when a baseline is zero.

## Seams under test

- `mean_mode_baselines(train_frame, numerical_columns, categorical_columns)` in a new
  `src/training/imputation_metrics.py`: per-column mean (numerical) and mode
  (categorical) from the **training** values only.
- `score_cells(cells, baselines) -> ImputationScores`: from a per-cell table (row, column,
  type, actual, imputed, and for categoricals the predicted id) to `rmse_num_z`,
  `mae_num_z`, `acc_cat`, `macro_f1_cat`, `n_num_cells`, `n_cat_cells`, `impute_score`,
  plus per-column rows for the artifact.

## Acceptance criteria

- [x] Mixed table: hand-computed literals for every metric on a six-cell example.
- [x] All-numerical input: `impute_score` equals the numerical ratio alone (`w_cat = 0`).
- [x] All-categorical input: `impute_score` equals the categorical ratio alone (`w_num = 0`).
- [x] Constant column (zero baseline): finite score, no exception; the rule is documented
      in the docstring (ratio is 1.0 when the model error is also zero, else the raw error).
- [x] `macro_f1_cat` with a class absent from a fold uses `zero_division=0`.
- [x] Pooled and per-column numbers agree under uniform counts.
- [x] Baselines are computed from the training values passed in, never from the scored
      cells (a test where the two differ).

## Constraints

- Numerical metrics are in scaled (z) space; original-unit numbers are ticket 07's
  concern and never rank.
- `impute_score` weights are the fractions of scored cells per type and sum to 1.

## Comments

- 2026-09-10, eight red-green slices in `tests/unit/test_imputation_metrics.py`:

  | Slice | Behaviour pinned |
  |---|---|
  | 1 | A baseline is the training fold's own mean or mode, ignoring its missing cells |
  | 2 | Error is pooled over the scored cells of each kind |
  | 3 | The ranking score is the share of the naive imputer's error left, weighted by cell counts |
  | 4 | A column the naive imputer never gets wrong still ranks |
  | 5 | A table of one kind of column is scored by that kind alone |
  | 6 | A category the model never reaches for scores zero rather than nothing |
  | 7 | The baseline comes from training, not from the cells being scored |
  | 8 | Each column reports its own error beside the pooled one |

  **Interface refinement.** Categorical values travel as their category text rather than
  encoder ids. Metrics only need equality and labels, so ids buy nothing, while text lets
  `mean_mode_baselines` work without a `LabelEncoder` and gives ticket 07's artifact
  readable values for free.

  **Correction inside slice 4.** Slice 3's implementation contained a zero-baseline guard
  written without a test driving it, returning 0.0 when both the model and a flawless
  baseline are perfect. Slice 4's test showed that to be wrong: matching a flawless
  baseline is parity, so the ratio is 1.0, and falling short of one is 1.0 plus the
  model's own error. Scoring 0.0 would have claimed the model beat a baseline that cannot
  be beaten.

  **Checked against real data**, not only literals. Scoring the naive imputer against
  itself on `credit-g_20nan` fold 1 (860 categorical and 488 numerical held-out hidden
  cells) reproduces the numbers measured by hand during ticket 04 and lands the score on
  exactly 1.0, which is what "no better than naive" has to mean:

  | | value |
  |---|---|
  | `acc_cat` | 0.587, matching the hand measurement |
  | `rmse_num_z` | 0.963, matching the hand measurement |
  | `impute_score` | 1.000000 exactly |

  The per-column table produced 80 rows over 20 columns and names `purpose` (0.290),
  `checking_status` (0.339) and `property_magnitude` (0.361) as the columns a naive
  imputer finds hardest, which is the diagnostic ticket 07 wants it for.
