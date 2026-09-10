# 05. Loss, error metrics and the imputation fold-ranking metric

Type: grilling
Status: resolved
Assignee: Diogo Neiss (with Claude); resolved 2026-09-10
Blocked by: 01, 03

## Question

Given the ground truth (ticket 03) and the literature protocol (ticket 01), what exactly
is computed and logged?

- **Loss** (training signal): CE over categorical `[MASK]` cells plus MSE over numerical
  `[MASK]` cells; how the two are combined (this may already be settled by ticket 04;
  confirm here).
- **Error metrics** (reported): numerical RMSE and/or MAE, in scaled space or original
  units (original units require retaining the `StandardScaler` in `PreparedDataset`);
  categorical accuracy and/or macro-F1; per-column mean vs pooled over cells; a count of
  scored cells per type.
- **Fold-ranking metric**: `f1_macro` is hardcoded in `src/training/summary.py`
  (raises when absent), `src/training/artifacts.py` (diagnostic manifest) and `opt.py`
  (`direction="maximize"`). The imputation task needs its own, and lower-is-better
  metrics invert the best/worst fold selection. Candidates: numerical RMSE alone, a
  combined score, or one metric per column type with numerical RMSE ranking.
- **Metric names**: the `test/<name>` and `cv/test/<name>/...` namespaces; a proposal
  such as `test/imputation/num_rmse`, `test/imputation/cat_accuracy`.
- **Glossary**: generalize *fold-ranking metric* in `CONTEXT.md` to be per task.

## Answer

Resolved 2026-09-10 in one grilling round; every recommendation was accepted.

**Loss.** Unchanged from ticket 04: per-type count-averaged cross-entropy plus
`lambda_num` times mean squared error, no auxiliary embedding term. Confirmed here, not
re-decided.

**Reported metrics**, computed for *both* populations of ticket 03 (self-masked cells on
every variant, induced-missing cells wherever the `_00nan` sibling exists):

| Metric | Definition |
|---|---|
| `rmse_num_z` | Root mean squared error over all scored numerical cells, pooled, in the scaled space the model reconstructs |
| `mae_num_z` | Mean absolute error over the same cells, same pooling |
| `acc_cat` | Accuracy over all scored categorical cells, pooled; argmax over the real vocabulary only |
| `macro_f1_cat` | Mean over categorical columns of that column's macro-F1, `zero_division=0` |
| `n_num_cells`, `n_cat_cells` | Counts of scored cells per type, so any number can be re-weighted or audited |

Pooling within a metric family is the majority convention in the literature; `zero_division=0`
matters because at 60-80% missingness a rare class can vanish from a fold.

**Fold-ranking metric.** Dataset composition, verified while resolving this ticket,
rules out any single-type metric:

| Composition | Datasets |
|---|---|
| All numerical | biodeg, kc2, letter, pendigits, spambase, vehicle |
| Mixed | credit-g (13 categorical / 7 numerical), electricity (1 / 7) |
| All categorical | kr-vs-kp (36 categorical / 0 numerical) |

So the ranking scalar is the research note's baseline-normalised sum:

```text
impute_score = w_num * (rmse_num_z / rmse_num_z_baseline)
             + w_cat * (err_cat    / err_cat_baseline)

err_cat = 1 - acc_cat
w_num, w_cat = fractions of scored cells of each type, w_num + w_cat = 1
```

- **Baselines are mean and mode imputation** computed on the **training fold's** observed
  values for each column and applied at exactly the scored test cells, so nothing leaks.
- **Lower is better**; 1.0 means "no better than imputing the column mean or mode".
- Both terms are dimensionless, which is what makes the sum meaningful across datasets.
- It degrades correctly: on kr-vs-kp `w_num = 0` and it is the categorical ratio alone;
  on the six all-numerical datasets `w_cat = 0` and it is the numerical ratio alone.
- **Guard required**: a constant column can produce a zero baseline. Division by zero
  must be handled explicitly rather than producing an infinity that poisons fold ranking.
- It is computed on the **self-masked** population, which exists on every variant
  including `_00nan`. Induced-missing metrics are the headline number where they exist,
  but they cannot rank folds on `_00nan`.

**Direction is an explicit property of the task**, carried alongside the ranking metric
name, rather than a negated or one-minus variant engineered to keep everything
maximising. `f1_macro` and "maximise" are hardcoded in three places today
(`summary.py` raises when the key is absent and selects best/worst by it, `artifacts.py`
builds the diagnostic manifest ranking, `opt.py` sets `direction="maximize"`). Ticket 09
threads `(ranking_metric, direction)` through all three. A faked maximise form would keep
the code unchanged at the cost of hiding what the number means.

**Metric names.** `FoldResult.metrics` keys are namespaced per population:

```text
impute/masked/rmse_num_z        impute/induced/rmse_num_z
impute/masked/acc_cat           impute/induced/acc_cat
impute/masked/impute_score      impute/induced/impute_score
```

The existing tracking code prefixes fold metrics with `test/` and the cross-validation
parent wraps them as `cv/test/<name>/<statistic>`, so a parent run carries
`cv/test/impute/masked/rmse_num_z/mean` and friends. Nested slashes are already the
house style and are legal MLflow metric names. Induced-missing keys are absent for a
whole run when no sibling exists, never for individual folds, so the summariser's
requirement that all folds share the same keys still holds.

**Per-column detail goes to an artifact, not to tracked metrics.** Spambase has 57
columns; per-column values across two populations and three measures would add roughly
340 series per fold to the store. Pooled metrics are the tracked metrics; per-column
values join the existing per-fold metric artifacts as a table.

Stated rather than asked: the decode stage logs `decode/train_loss`, `decode/val_loss`
and `decode/learning_rate` per epoch, mirroring the two existing stages, and logs
`decode/val_rmse_num_z` and `decode/val_acc_cat` per epoch the way fine-tuning logs
validation macro-F1.

Consequences for other tickets:

- Ticket 09 inherits the `(ranking_metric, direction)` contract, the best/worst inversion
  for lower-is-better metrics, and the decode-stage additions to the loss-band and
  stage-timing key lists.
- Graduated from the fog: ticket 13 (Optuna for the imputation task) and ticket 14 (a
  regression fixture for the imputation task).
- Cleared from the fog: metric granularity in MLflow, decided here; the decode stage's
  loss-band and timing names, now inside ticket 09's scope.

## Comments
