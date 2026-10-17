# An equal, validation-only tuning budget for the baselines (E42; pre-registered 2026-10-16)

This ticket was written and committed before any tuned baseline was scored, on 2026-10-16 at 22:47
GMT-3 (`5278337`). It freezes the grids, the selection signal and the reading. The driver is written
after E41's module. The plan is
[2026-10-17-experiment-improvements](../../plans/2026-10-17-experiment-improvements.md) (§2c, §3,
D4 and D8), accepted on 2026-10-16.

## Why

Threat 1 of the 2026-10-16 review. The model's configuration was chosen over E24–E39 on these
variants, while every baseline runs at fixed library or R defaults (ADR 0007 decision 5, ADR 0016
decision 7).

The review's probe found that one alternative hgb recipe (lr 0.05, 200 iterations, depth 3) lowers
the bar by 0.010–0.035 on 4 of 6 variants. That recipe was picked after seeing seed-42 test scores,
so it proves a stronger bar exists without measuring it. Under ADR 0019, the flip it reported on
kr-vs-kp_40nan is likely gone: E39's arm D sits 0.045 below lgbm, with interval
[−0.054, −0.037].

The user decided (D4) that a win counts only where the model beats both the untuned bar of 5 and
this tuned bar.

## Design (fixed before the first score)

### Grids

Every family's current setting is in its grid.

| family | grid |
|---|---|
| knn | k ∈ {5, 10, 20, 50}; uniform weights, as ADR 0007 |
| hgb | learning rate {0.05, 0.1} × iterations {100, 200, 400} × tree size {`max_depth=3`, the default}. Early stopping as today; the early-stopping split is drawn by original row before the hidden copy is stacked. `early_stopping` is never turned off. |
| lgbm | learning rate {0.05, 0.1} × rounds {100, 200, 400} × leaves {7, 31 (the default)}. The rest as ADR 0017 decision 3. |
| missforest | `max_features` ∈ {1/3 (regression) or sqrt (classification), the default; 1/2} × trees {100, 300}. The R stopping rule; the transform as logged today. |

**Shrinkage, all learned families, at 60nan and 80nan only:** each numerical fill is shrunk toward
the training mean by α ∈ {0, 0.25, 0.5}, crossed with the family's grid.

### Selection (D8)

- **Per fold, on the validation rows' own induced gaps** against the `_00nan` sibling. That is the
  signal the model's default checkpoint (K) uses.
- **Every configuration is fitted on the training fold only.** There is no refit on train plus
  validation, because the model does not refit either.
- **Only the chosen configuration is scored on the test split.** The validation score of every
  configuration is logged, and so is the choice.
- **Sensitivity reading:** the same selection on the masked validation cells, which is deployable.
- **The seam has to be built.** `recompute_folds`'s `fold_scorer` receives only `(train, numerical,
  categorical)`. The driver rebuilds the folds with `build_folds` and slices the sibling itself to
  score the validation gaps.

### Cells

- **The development set on today's default:** the cell map of E41
  ([01](01-linear-baselines.md)) on 101/202/303 and 404/505/606.
- **E44's 15 variants on its seeds 707/808/909,** computed after or beside E44's GPU pass. The
  baselines never see the model.
- **Running it:** gorgona8's CPU, in 4 processes, with the thread constants set to 4. It is
  off-store, and the CSVs go under `/scratch2/diogoneiss/trident-<date>-baseline-budget/`.
- **Estimated cost:** 25–40 CPU-h.

## Measures

**Primary,** per variant and seed set, on the induced `impute_score`, over 15 fold pairs with the
project's verdict rule:

1. **The tuned bar against the untuned bar of 5:** how much the budget buys.
2. **K against the tuned bar:** the verdict that, with K against the untuned bar, makes a win under
   D4.

**Reading, fixed now:** descriptive.

- No default changes. The tuned bar is reported beside the untuned one, and the thesis counts a
  win only where both agree (D4).
- A variant whose verdict changes between the two bars is listed by name.
- The tuned bar is not logged live while it costs about 12× a fit (plan §2b).

**Secondary:**

- the settings chosen per family and fold;
- the masked-selection sensitivity;
- the masked population;
- CPU time per family.

**Analysis:** `scripts/experiments/baseline_budget_report.py`, through the shared helper (T21).

## Known limits, stated in advance

- **The validation gaps' truth is an oracle on natural data,** for the baselines here as for K
  (threat 3). The masked-selection reading is the deployable twin.
- **A larger grid could lower the bar further.** The grids are a declared budget, not a search for
  the best possible imputer.
- **These are development cells,** apart from E44's.

## Note 2026-10-16, about 23:15 GMT-3: the output contract E44 reads

- **Directory and files.** The driver writes one CSV per (variant, seed) under
  `/scratch2/diogoneiss/trident-2026-10-17-baseline-budget/csv/`.
- **Columns.** Each file holds every tuned family's test metrics as
  `impute/<population>/baseline/<family>_tuned/<metric>`, plus the tuned bar as
  `impute/<population>/baseline/tuned_bar/impute_score`. The tuned bar is the tuned weak-tier family
  with the lowest CV mean on the run, as ADR 0007 decision 8 chooses the bar.
- **The confirmation reads it.** `scripts/experiments/confirmation_report.py` (E44) reads these
  columns.
