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

## Note 2026-10-16, about 23:45 GMT-3: the driver, and two passes instead of one

- **The driver landed:** `scripts/experiments/baseline_budget_run.py`, with the imputers in
  `src/training/tuned_baselines.py`.
  - A unit test shows each grid's first point (the family's current setting) fills exactly as the
    live baseline does.
  - On the large tables, the tuned hgb's early-stopping split is drawn by original row, as fixed
    above, so its first point differs there from the live `hgb`, whose split is random.
- **The smoke, on kc2_60nan at seed 707:** 21 min for 5 folds on a loaded machine. Only the run
  time and the chosen settings were looked at, no score.
- **The pass runs as two passes,** without changing any definition:
  1. **The weak tier** (knn, hgb, lgbm), which the tuned bar and E44's primary need. It runs first,
     on E44's folds.
  2. **missForest** (`--families missforest`, written to `csv_missforest/`), the strong tier's
     tuned member. It runs later.

## Outcome (2026-10-17, 23:31 GMT-3): the weak tier's pass

**Run.** The weak tier's pass (knn, hgb and lgbm) ran in 4 CPU lanes:

- E44's folds first, from 23:39 GMT-3 on 2026-10-16 to 03:23 on 2026-10-17;
- the development cells until 15:43 GMT-3.

All 207 (variant, seed) files exist. They took 55.8 process-hours at 4 threads each, against the
25–40 CPU-h estimated.

- **CPU time per family** was not recorded; only the time per file is in the lane logs.
- **The missForest pass** (`--families missforest`, secondary) started at 22:52 GMT-3 on
  2026-10-17, on the user's go. It is appended when it ends.

**The printout** (`scripts/experiments/baseline_budget_report.py`): each line is a mean difference
over the 15 fold pairs of the induced `impute_score` (lower is better), with the verdict rule and
the 1% SESOI. "budget" is the tuned bar minus the untuned bar, in percent of the untuned bar.

```text
## Seeds 101/202/303 (pre-registered)
- `kr-vs-kp_20nan`: tuned bar vs bar -0.0122 better; K vs bar -0.0253 better; K vs tuned bar -0.0131 none; K vs masked-selection tuned bar -0.0206 better; masked: tuned bar vs bar -0.0182 better; budget -2.0% of the bar; current setting kept knn 13%, hgb 7%, lgbm 13%
- `kr-vs-kp_40nan`: tuned bar vs bar -0.0221 better; K vs bar -0.0446 better; K vs tuned bar -0.0225 better; K vs masked-selection tuned bar -0.0273 better; masked: tuned bar vs bar -0.0183 better; budget -2.9% of the bar; current setting kept knn 0%, hgb 0%, lgbm 7%
- `kr-vs-kp_60nan`: tuned bar vs bar -0.0517 better; K vs bar -0.0379 better; K vs tuned bar +0.0137 worse; K vs masked-selection tuned bar +0.0037 none; masked: tuned bar vs bar -0.0507 better; budget -5.7% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `kr-vs-kp_80nan`: tuned bar vs bar -0.0225 better; K vs bar -0.0185 better; K vs tuned bar +0.0040 none; K vs masked-selection tuned bar -0.0032 none; masked: tuned bar vs bar -0.0140 none; budget -2.2% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `spambase_20nan`: tuned bar vs bar -0.0076 better; K vs bar -0.0068 none; K vs tuned bar +0.0009 none; K vs masked-selection tuned bar -0.0002 none; masked: tuned bar vs bar -0.0094 better; budget -0.9% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `spambase_40nan`: tuned bar vs bar -0.0189 better; K vs bar -0.0317 better; K vs tuned bar -0.0128 better; K vs masked-selection tuned bar -0.0155 better; masked: tuned bar vs bar -0.0216 better; budget -2.0% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `spambase_60nan`: tuned bar vs bar -0.0270 better; K vs bar -0.0409 better; K vs tuned bar -0.0139 better; K vs masked-selection tuned bar -0.0224 better; masked: tuned bar vs bar -0.0407 better; budget -2.8% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `spambase_80nan`: tuned bar vs bar -0.0094 better; K vs bar -0.0154 better; K vs tuned bar -0.0060 better; K vs masked-selection tuned bar -0.0079 better; masked: tuned bar vs bar -0.0164 better; budget -0.9% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `credit-g_20nan`: tuned bar vs bar -0.0189 better; K vs bar -0.0387 better; K vs tuned bar -0.0198 better; K vs masked-selection tuned bar -0.0234 better; masked: tuned bar vs bar -0.0221 better; budget -2.1% of the bar; current setting kept knn 7%, hgb 7%, lgbm 7%
- `credit-g_40nan`: tuned bar vs bar -0.0262 better; K vs bar -0.0436 better; K vs tuned bar -0.0175 better; K vs masked-selection tuned bar -0.0312 better; masked: tuned bar vs bar -0.0139 none; budget -2.7% of the bar; current setting kept knn 0%, hgb 7%, lgbm 7%
- `credit-g_60nan`: tuned bar vs bar -0.0197 better; K vs bar -0.0068 none; K vs tuned bar +0.0130 none; K vs masked-selection tuned bar +0.0056 none; masked: tuned bar vs bar -0.0262 better; budget -2.0% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `credit-g_80nan`: tuned bar vs bar +0.0000 equivalent; K vs bar +0.0105 worse; K vs tuned bar +0.0105 worse; K vs masked-selection tuned bar +0.0105 worse; masked: tuned bar vs bar -0.0101 none; budget +0.0% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `vehicle_20nan`: tuned bar vs bar -0.0053 none; K vs bar -0.0530 better; K vs tuned bar -0.0477 better; K vs masked-selection tuned bar -0.0500 better; masked: tuned bar vs bar -0.0099 none; budget -1.1% of the bar; current setting kept knn 53%, hgb 7%, lgbm 0%
- `vehicle_60nan`: tuned bar vs bar -0.0069 better; K vs bar -0.0636 better; K vs tuned bar -0.0567 better; K vs masked-selection tuned bar -0.0584 better; masked: tuned bar vs bar -0.0081 none; budget -1.0% of the bar; current setting kept knn 0%, hgb 13%, lgbm 0%
- `biodeg_20nan`: tuned bar vs bar -0.0087 better; K vs bar -0.0610 better; K vs tuned bar -0.0523 better; K vs masked-selection tuned bar -0.0525 better; masked: tuned bar vs bar -0.0015 none; budget -1.3% of the bar; current setting kept knn 67%, hgb 0%, lgbm 7%
- `biodeg_60nan`: tuned bar vs bar -0.0165 better; K vs bar -0.0978 better; K vs tuned bar -0.0813 better; K vs masked-selection tuned bar -0.0866 better; masked: tuned bar vs bar -0.0194 better; budget -1.9% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `kc2_20nan`: tuned bar vs bar +0.0327 none; K vs bar +0.0148 none; K vs tuned bar -0.0179 none; K vs masked-selection tuned bar -0.0389 none; masked: tuned bar vs bar +0.0228 none; budget +6.3% of the bar; current setting kept knn 73%, hgb 7%, lgbm 0%
- `kc2_60nan`: tuned bar vs bar +0.0209 worse; K vs bar -0.1713 better; K vs tuned bar -0.1922 better; K vs masked-selection tuned bar -0.1929 better; masked: tuned bar vs bar +0.0120 none; budget +2.9% of the bar; current setting kept knn 47%, hgb 0%, lgbm 0%
- `pendigits_20nan`: tuned bar vs bar -0.0071 better; K vs bar -0.0307 better; K vs tuned bar -0.0236 better; K vs masked-selection tuned bar -0.0240 better; masked: tuned bar vs bar +0.0000 equivalent; budget -2.0% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `letter_20nan`: tuned bar vs bar -0.0036 better; K vs bar -0.0391 better; K vs tuned bar -0.0355 better; K vs masked-selection tuned bar -0.0355 better; masked: tuned bar vs bar -0.0078 better; budget -0.8% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `electricity_20nan`: tuned bar vs bar -0.0226 better; K vs bar +0.0161 worse; K vs tuned bar +0.0387 worse; K vs masked-selection tuned bar +0.0388 worse; masked: tuned bar vs bar -0.0184 better; budget -3.8% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
  - tuned bar vs bar: better 17 / no difference 2 / equivalent 1 / worse 1 (fold-pair t, seed agreement)
  - K vs bar: better 16 / no difference 3 / worse 2 (fold-pair t, seed agreement)
  - K vs tuned bar: better 13 / no difference 5 / worse 3 (fold-pair t, seed agreement)
  - K vs masked-selection tuned bar: better 14 / no difference 5 / worse 2 (fold-pair t, seed agreement)
  - masked: tuned bar vs bar: better 12 / no difference 8 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
  - K's verdict changes between the bars: ['kr-vs-kp_20nan (better → none)', 'kr-vs-kp_60nan (better → worse)', 'kr-vs-kp_80nan (better → none)']

## Seeds 404/505/606 (pre-registered)
- `kr-vs-kp_40nan`: tuned bar vs bar -0.0159 better; K vs bar -0.0383 better; K vs tuned bar -0.0224 better; K vs masked-selection tuned bar -0.0322 better; masked: tuned bar vs bar -0.0232 better; budget -2.1% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `kr-vs-kp_60nan`: tuned bar vs bar -0.0476 better; K vs bar -0.0398 better; K vs tuned bar +0.0078 worse; K vs masked-selection tuned bar +0.0020 none; masked: tuned bar vs bar -0.0380 better; budget -5.2% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `kr-vs-kp_80nan`: tuned bar vs bar -0.0162 better; K vs bar -0.0210 better; K vs tuned bar -0.0049 none; K vs masked-selection tuned bar -0.0210 better; masked: tuned bar vs bar -0.0337 better; budget -1.6% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `spambase_40nan`: tuned bar vs bar -0.0188 better; K vs bar -0.0319 better; K vs tuned bar -0.0131 better; K vs masked-selection tuned bar -0.0145 better; masked: tuned bar vs bar -0.0250 better; budget -2.0% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `spambase_60nan`: tuned bar vs bar -0.0292 better; K vs bar -0.0435 better; K vs tuned bar -0.0143 better; K vs masked-selection tuned bar -0.0167 better; masked: tuned bar vs bar -0.0481 better; budget -3.0% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `spambase_80nan`: tuned bar vs bar -0.0096 better; K vs bar -0.0137 better; K vs tuned bar -0.0041 equivalent; K vs masked-selection tuned bar -0.0059 better; masked: tuned bar vs bar -0.0115 better; budget -1.0% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `credit-g_40nan`: tuned bar vs bar -0.0182 better; K vs bar -0.0374 better; K vs tuned bar -0.0192 better; K vs masked-selection tuned bar -0.0225 better; masked: tuned bar vs bar -0.0288 better; budget -1.9% of the bar; current setting kept knn 0%, hgb 7%, lgbm 0%
- `credit-g_60nan`: tuned bar vs bar -0.0142 better; K vs bar -0.0101 none; K vs tuned bar +0.0041 none; K vs masked-selection tuned bar -0.0045 none; masked: tuned bar vs bar -0.0204 none; budget -1.4% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `credit-g_80nan`: tuned bar vs bar +0.0000 equivalent; K vs bar +0.0112 worse; K vs tuned bar +0.0112 worse; K vs masked-selection tuned bar +0.0112 worse; masked: tuned bar vs bar -0.0096 none; budget +0.0% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `vehicle_60nan`: tuned bar vs bar -0.0024 equivalent; K vs bar -0.0634 better; K vs tuned bar -0.0610 better; K vs masked-selection tuned bar -0.0653 better; masked: tuned bar vs bar -0.0007 none; budget -0.3% of the bar; current setting kept knn 0%, hgb 7%, lgbm 7%
- `biodeg_60nan`: tuned bar vs bar -0.0141 better; K vs bar -0.0887 better; K vs tuned bar -0.0746 better; K vs masked-selection tuned bar -0.0786 better; masked: tuned bar vs bar -0.0217 better; budget -1.6% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `kc2_60nan`: tuned bar vs bar +0.0058 none; K vs bar -0.1900 better; K vs tuned bar -0.1958 better; K vs masked-selection tuned bar -0.2050 better; masked: tuned bar vs bar +0.0148 none; budget +0.8% of the bar; current setting kept knn 67%, hgb 0%, lgbm 0%
  - tuned bar vs bar: better 9 / no difference 1 / equivalent 2 / worse 0 (fold-pair t, seed agreement)
  - K vs bar: better 10 / no difference 1 / worse 1 (fold-pair t, seed agreement)
  - K vs tuned bar: better 7 / no difference 2 / equivalent 1 / worse 2 (fold-pair t, seed agreement)
  - K vs masked-selection tuned bar: better 9 / no difference 2 / worse 1 (fold-pair t, seed agreement)
  - masked: tuned bar vs bar: better 8 / no difference 4 / worse 0 (fold-pair t, seed agreement)
  - K's verdict changes between the bars: ['kr-vs-kp_60nan (better → worse)', 'kr-vs-kp_80nan (better → none)', 'spambase_80nan (better → equivalent)']

## Seeds 42/7/13 (run beside them, not pooled)
- `kr-vs-kp_20nan`: tuned bar vs bar -0.0040 none; K vs bar -0.0256 better; K vs tuned bar -0.0216 better; K vs masked-selection tuned bar -0.0245 better; masked: tuned bar vs bar -0.0096 none; budget -0.7% of the bar; current setting kept knn 0%, hgb 7%, lgbm 13%
- `kr-vs-kp_40nan` †: tuned bar vs bar -0.0270 better; K vs bar -0.0143 better; K vs tuned bar +0.0127 worse; K vs masked-selection tuned bar +0.0094 worse; masked: tuned bar vs bar -0.0245 better; budget -3.5% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `kr-vs-kp_60nan`: tuned bar vs bar -0.0474 better; K vs bar -0.0380 better; K vs tuned bar +0.0094 none; K vs masked-selection tuned bar -0.0119 better; masked: tuned bar vs bar -0.0577 better; budget -5.3% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `kr-vs-kp_80nan`: tuned bar vs bar -0.0191 better; K vs bar -0.0241 better; K vs tuned bar -0.0051 none; K vs masked-selection tuned bar -0.0178 better; masked: tuned bar vs bar -0.0111 none; budget -1.9% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `spambase_20nan`: tuned bar vs bar -0.0113 better; K vs bar -0.0183 better; K vs tuned bar -0.0070 none; K vs masked-selection tuned bar -0.0061 none; masked: tuned bar vs bar -0.0079 better; budget -1.3% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `spambase_40nan` †: tuned bar vs bar -0.0214 better; K vs bar -0.0341 better; K vs tuned bar -0.0128 better; K vs masked-selection tuned bar -0.0160 better; masked: tuned bar vs bar -0.0222 better; budget -2.3% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `spambase_60nan`: tuned bar vs bar -0.0294 better; K vs bar -0.0423 better; K vs tuned bar -0.0129 better; K vs masked-selection tuned bar -0.0167 better; masked: tuned bar vs bar -0.0494 better; budget -3.0% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `spambase_80nan`: tuned bar vs bar -0.0093 better; K vs bar -0.0129 better; K vs tuned bar -0.0036 equivalent; K vs masked-selection tuned bar -0.0051 better; masked: tuned bar vs bar -0.0076 better; budget -0.9% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `credit-g_20nan`: tuned bar vs bar -0.0268 better; K vs bar -0.0362 better; K vs tuned bar -0.0094 none; K vs masked-selection tuned bar -0.0220 better; masked: tuned bar vs bar -0.0301 better; budget -2.9% of the bar; current setting kept knn 0%, hgb 0%, lgbm 7%
- `credit-g_40nan` †: tuned bar vs bar -0.0227 better; K vs bar -0.0082 none; K vs tuned bar +0.0145 none; K vs masked-selection tuned bar +0.0057 none; masked: tuned bar vs bar -0.0441 better; budget -2.4% of the bar; current setting kept knn 0%, hgb 0%, lgbm 7%
- `credit-g_60nan`: tuned bar vs bar -0.0081 better; K vs bar -0.0078 none; K vs tuned bar +0.0003 equivalent; K vs masked-selection tuned bar -0.0078 none; masked: tuned bar vs bar -0.0059 none; budget -0.8% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `credit-g_80nan`: tuned bar vs bar +0.0000 equivalent; K vs bar +0.0079 worse; K vs tuned bar +0.0079 worse; K vs masked-selection tuned bar +0.0079 worse; masked: tuned bar vs bar -0.0029 none; budget +0.0% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `vehicle_20nan`: tuned bar vs bar -0.0059 none; K vs bar -0.0552 better; K vs tuned bar -0.0493 better; K vs masked-selection tuned bar -0.0560 better; masked: tuned bar vs bar +0.0041 none; budget -1.3% of the bar; current setting kept knn 60%, hgb 7%, lgbm 13%
- `vehicle_60nan`: tuned bar vs bar -0.0038 none; K vs bar -0.0557 better; K vs tuned bar -0.0519 better; K vs masked-selection tuned bar -0.0520 better; masked: tuned bar vs bar -0.0070 none; budget -0.5% of the bar; current setting kept knn 0%, hgb 7%, lgbm 7%
- `biodeg_20nan`: tuned bar vs bar -0.0110 better; K vs bar -0.0491 better; K vs tuned bar -0.0381 better; K vs masked-selection tuned bar -0.0422 better; masked: tuned bar vs bar -0.0063 none; budget -1.7% of the bar; current setting kept knn 100%, hgb 0%, lgbm 7%
- `biodeg_60nan`: tuned bar vs bar -0.0180 better; K vs bar -0.1052 better; K vs tuned bar -0.0872 better; K vs masked-selection tuned bar -0.0909 better; masked: tuned bar vs bar -0.0210 better; budget -2.1% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `kc2_20nan`: tuned bar vs bar -0.0000 equivalent; K vs bar -0.0251 none; K vs tuned bar -0.0251 none; K vs masked-selection tuned bar -0.0275 none; masked: tuned bar vs bar +0.0000 equivalent; budget -0.0% of the bar; current setting kept knn 100%, hgb 0%, lgbm 0%
- `kc2_60nan`: tuned bar vs bar +0.0071 none; K vs bar -0.1406 better; K vs tuned bar -0.1477 better; K vs masked-selection tuned bar -0.1585 better; masked: tuned bar vs bar +0.0236 none; budget +1.0% of the bar; current setting kept knn 93%, hgb 0%, lgbm 0%
- `pendigits_20nan` †: tuned bar vs bar -0.0075 better; K vs bar -0.0176 better; K vs tuned bar -0.0100 better; K vs masked-selection tuned bar -0.0103 better; masked: tuned bar vs bar +0.0022 none; budget -2.1% of the bar; current setting kept knn 13%, hgb 0%, lgbm 0%
- `letter_20nan` †: tuned bar vs bar -0.0042 better; K vs bar -0.0136 better; K vs tuned bar -0.0094 better; K vs masked-selection tuned bar -0.0094 better; masked: tuned bar vs bar -0.0084 better; budget -0.9% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
- `electricity_20nan`: tuned bar vs bar -0.0320 better; K vs bar +0.0121 none; K vs tuned bar +0.0441 worse; K vs masked-selection tuned bar +0.0437 worse; masked: tuned bar vs bar -0.0238 better; budget -5.3% of the bar; current setting kept knn 0%, hgb 0%, lgbm 0%
  - tuned bar vs bar: better 10 / no difference 4 / equivalent 2 / worse 0 (fold-pair t, seed agreement)
  - K vs bar: better 12 / no difference 3 / worse 1 (fold-pair t, seed agreement)
  - K vs tuned bar: better 7 / no difference 5 / equivalent 2 / worse 2 (fold-pair t, seed agreement)
  - K vs masked-selection tuned bar: better 11 / no difference 3 / worse 2 (fold-pair t, seed agreement)
  - masked: tuned bar vs bar: better 7 / no difference 8 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
  - K's verdict changes between the bars: ['kr-vs-kp_60nan (better → none)', 'kr-vs-kp_80nan (better → none)', 'spambase_20nan (better → none)', 'spambase_80nan (better → equivalent)', 'credit-g_20nan (better → none)', 'credit-g_60nan (none → equivalent)', 'electricity_20nan (none → worse)']

The tuned bar's family, over every fold read: hgb_tuned 41%, lgbm_tuned 39%, knn_tuned 15%, mean_mode 6%
```

**Reading, as fixed (descriptive).**

- **The budget lowers the bar on most variants, more than the review guessed.** The tuned bar is
  better than the untuned bar of 5:
  - on 17 of 21 variants on 101/202/303 and on 9 of 12 on 404/505/606;
  - typically by 1–3% of the bar, at most by 5.7% (kr-vs-kp_60nan; 5.2% on 404/505/606).
  - It is never worse on 404/505/606, and worse once on 101/202/303: kc2_60nan, +2.9%. There,
    selection on kc2's few validation gaps picks settings that do not transfer. kc2_20nan
    (+6.3%) has no verdict.
- **K against the tuned bar,** beside K against the untuned bar:

  | seeds | K − untuned bar | K − tuned bar |
  |---|---|---|
  | 101/202/303 | better 16 / none 3 / worse 2 | better 13 / none 5 / worse 3 |
  | 404/505/606 | better 10 / none 1 / worse 1 | better 7 / none 2 / equivalent 1 / worse 2 |

- **The variants whose verdict changes between the bars** (pre-registered sets):
  - kr-vs-kp_60nan: better → worse, on both sets;
  - kr-vs-kp_80nan: better → no difference, on both sets;
  - kr-vs-kp_20nan: better → no difference (101/202/303);
  - spambase_80nan: better → equivalent (404/505/606).
  - So kr-vs-kp, the all-categorical table, is where the budget catches up with the model.
- **The masked-selection sensitivity** (the deployable way to tune the baselines): K is better
  than that bar on 14 of 21 and 9 of 12. Tuning on the masked cells buys less than tuning on the
  validation gaps, as expected of the deployable signal.
- **The masked population:** the tuned bar is better on 12 of 21 and 8 of 12, never worse.
- **The current settings are rarely kept.**
  - hgb and lgbm keep their current setting on at most 13% of folds, mostly 0%.
  - knn keeps k=5 on the small tables (vehicle_20nan 53%, biodeg_20nan 67%, kc2 47–100%).
  - The tuned bar's family, over every fold read: hgb 41%, lgbm 39%, knn 15%, mean/mode 6%.
- **42/7/13** (run beside, not pre-registered, not pooled) reads the same way: K against the tuned
  bar is better on 7 and worse on 2 of 16, and K's verdict changes on 7 variants.

**What this means for the thesis (D4, already applied in E44's rule):** a development-set win is
counted only where K beats both bars. That is 13 of 21 variants on 101/202/303 and 7 of 12 on
404/505/606, instead of 16 and 10. No default changes.

**For the user:** whether the development thesis table
(`docs/reports/2026-10-17-thesis-table-development.md`) gains the tuned-bar column.

