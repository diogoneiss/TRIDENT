# An offline secondary metric panel; impute_score stays the only primary

## Status

Accepted. The plan of 2026-10-17 proposed it, and the user accepted every recommendation on
2026-10-16 at about 22:45 GMT-3: option B of this ADR and D16. It is written on 2026-10-19, after
the panel had been read on E40, E44, E45 and E46. It amends ADR
[0004](0004-imputation-decoder-task.md) decision 6 (what a run scores) by adding a secondary layer
beside it, offline. `score_cells` and every live key are unchanged. Task T24.

## Context

`impute_score` folds two kinds of cell into one number:

- the RMSE ratio to the mean/mode fill on numerical cells;
- the error ratio on categorical cells;
- weighted by the cells of each kind.

The review of 2026-10-16 named this threat 4: the metric pair decides the missForest and
categorical verdicts, and heavy tails decide kc2 and biodeg. A claim that holds for `impute_score`
could still fail for:

- a reader who scores per kind;
- a reader who scores per column;
- a reader who scores the imputed distribution rather than cell-wise error.

**Three ways to answer it:**

- **A:** `impute_score` only.
- **B:** an offline panel of secondary metrics, with a claim standing only where the per-kind
  readings agree.
- **C:** change `score_cells` so every run logs the panel live. That invalidates all of the
  baseline cache's entries, because the cache key hashes `imputation_metrics.py`.

**What the panel showed, read by script:**

| study | what it found |
|---|---|
| E40 (development, 12 variants at 40nan+) | the ratio metrics never disagree with `impute_score`; the only disagreements are distributional (dispersion, Wasserstein, category TV, abs_bias) on credit-g_40/80nan, kr-vs-kp_60nan and spambase_40/60/80nan |
| E44 (confirmation, 15 variants) | the same: ratios agree everywhere. At 80nan on letter and pendigits, the model's fills are more shrunk than the boosters' (dispersion 0.485 against 0.543, and 0.627 against 0.654). kc2's wins come from the non-twin cells. Dropping the context-free rows changes no verdict. |
| E45 and E46 (MNAR) | every imputer under-predicts under MNAR, and the model's signed bias moves least of 10 methods on 9 of 10 pairs |

## Decision

1. **`impute_score` stays the only primary metric.** It ranks folds (ADR 0004), selects
   checkpoints (ADR 0015) and carries every pre-registered decision rule.
2. **The secondary panel is computed offline** by `scripts/experiments/metric_panel.py` and read by
   `metric_panel_report.py`.
   - **Its sources:** a study's fold ledgers (the model's fills, all five folds) and a refit of
     every baseline family on the same folds.
   - **Per (variant, seed, fold, method), it computes:**
     - the RMSE, MAE and column-normalised numerical ratios;
     - the categorical error and balanced-error ratios;
     - the absolute and signed bias per numerical column, and the share of columns under the truth;
     - the dispersion ratio;
     - the 1-D Wasserstein distance;
     - the category-frequency total variation;
     - `impute_score` on twin cells, other cells and cells of rows with some context, with each
       subset's share.
   - It writes CSVs beside the study, never the store.
3. **The reading rule:**
   - A claim about `impute_score` stands as stated where the panel's per-kind ratios agree with it.
   - Every disagreement, a metric with the opposite verdict, is listed by name in the study's
     Outcome.
   - Balanced accuracy is preferred over macro-F1, which rare-label noise inflates.
4. **Distributional disagreements are reported as caveats, not reversals.** A squared-error
   imputer shrinks its fills toward the centre at heavy missingness. Where the panel shows that,
   the win holds for point accuracy and not for the distribution, and the Outcome says so (E44's
   80nan wins).
5. **The per-type bar** (the best baseline per kind of cell, recombined with `impute_score`'s
   weights) is a reported secondary opponent in `report_common` (ADR 0020), never the bar of
   record.
6. **Option C is revisited only if a panel metric becomes primary.** Even then it would land
   between studies, with a cache warm pass, because it misses every cache entry.

## Consequences

- **Every claim in the thesis about the model against the baselines now has a panel check behind
  it** on the development and confirmation sets (E40, E44). None was reversed. The 80nan wins
  carry the distribution caveat.
- **The panel costs CPU time per study**, mostly the missForest refits: E44's took about a night
  on two lanes, and E45's and E46's together about an hour on four.
- **The ledgers must be kept.** A study that wants the panel keeps its fold ledgers. They are
  written by default under each cell's results directory.
- **`metric_panel.py` changed twice, and each change is in its own commit:**
  - the non-twin subset was empty before `c242341`;
  - the signed bias was added in `e90ccff`.
  - The panels read before those commits lack those columns, and their other columns are
    unchanged.
