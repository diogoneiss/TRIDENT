# Keep missForest outside the baseline bar, and put a single-pass LightGBM in it

## Status

Accepted (2026-10-06, about 15:00 GMT-3), the user's decision the day ADR
[0016](0016-missforest-baselines.md) was applied. It reverses two of 0016's choices and keeps
the rest: missForest stays, computed and cached as 0016 decided, but leaves the bar; the
iterated LightGBM learner (`missforest_lgbm`) is withdrawn; a single-pass LightGBM (`lgbm`),
the `hgb` recipe with LightGBM, joins the bar. ADR 0007's protocol, best baseline and gap
are otherwise unchanged.

## Context

ADR 0016's backfill made missForest the bar on 11 of 21 variants of today's default (E35)
and on 476 of 873 runs, and it would keep doing so: an iterated forest over the training
fold is close to the strongest single imputer a tabular table admits. A bar that is almost
always the same strong imputer stops saying how the model sits among ordinary baselines,
and its margin over the model is within the inductive-versus-transductive band 0016
measured on six variants. The user wants it read on its own, as an external comparison.

0016's LightGBM variant was a misreading: the user had asked for a LightGBM run like
`hgb`, one model per column in a single pass, not missForest with LightGBM. It never set
the bar (worse than the forest on every variant of both populations).

## Decision

1. **The bar is the best of `mean_mode`, `knn5`, `knn10`, `hgb` and `lgbm`.**
   `summary.OUTSIDE_THE_BAR` names what never enters it: missForest (an external baseline,
   `summary.EXTERNAL_BASELINES`), `missforest_lgbm` (withdrawn,
   `summary.WITHDRAWN_BASELINES`) and `best` itself. The live summary and the backfill's
   `best_statistics` read that one constant, so the two cannot drift. `baseline/best`, the
   tag `best_baseline/<population>` and `gap_to_best_baseline*` follow.

2. **missForest gets a gap of its own.** Every cross-validated parent logs
   `impute/<population>/gap_to_missforest/impute_score` and `..._pct/impute_score`, the
   paired per-fold difference and its percentage of missForest's mean: the arithmetic of
   `gap_to_best_baseline` (ADR 0007 decision 9), shared through `summary._paired_gaps`.
   `baseline/missforest/*` keeps its keys.

3. **`lgbm`: the `hgb` recipe, LightGBM learner.** `src/training/lightgbm_baseline.py`
   subclasses `GradientBoostingImputer` and overrides only the model: the training rows
   where the column is observed plus a copy with a fifth of the feature cells hidden,
   gaps routed natively, one pass, categories declared categorical. LightGBM's defaults
   (100 rounds, 31 leaves, learning rate 0.1, at least 20 rows a leaf), seeded, and
   `deterministic=True, force_row_wise=True`: a unit test fills alike at one and at eight
   threads. The one default that differs in practice from `hgb`: scikit-learn stops early
   on folds above 10,000 rows (letter, pendigits, electricity), LightGBM trains all 100
   rounds. The override returns LightGBM models where the parent's type says
   scikit-learn's, through a `cast`: the parent only calls `predict`, and typing it
   honestly would mean editing `imputation_baselines.py`, whose source keys the ADR 0007
   cache.

4. **`missforest_lgbm` is withdrawn.** `missforest.py` keeps the forest alone. Its numbers
   stay on the 873 runs ADR 0016's backfill wrote and on E38's 63 live runs: MLflow deletes
   no metric through its API, and the store is never written by hand. Nothing reads them.

5. **Three cache families:** ADR 0007's four (key unchanged since ADR 0007, so its entries
   stay valid), `lgbm`, and missForest. Editing `missforest.py` invalidated missForest's
   entries; the backfill below recomputes them.

6. **missForest predicts on one thread** (a defect of ADR 0016, found by this ADR's first
   dry run). A forest grown on several threads is the same forest, but scikit-learn also
   *predicts* on them, adding each tree's output into a shared sum in whatever order the
   trees finish: every call moves the fills in the last bits (30 calls out of 30 on a toy
   forest), and where a vote is nearly tied a category flips. ADR 0016's thread test passed
   on data without ties. Caught on `kr-vs-kp_60nan`, seed 7: the logged induced
   `acc_cat` mean was 0.824166, and two recomputations, by the module before and after
   decision 4, both gave 0.824022. The forest is now set to one thread once grown;
   repeated fills are bitwise identical (unit test on a noisy target, red before the
   fix), and so are `lgbm`'s, whose LightGBM sums each row's trees in a fixed order.

7. **The backfill, `scripts/backfill_lgbm_baseline.py`,** replaces
   `scripts/backfill_missforest_baselines.py` (which, left runnable, would re-derive a bar
   with missForest in it). For every imputation parent with ADR 0007 baselines that lacks
   `lgbm` or the mark `lgbm_backfilled`: recompute every baseline through the live cache;
   refuse the run unless its mean/mode means are reproduced within 1e-9 (its own cells);
   **correct** its missForest numbers (parent statistics and the children's fold values)
   wherever the single-threaded recomputation differs from what ADR 0016 logged, and report
   by how much; write `lgbm` on parent and children and `gap_to_missforest*` on the parent;
   choose the bar again and, where it differs from the tag, rewrite `baseline/best`, the
   tag and `gap_to_best_baseline*`. KNN and hgb are recomputed but not checked: on runs
   recorded on the Windows PC their last digits differ by machine (the first dry run found
   KNN 3.4e-4 off on `vehicle_60nan` of 2026-09-25), which is why the cache keys on the
   machine; a bar that moves to one of them takes its fold values from the run's own
   `raw_fold_metrics.csv` and its gap only if they average to the logged means. On the 476 runs ADR 0016 moved to
   missForest that leaves `baseline/best/*` with three step-0 entries (the original, the
   missForest one, this one); MLflow's latest value is this one. `cv_summary.json`
   artifacts are not rewritten: E38's 63 runs name missForest as best there. The parent's
   mark is written last and the mirror index is shared across trees, as in 0016's second
   pass.

8. **Applied after E38.** Every E38 cell is a fresh process reading the checkout, so this
   landed only after its last cell. The user stopped E38 early; its queues exited at 21:12
   GMT-3 with 15 of 63 cells, and ADR 0017 landed at 22:06 GMT-3. E38 is one baseline set
   throughout, and the backfill brings its runs to this ADR with the rest.

## Considered options

- **Leave missForest in the bar.** Rejected by the user: it would be the bar on most
  variants and the comparison would stop saying where the model sits among plain imputers.
- **Keep `missforest_lgbm` as a second external baseline.** Rejected: it was never asked
  for, never set the bar, and costs up to a minute per fold.
- **`lgbm` inside the ADR 0007 family, by giving `GradientBoostingImputer` a learner
  parameter.** Cleaner typing, rejected because that module's source keys the ADR 0007
  cache, and because the backfill must compute `lgbm` without refitting KNN.
- **Filtering `missforest_lgbm` out of the existing missForest cache entries instead of
  editing the module.** Saved a few hours of recomputation at the price of dead code, and
  lost the reproduction check of decision 5.

## Consequences

- Fourteen fold keys per scored population for `lgbm`; fourteen parent keys per population
  for `gap_to_missforest*`; `missforest_lgbm`'s keys stop growing.
- ADR 0016's Outcome table (the bar of six) is superseded by this ADR's.
- `src/mlflow_utils.LGBM_BACKFILLED_TAG` marks the trees the backfill wrote.

## Outcome

Pending: the backfill, after E38.
