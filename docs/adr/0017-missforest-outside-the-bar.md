# Keep missForest outside the baseline bar, and put a single-pass LightGBM in it

## Status

Accepted (2026-10-06, about 15:00 GMT-3), the user's decision the day ADR
[0016](0016-missforest-baselines.md) was applied. It reverses two of 0016's choices and keeps
the rest: missForest stays, computed and cached as 0016 decided, but leaves the bar; the
iterated LightGBM learner (`missforest_lgbm`) is withdrawn; a single-pass LightGBM (`lgbm`),
the `hgb` recipe with LightGBM, joins the bar. ADR 0007's protocol, best baseline and gap
are otherwise unchanged.

**Amended 2026-10-17 by [ADR 0021](0021-two-tiers-of-baseline-imputers.md).** missForest also
belongs to the strong tier, with the linear arms. The bar and `gap_to_missforest` are unchanged.

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

**Backfill applied 2026-10-07, 00:37 to 02:29 GMT-3, plus a second pass at 02:35.** 887 imputation
parents (ADR 0016's 873 and E38's 15, the `kc2_60nan` run of 2026-09-25 aside) carry `lgbm`
and `gap_to_missforest*`, marked `lgbm_backfilled=true` with their children and mirrors; the
bar moved on 660 of them (from
missForest back to `hgb`, KNN or the mean/mode fill, or to `lgbm`). The store was copied first
with SQLite's backup API to `/var/tmp/diogoneiss/backups/mlflow.db.before-lgbm-backfill`.

- **missForest corrected on 102 runs, all `kr-vs-kp` and `vehicle`,** by up to 0.0056 on a CV
  mean (decision 6): on tables of categories a flipped near-tie changes the next rounds'
  features, so the threaded predictions had drifted further than the 1.4e-4 first seen. No
  numerical table needed a correction.
- **Refused:** the superseded electricity run of 2026-09-25 (its mean/mode, as always), and
  `impute_kc2_60nan_20260925_031226`, whose masked bar moves back to `knn10` but whose
  `knn10` fold values, logged on the Windows PC by ADR 0007's backfill and absent from its
  artifact, are 2.3e-3 off when recomputed here. It is the one run whose
  `best_baseline/*` still names missForest; 30 later runs of the variant supersede it.
- **Second pass, `--fold_tolerance 1e-3`,** on `letter_20nan` and `credit-g_40nan` of
  2026-09-24/25, refused at 1e-9 for the same reason (1.4e-5 and 6.0e-4): their gap to the
  bar is taken from fold values recomputed here, within 6e-4 of the logged means.

**The bar, before and after, on today's default** (E35: seeds 42, 7 and 13, five folds; the
model's score is ADR 0015's K readout; each cell the mean over the three seeds; lower is
better):

| variant | induced: model | bar of 4 | bar of 5 | `lgbm` | missForest | masked: model | bar of 5 | missForest |
|---|---|---|---|---|---|---|---|---|
| `biodeg_20nan` | 0.587 | hgb 0.636 | lgbm 0.633 | 0.633 | 0.618 | 0.544 | hgb 0.627 | 0.570 |
| `biodeg_60nan` | 0.761 | hgb 0.866 | hgb 0.866 | 0.868 | 0.762 | 0.809 | hgb 0.873 | 0.786 |
| `credit-g_20nan` | 0.882 | hgb 0.918 | hgb 0.918 | 0.922 | 0.885 | 0.880 | lgbm 0.927 | 0.896 |
| `credit-g_40nan` | 0.956 | hgb 0.964 | hgb 0.964 | 0.968 | 0.931 | 0.982 | hgb 0.997 | 0.950 |
| `credit-g_60nan` | 0.992 | mean_mode 1.000 | mean_mode 1.000 | 1.079 | 1.021 | 0.980 | mean_mode 1.000 | 1.024 |
| `credit-g_80nan` | 1.008 | mean_mode 1.000 | mean_mode 1.000 | 1.112 | 1.093 | 1.008 | mean_mode 1.000 | 1.081 |
| `electricity_20nan` | 0.618 | hgb 0.606 | hgb 0.606 | 0.612 | 0.619 | 0.686 | hgb 0.690 | 0.710 |
| `kc2_20nan` | 0.541 | knn5 0.567 | knn5 0.567 | 0.741 | 0.605 | 0.530 | knn5 0.559 | 0.592 |
| `kc2_60nan` | 0.572 | knn5 0.717 | knn5 0.717 | 0.819 | 0.675 | 0.552 | knn10 0.717 | 0.626 |
| `kr-vs-kp_20nan` | 0.586 | hgb 0.612 | lgbm 0.608 | 0.608 | 0.658 | 0.649 | lgbm 0.674 | 0.716 |
| `kr-vs-kp_40nan` | 0.758 | hgb 0.773 | lgbm 0.764 | 0.764 | 0.801 | 0.781 | lgbm 0.777 | 0.824 |
| `kr-vs-kp_60nan` | 0.865 | hgb 0.903 | hgb 0.903 | 0.905 | 0.935 | 0.887 | lgbm 0.911 | 0.941 |
| `kr-vs-kp_80nan` | 0.976 | mean_mode 1.000 | mean_mode 1.000 | 1.074 | 1.021 | 1.000 | mean_mode 1.000 | 1.017 |
| `letter_20nan` | 0.459 | knn10 0.473 | knn10 0.473 | 0.499 | 0.462 | 0.496 | knn10 0.542 | 0.535 |
| `pendigits_20nan` | 0.332 | knn10 0.350 | knn10 0.350 | 0.363 | 0.334 | 0.369 | knn10 0.401 | 0.409 |
| `spambase_20nan` | 0.848 | hgb 0.866 | lgbm 0.865 | 0.865 | 0.859 | 0.865 | lgbm 0.886 | 0.923 |
| `spambase_40nan` | 0.903 | hgb 0.937 | lgbm 0.937 | 0.937 | 0.942 | 0.897 | lgbm 0.936 | 0.972 |
| `spambase_60nan` | 0.942 | hgb 0.984 | hgb 0.984 | 0.985 | 0.981 | 0.904 | lgbm 0.982 | 0.974 |
| `spambase_80nan` | 0.987 | mean_mode 1.000 | mean_mode 1.000 | 1.043 | 1.033 | 0.993 | mean_mode 1.000 | 1.058 |
| `vehicle_20nan` | 0.418 | hgb 0.473 | lgbm 0.473 | 0.473 | 0.452 | 0.420 | knn5 0.505 | 0.472 |
| `vehicle_60nan` | 0.645 | hgb 0.701 | hgb 0.701 | 0.702 | 0.647 | 0.616 | lgbm 0.668 | 0.616 |

- **The model is below the bar on 19 of 21 variants induced and 18 of 21 masked, as with
  ADR 0007's four baselines:** `lgbm` sets the bar on six variants induced and eight masked,
  by at most 0.009 below `hgb` (`kr-vs-kp_40nan` induced), and moves no verdict.
- **Against missForest, compared apart, the model is lower on 20 of 21 induced** (all but
  `credit-g_40nan`, 0.956 against 0.931) **and 18 of 21 masked** (not `biodeg_60nan`,
  `credit-g_40nan`, nor `vehicle_60nan`, a tie at 0.616). Six of the induced wins are
  within the 0.016 inductive-versus-transductive band of ADR 0016: `biodeg_60nan`,
  `electricity_20nan`, `pendigits_20nan`, `vehicle_60nan`, `credit-g_20nan` and
  `letter_20nan`, by 0.001 to 0.003. They are ties to a canonical missForest.
- Where missForest is lowest it is so by a wide margin on the small mixed tables
  (`biodeg_60nan` 0.762 against a bar of 0.866), which is the reason it stays outside the bar.
