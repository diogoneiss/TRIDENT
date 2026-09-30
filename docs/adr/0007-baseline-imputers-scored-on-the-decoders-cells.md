# Score baseline imputers on exactly the cells the decoder is scored on

## Status

Accepted (2026-09-24). Extends [ADR 0004](0004-imputation-decoder-task.md) decision 6,
which ranks folds by how much of a mean/mode imputer's error the decoder leaves, and
resolves finding F-06-2 of the imputation critique
(`docs/reviews/imputation-critique/CONSOLIDATED.md`, §3.5): the denominator of that
ratio was computed on every fold and persisted nowhere. Planned and implemented
test-first on 2026-09-24; the outcome section records what the first runs showed.

**Amended 2026-09-27** after the first night's results: the KNN baseline runs at two
neighbourhood sizes, `knn5` and `knn10`, and a comparison reads the better of the two
(decisions 2, 3, 5 and 6, and *How to compare*); and the induced truth of an
integer-coded categorical column is spelt the way the variant spells it, which fixes
every induced categorical score on electricity (*Outcome*).

**Amended again 2026-09-27**, on the user's go-ahead for the deferred follow-ups: a
third learned baseline, `hgb`, one gradient-boosting model per column (decisions 1, 2,
3, 5 and 6); a comparison reads the **baseline bar**, the lowest score any baseline
reached, in place of the KNN bar (*How to compare*); the per-column artifact carries
every baseline's errors (decision 7); and `scripts/backfill_baseline_imputers.py`
brings runs recorded before these amendments up to the current set (*Consequences*).

**Amended a third time 2026-09-28**, at the user's request: every cross-validated run
also logs its best baseline under `baseline/best` and names it in the tag
`best_baseline/<population>` (decision 8), which reverses the option rejected below;
and the backfill was applied (*Consequences*, *Outcome*).

**Amended a fourth time 2026-09-28**, at the user's request, to give a comparison or a
search one number to sort and minimise: every cross-validated run also logs the
model's gap to its best baseline, absolute and in percent of the bar (decision 9), and
the backfill wrote it onto every run that already carried a best baseline
(*Consequences*).

## Context

`impute_score` says how the decoder compares with filling the column mean or mode, and
nothing else. Two questions it cannot answer: how low that bar is on a given variant (a
ratio of 0.9 against a naive RMSE of 1.05 is a different achievement from 0.9 against
0.60), and whether a plain tabular imputer would have cleared it by more. The review had
already asked for the first number; the second is what a reader of the thesis will ask.

Facts established while planning, all against scikit-learn 1.9.0 as pinned by `uv.lock`:

- `KNNImputer` is deterministic and draws nothing from `numpy.random`; its `transform` on
  a test split costs 56 s on `electricity_20nan` (9,063 test rows), 8 s on `letter_20nan`,
  under 1 s on `spambase_20nan`, and `fit` only stores the training matrix.
- With the default `keep_empty_features=False`, a training column with no observed value
  is dropped from the output and every column after it shifts one place left. With
  `True` it is kept and filled with `0.0`, the mean in scaled space.
- A test row with every coordinate missing gets each column's training mean; a one-hot
  block then decodes to the training mode. No NaN ever comes back.
- Writing NaN into an `int64` column to hide a cell upcasts it to `float64`, and
  `as_category_strings` then renders `"3.0"` where the scored truth says `"3"`. Electricity's
  `day` is such a column: every categorical guess would have missed by its spelling.
- `summary._validate_final_metrics` aborts a cross-validation run on any non-finite value
  and on any fold whose key set differs from the others'.

## Decision

1. **A pure module, `src/training/imputation_baselines.py`, fully under the strict mypy
   gate.** A `BaselineImputer` protocol (`name`, `fit(frame)`, `impute(frame, hidden)`)
   takes a boolean mask of the cells to hide and never writes into the frame. Three
   implementations, `MeanModeImputer`, `KnnImputer` and `GradientBoostingImputer`, and
   `score_baselines`, which hides
   the scored cells, reads each baseline's fill back at the same `(row, column)`
   positions, and scores it through the same `score_cells` with the same naive
   denominators. The scored-cell table is copied, never mutated: it goes on to the cell
   ledger and the per-column artifact with the model's guesses in it.

2. **The recipes.** The mean/mode baseline is `mean_mode_baselines` itself, so its
   `impute_score` is 1.0 by construction (a unit test pins that exactly on a table of
   one kind, within an ulp on a mixed one). The KNN baselines are `KNNImputer(n_neighbors=k,
   weights="uniform", keep_empty_features=True)` at `k = 5` and at `k = 10`, each its own
   baseline, over numerical columns as the frame
   holds them (already scaled) and one one-hot block per categorical column over the
   categories the **training fold** shows; a hidden categorical cell blanks its whole
   block, a filled block decodes to its largest entry, ties to the first category in
   sorted order. The vocabulary is fixed at fit because it fixes the matrix width, and it
   is the training fold's alone: a category present only in the test fold can be produced
   by the model, whose embedder saw the whole variant, and never by KNN. That asymmetry is
   deliberate and makes the baseline slightly conservative; on `kr-vs-kp`'s `spcop` it is
   one cell. A training categorical column with no observed value is refused by name.

   The gradient-boosting baseline, `hgb`, trains one `HistGradientBoostingRegressor` or
   `HistGradientBoostingClassifier` (scikit-learn defaults, `random_state = 0`) per
   column, on the training rows where that column is observed, with every other column
   as a feature: numbers as scaled, categories as their index in the training fold's
   vocabulary, declared categorical to the booster. It fills a hidden cell in a single
   pass from the rest of its row, with every other hidden or missing cell entering as a
   gap the booster routes natively; nothing is iterated, so no fill feeds another. Each
   model also trains on a copy of its rows with a fifth of the feature cells hidden
   (the evaluation mask's default rate, from a seeded stream of its own): a booster
   that never saw a feature missing sends every gap in it down whichever branch held
   more rows, which on a complete training fold, as on a `_00nan` variant, means the
   rule it learned is not applied at all; a unit test on a step function pins the
   difference (-3 instead of 3 without the copy). A feature with no observed value is
   left out of that column's model, and a column with no observed value, one observed
   value or nothing to learn from is filled with a constant (zero, that value, or its
   mean or commonest category).

3. **Metric keys.** `impute/<population>/baseline/<name>/<metric>` with `<name>` in
   `mean_mode`, `knn5`, `knn10`, `hgb` and `<metric>` in `rmse_num_z`, `mae_num_z`,
   `acc_cat`, `macro_f1_cat`, `impute_score`. The cell counts are the model's own and
   are not repeated. The 21 runs recorded on 2026-09-24/25 logged the `k = 5` baseline
   under the name `knn`; the backfill script adds the current names beside it. Presence
   per kind mirrors the model's keys, so the fold-parity rule holds wherever it held
   before. Tracking prefixes `test/` per fold and the CV parent wraps
   `cv/test/.../<statistic>`, as for every other metric; nothing is registered anywhere.

4. **Populations.** Scored on the masked population, on every extra-rate population, and
   on the induced population where the `_00nan` sibling exists. Not scored on the
   `null_token` diagnostic (the same induced cells; the numbers would be identical) nor on
   the `validation/` family of the search objective (the same data for every trial, so a
   constant). An empty population yields no key, exactly as it yields no model score. An
   Optuna trial computes the baselines on its test populations like any other run.

5. **No configuration key and no flag.** The two KNN sizes (`KNN_NEIGHBOURS`), the
   booster's seed (`BOOSTING_SEED`) and its training-time hiding rate are constants in
   the module. A baseline is a fixed bar; a tunable one would be a second model, and a
   key would enter `complete_configuration` and every promoted file.

6. **Cost.** One `fit` per baseline per fold and one `transform` per KNN baseline per
   scored population. The booster's fit is the dearest, one model per column: 24 s per
   fold on `spambase` (57 columns), 12 s on `credit-g` (13 multi-class columns), 9 s on
   `kr-vs-kp`, 3 to 4 s elsewhere including `electricity`; its fills take under half a
   second. None of it runs on
   the GPU and none of it touching the global numpy or torch streams, so every seeded
   result and both regression fixtures are unchanged. The ranking metric, the search
   objective, promotion and the classification task are untouched.

7. **Per-column errors.** `metrics/per_column_imputation.csv` carries, for the masked and
   induced populations, every baseline's errors column by column as rows beneath the
   model's: the same `column`, `metric`, `value` layout with the metric named
   `baseline/<name>/<metric>` and the counts left out. Rows rather than columns, because
   the file is long-form and its columns are pinned by a unit test. This answers the half
   of F-06-2 the pooled metrics could not: an `acc_cat` of 0.93 on a column now sits beside
   the mode's accuracy on the same cells. The extra-rate populations are not in the file,
   as the model's own rows for them never were.

8. **The best baseline, logged.** The cross-validation summary picks, per scored
   population, the baseline whose mean `impute_score` over the folds is lowest (a tie
   to the name that sorts first), copies every statistic of every metric of *that*
   baseline under `impute/<population>/baseline/best/<metric>` (so the parent carries
   `cv/test/impute/induced/baseline/best/impute_score/mean`), and tags the parent
   `best_baseline/<population>` with its name. It copies one imputer's numbers, never
   the best of each metric taken separately, and it is chosen once per run, on the
   cross-validated mean, so every statistic describes the same baseline on every fold.
   Diagnostic children carry no `best`: their fold values are the named baseline's.
   Classification summaries carry nothing of it.

9. **The gap to the best baseline, logged.** The same summary logs, per population,
   `impute/<population>/gap_to_best_baseline/impute_score`: on each fold, the model's
   `impute_score` minus the best baseline's on that fold, summarised into the usual
   seven statistics. Negative means the model beats the bar, so lower stays better,
   as for every imputation score. Paired fold by fold, its interval is the gap's own,
   narrower than two marginal intervals set side by side; its mean is exactly the
   difference of the two logged means. Beside it,
   `impute/<population>/gap_to_best_baseline_pct/impute_score` divides every fold's gap
   by the bar's cross-validated mean, so its mean is the gap of the means over the
   bar's mean, the number the two logged means give by hand, and its interval is the
   absolute one rescaled. **Each fold divided by its own bar was rejected:** the mean
   of such ratios is not a ratio of means, and on `kc2_20nan`, whose bar ranges from
   0.37 to 0.80 across folds, the folds' percentages average +2.6% where the means
   are 2.1% apart in the model's favour, a sign no search should be steered by. A bar
   whose mean is 0 leaves no percentage, and none is logged. Only `impute_score` gets a
   gap, as asked; the key leaves room for another metric. The keys stay out of the
   `baseline/` namespace, where `gap_to_best_baseline` would read as a baseline's name.
   Like `best`, it lives on the cross-validated parent (and in `cv_summary.json`) only:
   single-split runs and Optuna trials choose no best baseline, so they carry no gap,
   and the search objective is untouched.

## How to compare

On a `_20nan`..`_80nan` variant, read `cv/test/impute/induced/impute_score/mean` beside
the **baseline bar**, logged as `cv/test/impute/induced/baseline/best/impute_score/mean`
with the tag `best_baseline/impute/induced` naming it: the lowest mean over
`mean_mode`, `knn5`, `knn10` and `hgb`, chosen per run and per population, same folds,
same cells. All are lower-is-better with 1.0 at mean/mode parity, so the bar is
never above 1.0; the model does something no baseline does only where its number is
below it. `cv/test/impute/induced/gap_to_best_baseline/impute_score/mean` is that
difference in one column (negative where the model is below the bar), its `ci95_upper`
below 0 is the paired reading of a win, and `..._pct/...` puts variants of different
difficulty on one scale. Choosing the best baseline after seeing its test score favours the
baselines, so the bar errs against the model, which is the safe direction for a claim
that the model beats them. The KNN bar of the first amendment is the same choice
restricted to `knn5` and `knn10`.
`cv/test/impute/induced/baseline/mean_mode/rmse_num_z/mean` and
`.../baseline/mean_mode/acc_cat/mean` are the bar itself in absolute units. On any variant
the `impute/masked/...` family reads the same way. `baseline/mean_mode/impute_score` must be
1.0 on every run; anything else means the fill and the score's denominator disagree.

## Considered options

- **`IterativeImputer` (MICE-style) as the stronger baseline.** Rejected: linear on a
  one-hot matrix, stochastic (a `random_state` to discipline on every run), behind an
  experimental import, and not obviously stronger than KNN on mixed tables.
- **A missForest-style imputer, one `HistGradientBoosting` model per column.** Deferred
  at first, when the user chose KNN as the first bar; adopted in the second amendment as
  `hgb`. **Iterating it as missForest does** (fill, refit on the fills, repeat) was
  rejected: it multiplies the dearest fit by the number of rounds and turns the baseline
  transductive-leaning; a single pass with native gaps, trained on a copy with hidden
  cells, recovers the rule a hidden neighbour would have given.
- **Logging only the naive errors.** Rejected: answers the review's finding but not the
  question a reader asks, which is whether the bar is low.
- **Baseline columns in `per_column_imputation.csv`.** Rejected in favour of rows
  (decision 7): the file is long-form and its columns are pinned by a unit test.
- **A configuration key for `k`.** Rejected, decision 5.
- **A single KNN at `k = 10`**, the stronger size on most variants. Rejected in the
  amendment: `k = 5` is the better bar on the small, low-missingness tables (`biodeg`,
  `kc2` at 20-40%), and logging both costs one more transform.
- **A derived `baseline/knn_best` metric.** Rejected at first: which size is better is
  a per-run question, and the fold-level logger could only answer it fold by fold,
  mixing neighbourhoods across the folds of one run. **Reversed 2026-09-28** at the
  user's request, generalised to every baseline and computed by the cross-validation
  summary, where the per-run choice is available, which removes the objection
  (decision 8).

10. **The baselines' scores are cached across runs that share a fold (added 2026-09-30,
    the user's decision).** They never depend on the model, so every arm of a study and every
    trial of a search computed the same numbers: 3.6 to 9.1 s per fold, up to 38% of a
    credit-g_20nan cell. `src/training/baseline_cache.py` keys a file under
    `results/baseline_cache/` (a fixed path from the checkout root, never a run's own output
    directory) on a hash of everything that enters the computation: the machine, the
    scikit-learn, pandas and numpy versions, the source of `imputation_baselines.py` and
    `imputation_metrics.py`, the training and test frames, the scored cells' positions,
    kinds and truths (never the model's guesses), the naive baselines and the column lists.
    The imputers are fit only when some population misses. `--no_baseline_cache` turns it
    off; an imputation parent carries `baseline_cache=on|off`, provenance only and not
    backfilled, since a hit is the computation. Checked: the pre-training ablation's check
    cell (`credit-g_20nan` s42 B, GPU) gives all 57 fold columns identical with the cache
    empty and full (96.5 s, then 49.0 s).

## Consequences

- Twenty more keys per scored population per fold (`test/...`), 140 per population on a
  CV parent. On `credit-g_20nan` without extra rates: 40 fold keys, 280 parent keys.
- `stubs/sklearn/impute` and `stubs/sklearn/ensemble` bring the hand-written
  scikit-learn stubs to sixteen symbols.
- The baselines depend only on the dataset, the seed, the fold count and
  `EVAL_MASK_RATE`, never on the model, so they can be recomputed for every existing
  imputation run without training. `scripts/backfill_baseline_imputers.py` does that for
  every finished imputation parent that logged baselines but lacks some of the current
  ones, and writes the missing ones where a live run would have (the seven statistics
  on the parent at step 0, each diagnostic child's fold value), tags the tree
  `baselines_backfilled=true` and re-mirrors it. It writes a run only if its
  recomputation reproduces, within 1e-9, every baseline mean the run already logged (a
  legacy `knn` checked as `knn5`), so the electricity run scored before the spelling fix
  is refused on its own numbers. Dry run by default. **Applied 2026-09-28**: 21 run
  trees written and re-mirrored, each reproducing every baseline it had logged to
  machine precision (2.2e-16 at worst), the 2026-09-25 electricity run refused. One
  correction on the way: a first pass capped scikit-learn's working memory, which on
  `letter` (integer features, many tied distances) changed which neighbour won 13
  ties in a fold and moved a logged score by 3e-5, failing the check; the script now
  leaves it at the default a live run uses and checks to 1e-9. A second pass derived
  `baseline/best` for all 22 runs from their logged statistics, without recomputing.
  The store was copied before each pass. A third pass (decision 9) gives every run
  with a best baseline its gap to it, from each fold's own numbers: the model's from the
  run's `metrics/raw_fold_metrics.csv`, a baseline the run did not log live (`knn10`
  and `hgb` on the 2026-09-24/25 runs) from the recomputation cache. It writes only
  where those fold values average to the run's own logged model and `baseline/best`
  means within 1e-9, tags the run `best_baseline_gap_backfilled=true` (not
  `baselines_backfilled`: two of the runs logged their baselines live) and re-mirrors
  it. **Applied 2026-09-28** to all 24 runs with a best baseline, none refused; each
  planned percentage matched the hand calculation from the logged means to 2e-14, and
  none disagreed in sign with its absolute gap. The superseded electricity run of
  2026-09-25 got a gap, as it got a best, from its own defective induced numbers.
- Fourteen more keys per scored population on a CV parent, seven for each gap.
- `CONTEXT.md` fixes the terms *baseline imputer* and *baseline bar*.
- Seven statistics per metric of the best baseline on each population of a parent run
  (35 on a mixed table), and one tag per population.

## Outcome (2026-09-24/25 and 27: 22 five-fold runs with the change, seed 42, cosine)

**The model's numbers did not move.** Every new run that has an earlier run of the same
seed, fold count, schedule and configuration file reproduces it **to six decimals** on
every model metric present: `credit-g_20nan` equals its runs of 2026-09-11 21:37 and
2026-09-17 00:36 / 23:48; `credit-g_40nan` equals 2026-09-11 21:48 and 2026-09-17 00:25;
`kr-vs-kp_20nan`, `kr-vs-kp_40nan`, `spambase_20nan` and `spambase_40nan` equal their
2026-09-11/12 runs (max difference 0.0). The two credit-g runs of 2026-09-18 morning
differ from all their siblings (e.g. `masked/impute_score` 0.916 vs 0.926) and did so
before this change; they are the outliers, not the new runs. Both regression fixtures
passed unedited, and `baseline/mean_mode/impute_score` logged exactly 1.0 on every fold
of every run.

**The bar and the model, after the backfill.** `cv/test/impute/<population>/.../impute_score/mean`,
lower is better, 1.0 is mean/mode parity, 95% intervals the summary's own. The **best
baseline** column is `baseline/best` as every run now logs it (decision 8): the baseline
with the lowest mean, its interval, bold where it is disjoint from the model's and below.
Every number is logged in the store: `knn10` and `hgb` of the 2026-09-24/25 runs by the
backfill of 2026-09-28, the `electricity_20nan` row is the rerun of 2026-09-27.

| variant | config | induced: model | knn5 | knn10 | hgb | induced: best baseline | masked: model | masked: best baseline |
|---|---|---|---|---|---|---|---|---|
| `credit-g_20nan` | promoted | 0.913 [0.868, 0.958] | 1.014 | 0.955 | 0.936 | hgb 0.936 [0.912, 0.959] | 0.926 | knn10 0.923 |
| `credit-g_40nan` | promoted | 0.968 [0.940, 0.995] | 1.043 | 0.987 | 0.958 | hgb 0.958 [0.907, 1.008] | 0.965 | mean_mode 1.000 |
| `credit-g_60nan` | defaults | 1.012 [0.983, 1.041] | 1.121 | 1.053 | 1.048 | mean_mode 1.000 [1.000, 1.000] | 1.012 | mean_mode 1.000 |
| `credit-g_80nan` | defaults | 1.024 [1.008, 1.040] | 1.100 | 1.048 | 1.113 | **mean_mode 1.000 [1.000, 1.000]** | 0.995 | mean_mode 1.000 |
| `kr-vs-kp_20nan` | promoted | 0.603 [0.559, 0.646] | 0.721 | 0.695 | 0.612 | hgb 0.612 [0.565, 0.659] | 0.660 | hgb 0.677 |
| `kr-vs-kp_40nan` | promoted | 0.778 [0.734, 0.822] | 0.927 | 0.883 | 0.772 | hgb 0.772 [0.746, 0.797] | 0.802 | hgb 0.787 |
| `kr-vs-kp_60nan` | defaults | 0.904 [0.871, 0.936] | 1.064 | 1.004 | 0.903 | hgb 0.903 [0.888, 0.918] | 0.869 | hgb 0.863 |
| `kr-vs-kp_80nan` | defaults | 1.008 [0.945, 1.071] | 1.155 | 1.075 | 1.073 | mean_mode 1.000 [1.000, 1.000] | 1.048 | mean_mode 1.000 |
| `spambase_20nan` | promoted | 0.886 [0.859, 0.913] | 0.995 | 0.939 | 0.865 | hgb 0.865 [0.834, 0.897] | 0.879 | hgb 0.874 |
| `spambase_40nan` | promoted | 0.930 [0.909, 0.952] | 1.020 | 0.974 | 0.936 | hgb 0.936 [0.903, 0.970] | 0.930 | hgb 0.957 |
| `spambase_60nan` | defaults | 0.962 [0.937, 0.988] | 1.083 | 1.032 | 0.987 | hgb 0.987 [0.976, 0.998] | 0.912 | hgb 1.000 |
| `spambase_80nan` | defaults | 1.015 [0.988, 1.042] | 1.095 | 1.046 | 1.041 | mean_mode 1.000 [1.000, 1.000] | 0.987 | mean_mode 1.000 |
| `vehicle_20nan` | defaults | 0.484 [0.421, 0.548] | 0.508 | 0.499 | 0.471 | hgb 0.471 [0.425, 0.518] | 0.486 | knn10 0.500 |
| `vehicle_60nan` | defaults | 0.677 [0.585, 0.769] | 0.779 | 0.731 | 0.696 | hgb 0.696 [0.627, 0.765] | 0.680 | hgb 0.692 |
| `biodeg_20nan` | defaults | 0.674 [0.509, 0.839] | 0.725 | 0.744 | 0.644 | hgb 0.644 [0.452, 0.837] | 0.596 | hgb 0.607 |
| `biodeg_60nan` | defaults | 0.899 [0.842, 0.955] | 0.979 | 0.929 | 0.864 | hgb 0.864 [0.842, 0.887] | 0.829 | hgb 0.847 |
| `kc2_20nan` | defaults | 0.586 [0.476, 0.696] | 0.608 | 0.633 | 0.746 | knn5 0.608 [0.405, 0.810] | 0.541 | knn5 0.553 |
| `kc2_60nan` | defaults | 0.754 [0.605, 0.902] | 0.748 | 0.742 | 0.840 | knn10 0.742 [0.533, 0.951] | 0.596 | knn10 0.715 |
| `pendigits_20nan` | defaults | 0.399 [0.384, 0.414] | 0.358 | 0.348 | 0.365 | **knn10 0.348 [0.339, 0.357]** | 0.435 | knn10 0.407 |
| `letter_20nan` | defaults | 0.515 [0.507, 0.522] | 0.484 | 0.473 | 0.500 | **knn10 0.473 [0.468, 0.478]** | 0.551 | knn10 0.544 |
| `electricity_20nan` | defaults | 0.669 [0.620, 0.718] | 0.910 | 0.862 | 0.593 | hgb 0.593 [0.547, 0.640] | 0.710 | hgb 0.666 |

The stronger bar changes the verdict. On the induced population the model is below the
best baseline on 6 of 21 variants (`credit-g_20nan`, `kc2_20nan`, `kr-vs-kp_20nan`, `spambase_40nan`, `spambase_60nan`, `vehicle_60nan`),
never with disjoint intervals; a baseline is below the model with disjoint intervals on
3 (`credit-g_80nan` by the mean/mode fill, `pendigits_20nan` and `letter_20nan` by
`knn10`). `hgb` sets the bar on 13 of 21; at 80% missingness, and on `credit-g_60nan`,
nothing beats filling the column mean or mode; KNN holds it on `kc2`, `pendigits` and
`letter` (`knn5` on `kc2_20nan`, `knn10` elsewhere). Before `hgb` existed, the KNN bar alone put the model below it on 18 of 21.

**A defect the baselines exposed on `electricity_20nan`, fixed 2026-09-27.** The
variant's only categorical column, `day`, holds NaN, so pandas reads it as `float64` and
its categories stringify as `"1.0"`…`"7.0"`; the `_00nan` sibling has no NaN there, reads
as `int64`, and its truth stringified as `"1"`…`"7"`. `_within_vocabulary` therefore
treated every induced `day` cell as a category "the variant never shows" (the run log
warned of 9,063 such cells per fold) and scored it a miss: induced `acc_cat` and
`macro_f1_cat` were **0.0 for the model and both baselines** on 1,812 categorical cells
per fold, pinning the categorical term of the induced `impute_score` at parity. It
predates this ADR: it is the dtype hazard of the Context section, arising between the
variant and its sibling rather than inside one frame. A survey of every variant found it
on electricity alone, at all four missingness levels.

The fix spells the sibling's numeric-coded categorical columns in the variant's dtype
before the induced truth is encoded (`decoding._in_variant_dtypes`; numeric-to-numeric
differences only, so a column of words is untouched), with a unit test on an
integer-coded toy column. The rerun `impute_electricity_20nan_20260927_195444` shows the
fix and nothing else: not one "never shows" warning; induced `acc_cat` 0.000 → **0.280**
for the model, 0.145 for the mode, 0.173 / 0.191 for `knn5` / `knn10`; induced
`impute_score` 0.689 → 0.669 for the model and 0.914 → 0.910 for `knn5`; the induced
numerical metrics and every masked metric identical to the defective run to four
decimals, and `knn5` identical to its old `knn`. The defective run
`impute_electricity_20nan_20260925_044917` (and its mirror) stays in the store; it is
superseded, and its induced categorical numbers must not be read. No other finished run
was affected: every earlier electricity imputation run had `FAILED`.

**How much the KNN bar depends on `k`** (no training: the runs' own folds and masks,
seed 42, 27 variants, `impute_score` averaged over 5 folds, induced population):
`k = 10` scores below `k = 5` on 20 of the 27 variants, by 0.01 to 0.08, and `k = 20`
lower still wherever missingness is 40% or more (`credit-g_20nan` 1.014 → 0.955 → 0.938;
`spambase_20nan` 0.995 → 0.939 → 0.925; `kr-vs-kp_40nan` 0.927 → 0.883 → 0.850;
`pendigits_20nan` 0.358 → 0.348). Distance weighting is worse than uniform at `k = 5` on
21 of 27. `k = 5` is the *weaker* of the reasonable bars at high missingness. **Decided
2026-09-27:** both sizes are logged as their own baselines and a comparison reads the
better one (see the amendment in *Status*).

**Cost in situ.** Run times with the baselines: credit-g 5 min, kr-vs-kp 18 min, spambase
38 min, vehicle and kc2 4 min, biodeg 8 min, pendigits 33 min, letter 60 min and
electricity 75 min for five folds, all `FINISHED`. The earlier runs of the same
configurations took the same time to the minute (credit-g 4.7–5.2, kr-vs-kp 17.8–18.4,
spambase 37.9–41.9), so on those tables the baselines cost nothing visible; on letter and
electricity the KNN transform is the 10–60 s per population per fold measured before.
The electricity rerun, with two KNN transforms per population, took 134 min against
75; its folds took 14 to 39 min on a machine with 0.3 GB free, while the second KNN
transform adds about 2 min per fold there, so most of the difference is memory pressure
rather than the baselines.
