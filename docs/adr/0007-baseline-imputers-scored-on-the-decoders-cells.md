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
   takes a boolean mask of the cells to hide and never writes into the frame. Two
   implementations, `MeanModeImputer` and `KnnImputer`, and `score_baselines`, which hides
   the scored cells, reads each baseline's fill back at the same `(row, column)`
   positions, and scores it through the same `score_cells` with the same naive
   denominators. The scored-cell table is copied, never mutated: it goes on to the cell
   ledger and the per-column artifact with the model's guesses in it.

2. **The two recipes.** The mean/mode baseline is `mean_mode_baselines` itself, so its
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

3. **Metric keys.** `impute/<population>/baseline/<name>/<metric>` with `<name>` in
   `mean_mode`, `knn5`, `knn10` and `<metric>` in `rmse_num_z`, `mae_num_z`, `acc_cat`,
   `macro_f1_cat`, `impute_score`. The cell counts are the model's own and are not
   repeated. The 21 runs recorded on 2026-09-24/25, before the amendment, carry the
   `k = 5` baseline under the name `knn` and have no `knn10`. Presence per kind mirrors the model's keys, so the fold-parity rule holds
   wherever it held before. Tracking prefixes `test/` per fold and the CV parent wraps
   `cv/test/.../<statistic>`, as for every other metric; nothing is registered anywhere.

4. **Populations.** Scored on the masked population, on every extra-rate population, and
   on the induced population where the `_00nan` sibling exists. Not scored on the
   `null_token` diagnostic (the same induced cells; the numbers would be identical) nor on
   the `validation/` family of the search objective (the same data for every trial, so a
   constant). An empty population yields no key, exactly as it yields no model score. An
   Optuna trial computes the baselines on its test populations like any other run.

5. **No configuration key and no flag.** The two sizes, 5 and 10, with uniform weights
   are a constant in the module (`KNN_NEIGHBOURS`). A baseline is a fixed bar; a tunable one would be a second model, and a key
   would enter `complete_configuration` and every promoted file.

6. **Cost.** One `fit` per baseline per fold and one `transform` per KNN baseline per
   scored population (two since the amendment), none of it on
   the GPU and none of it touching the global numpy or torch streams, so every seeded
   result and both regression fixtures are unchanged. The ranking metric, the search
   objective, promotion and the classification task are untouched.

## How to compare

On a `_20nan`..`_80nan` variant, read `cv/test/impute/induced/impute_score/mean` beside
the **KNN bar**: the lower of `cv/test/impute/induced/baseline/knn5/impute_score/mean` and
`.../baseline/knn10/impute_score/mean`, chosen per run and per population, same folds,
same cells. All are lower-is-better with 1.0 at mean/mode parity; the model is doing
something a plain tabular imputer does not only where its number is below that bar.
Choosing the better neighbourhood after seeing its test score favours the baseline, so
the bar errs against the model, which is the safe direction for a claim that the model
beats it. For the runs of 2026-09-24/25, `knn` is `knn5` and `knn10` was computed offline
on the same folds and cells (the offline `knn5` reproduces the logged `knn` to 1e-8).
`cv/test/impute/induced/baseline/mean_mode/rmse_num_z/mean` and
`.../baseline/mean_mode/acc_cat/mean` are the bar itself in absolute units. On any variant
the `impute/masked/...` family reads the same way. `baseline/mean_mode/impute_score` must be
1.0 on every run; anything else means the fill and the score's denominator disagree.

## Considered options

- **`IterativeImputer` (MICE-style) as the stronger baseline.** Rejected: linear on a
  one-hot matrix, stochastic (a `random_state` to discipline on every run), behind an
  experimental import, and not obviously stronger than KNN on mixed tables.
- **A missForest-style imputer, one `HistGradientBoosting` model per column.** Deferred:
  it is the strongest cheap tabular imputer, handles gaps and categories natively, and
  fits the same protocol as a third implementation without touching the decode stage. It
  costs one model per column per fold, needs `sklearn.ensemble` stubs and an explicit
  integer `random_state`, and the user chose KNN as the first bar. See the follow-up.
- **Logging only the naive errors.** Rejected: answers the review's finding but not the
  question a reader asks, which is whether the bar is low.
- **Baseline rows in `per_column_imputation.csv`.** Deferred: the artifact's columns are
  pinned by a unit test and the review's request for it is separate from this one.
- **A configuration key for `k`.** Rejected, decision 5.
- **A single KNN at `k = 10`**, the stronger size on most variants. Rejected in the
  amendment: `k = 5` is the better bar on the small, low-missingness tables (`biodeg`,
  `kc2` at 20-40%), and logging both costs one more transform.
- **A derived `baseline/knn_best` metric.** Rejected: which size is better is a per-run
  question, and the fold-level logger could only answer it fold by fold, mixing
  neighbourhoods across the folds of one run. The choice is made when reading.

## Consequences

- Fifteen more keys per scored population per fold (`test/...`), 105 per population on a
  CV parent. On `credit-g_20nan` without extra rates: 30 fold keys, 210 parent keys.
- `stubs/sklearn/impute/__init__.pyi` is the fourteenth hand-written scikit-learn stub.
- The baselines depend only on the dataset, the seed, the fold count and
  `EVAL_MASK_RATE`, never on the model, so they can be recomputed for every existing
  imputation run without training. A backfill of the seed-42 runs in the store is a
  follow-up, not part of this change.
- A `CONTEXT.md` entry fixes the term *baseline imputer*.

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

**The bar and the model.** `cv/test/impute/<population>/.../impute_score/mean`, lower is
better, 1.0 is mean/mode parity; 95% intervals are the summary's own (Student-t over the
five folds). The **KNN bar** is the lower of `knn5` and `knn10`, chosen per run and per
population, and is bold where it beats the model. For the runs of 2026-09-24/25, `knn5`
is the logged `knn` and `knn10` was computed offline on the same folds and cells; the
`electricity_20nan` row is the rerun of 2026-09-27 with both sizes logged and the induced
truth spelt correctly (see below). "promoted" is the variant's `*.imputation.json`,
otherwise the defaults. The mean/mode column is the naive bar in absolute units on the
induced cells: pooled RMSE in scaled space, then categorical accuracy.

| variant | config | induced: model | knn5 | knn10 | induced: KNN bar | mean/mode RMSE, acc | masked: model | masked: KNN bar |
|---|---|---|---|---|---|---|---|---|
| `credit-g_20nan` | promoted | 0.913 [0.868, 0.958] | 1.014 | 0.955 | k=10: 0.955 [0.911, 0.999] | 1.025, 0.590 | 0.926 | **k=10: 0.923** |
| `credit-g_40nan` | promoted | 0.968 [0.940, 0.995] | 1.043 | 0.987 | k=10: 0.987 [0.958, 1.016] | 1.041, 0.579 | 0.965 | k=10: 1.031 |
| `credit-g_60nan` | defaults | 1.012 [0.983, 1.041] | 1.121 | 1.053 | k=10: 1.053 [1.035, 1.072] | 0.986, 0.591 | 1.012 | k=10: 1.047 |
| `credit-g_80nan` | defaults | 1.024 [1.008, 1.040] | 1.100 | 1.048 | k=10: 1.048 [1.024, 1.071] | 0.961, 0.586 | 0.995 | k=10: 1.062 |
| `kr-vs-kp_20nan` | promoted | 0.603 [0.559, 0.646] | 0.721 | 0.695 | k=10: 0.695 [0.665, 0.724] | –, 0.811 | 0.660 | k=10: 0.738 |
| `kr-vs-kp_40nan` | promoted | 0.778 [0.734, 0.822] | 0.927 | 0.883 | k=10: 0.883 [0.860, 0.906] | –, 0.813 | 0.802 | k=10: 0.903 |
| `kr-vs-kp_60nan` | defaults | 0.904 [0.871, 0.936] | 1.064 | 1.004 | k=10: 1.004 [0.985, 1.023] | –, 0.812 | 0.869 | k=10: 1.009 |
| `kr-vs-kp_80nan` | defaults | 1.008 [0.945, 1.071] | 1.155 | 1.075 | k=10: 1.075 [1.043, 1.107] | –, 0.813 | 1.048 | k=10: 1.084 |
| `spambase_20nan` | promoted | 0.886 [0.859, 0.913] | 0.995 | 0.939 | k=10: 0.939 [0.934, 0.944] | 0.986, – | 0.879 | k=10: 0.939 |
| `spambase_40nan` | promoted | 0.930 [0.909, 0.952] | 1.020 | 0.974 | k=10: 0.974 [0.938, 1.010] | 1.027, – | 0.930 | k=10: 0.992 |
| `spambase_60nan` | defaults | 0.962 [0.937, 0.988] | 1.083 | 1.032 | k=10: 1.032 [1.022, 1.042] | 1.175, – | 0.912 | k=10: 1.041 |
| `spambase_80nan` | defaults | 1.015 [0.988, 1.042] | 1.095 | 1.046 | k=10: 1.046 [1.035, 1.057] | 1.158, – | 0.987 | k=10: 1.075 |
| `vehicle_20nan` | defaults | 0.484 [0.421, 0.548] | 0.508 | 0.499 | k=10: 0.499 [0.440, 0.559] | 1.002, – | 0.486 | k=10: 0.500 |
| `vehicle_60nan` | defaults | 0.677 [0.585, 0.769] | 0.779 | 0.731 | k=10: 0.731 [0.651, 0.810] | 1.051, – | 0.680 | k=10: 0.743 |
| `biodeg_20nan` | defaults | 0.674 [0.509, 0.839] | 0.725 | 0.744 | k=5: 0.725 [0.573, 0.876] | 1.318, – | 0.596 | k=5: 0.675 |
| `biodeg_60nan` | defaults | 0.899 [0.842, 0.955] | 0.979 | 0.929 | k=10: 0.929 [0.923, 0.936] | 1.010, – | 0.829 | k=10: 0.940 |
| `kc2_20nan` | defaults | 0.586 [0.476, 0.696] | 0.608 | 0.633 | k=5: 0.608 [0.405, 0.810] | 1.045, – | 0.541 | k=5: 0.553 |
| `kc2_60nan` | defaults | 0.754 [0.605, 0.902] | 0.748 | 0.742 | **k=10: 0.742 [0.533, 0.951]** | 1.149, – | 0.596 | k=10: 0.715 |
| `pendigits_20nan` | defaults | 0.399 [0.384, 0.414] | 0.358 | 0.348 | **k=10: 0.348 [0.339, 0.357]** | 1.002, – | 0.435 | **k=10: 0.407** |
| `letter_20nan` | defaults | 0.515 [0.507, 0.522] | 0.484 | 0.473 | **k=10: 0.473 [0.468, 0.478]** | 0.993, – | 0.551 | **k=10: 0.544** |
| `electricity_20nan` | defaults | 0.669 [0.620, 0.718] | 0.910 | 0.862 | k=10: 0.862 [0.844, 0.881] | 0.956, 0.145 | 0.710 | k=10: 0.904 |

On the induced population the model is below the KNN bar on 18 of 21 variants, with
disjoint intervals on 6 of them (`kr-vs-kp` at 20/40/60%, `spambase` at 20/60%,
`electricity_20nan`) and overlapping ones on the other 12; KNN is lower on the remaining
3, with disjoint intervals on 2. Two readings survive the stronger bar. On every mixed or categorical table
(credit-g, kr-vs-kp, spambase, electricity) the decoder stays below KNN at every
missingness level, though on credit-g the margin shrinks to within the intervals once
`k = 10` is allowed (`credit-g_20nan` 0.913 against 0.955). On the two large, clean
numerical tables KNN wins with disjoint intervals, and by more at `k = 10`:
`pendigits_20nan` 0.348 against the model's 0.399 and `letter_20nan` 0.473 against 0.515
on the induced cells. KNN is also lower on the masked cells of both (0.407 vs 0.435,
disjoint; 0.544 vs 0.551, within the intervals), where at `k = 5` the model had still won
letter's masked cells. A plain neighbour lookup is the stronger imputer where rows are near-duplicates of
each other, which is what the bar is for. `kc2_60nan` is the one small table where the bar
sits below the model (0.742 vs 0.754), well inside both intervals.

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
