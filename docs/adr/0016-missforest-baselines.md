# Add missForest, with random forests and with LightGBM, to the baseline imputers

## Status

Accepted (2026-10-06). The user asked for a state-of-the-art baseline beside ADR
[0007](0007-baseline-imputers-scored-on-the-decoders-cells.md)'s mean/mode, KNN and single-pass
gradient boosting, computed and cached like them and backfilled onto every run; chose
missForest from four candidates (*Considered options*), asked for a LightGBM variant beside
the forest, and deferred ReMasker ([BACKLOG I3](../BACKLOG.md)). Extends ADR 0007: its
protocol, metric layout, best baseline, gap and cache all apply; this ADR records only
what differs.

## Context

missForest (Stekhoven and Bühlmann, *Bioinformatics* 28(1), 2012) is the imputer the
benchmark literature keeps finding hard to beat on mixed tables (Jäger, Allhorn and
Bießmann, *Frontiers in Big Data*, 2021; Jarrett et al., HyperImpute, ICML 2022, where it is
the strongest baseline the AutoML method is measured against). It starts every gap at its
column's mean or mode, then, round after round, refits one random forest per column on the
other columns' current fills and rewrites that column's gaps, until the fills settle.

Two things separate it from what ADR 0007 already logs:

- `hgb` is a single pass: each column's model sees every other gap as a gap, so no fill
  ever informs another. missForest iterates, so a row with two gaps gets each from the
  other's best guess.
- The R package is transductive: it imputes one incomplete table, fitting on the very rows
  it fills. Every ADR 0007 baseline is inductive, fit on the training fold and asked to
  fill the test fold, which is also how the model is scored.

Measured while planning, on gorgona8 (fold 1 of seed 42, `_20nan`, fit plus one
population's fill, eight threads):

| table | training rows × columns | `missforest` | `missforest_lgbm` |
|---|---|---|---|
| kc2 | 364 × 21 | 15 s + 2.6 s (6 rounds) | 1.8 s + 0.0 s (3) |
| vehicle | 591 × 18 | 9.5 s + 1.6 s (4) | 2.2 s + 0.0 s (3) |
| credit-g | 700 × 20 | 21 s + 2.3 s (6) | 10 s + 0.1 s (4) |
| biodeg | 738 × 41 | 28 s + 5.1 s (5) | 15 s + 0.2 s (5) |
| kr-vs-kp | 2,236 × 36 | 26 s + 2.6 s (4) | 11 s + 0.2 s (5) |
| spambase | 3,220 × 57 | 41 s + 7.4 s (4) | 53 s + 0.2 s (5) |
| pendigits | 7,693 × 16 | 24 s + 2.3 s (8) | 13 s + 0.1 s (7) |
| letter | 14,000 × 16 | 32 s + 3.7 s (8) | 22 s + 0.2 s (10) |
| electricity | 31,717 × 8 | 22 s + 1.9 s (7) | 9.2 s + 0.3 s (5) |
| credit-g, 80nan | 700 × 20 | 24 s + 4.2 s (9) | 5.2 s + 0.1 s (10) |
| spambase, 80nan | 3,220 × 57 | 57 s + 12 s (10) | 59 s + 0.4 s (9) |

Both learners run every round to convergence well inside the ten-round cap at 20%
missingness and reach it at 80%. Nothing here is slower than the KNN transforms ADR 0007
already pays on electricity and letter.

## Decision

1. **A module of its own, `src/training/missforest.py`,** under the strict mypy gate:
   `MissForestImputer(numerical, categorical, learner)` implements ADR 0007's
   `BaselineImputer` protocol, and `fit_missforest_imputers` fits both learners on a fold.
   `imputation_baselines.py` and `imputation_metrics.py` are not touched, byte for byte
   (decision 6 says why).

2. **The recipe, inductive.** `fit` runs the paper's algorithm on the training fold: gaps
   start at the training mean or mode (categories as their index in the training fold's
   vocabulary, as `hgb` encodes them); each round visits the columns in order of
   increasing missingness, stable on ties, fits the column's model on its observed rows
   with every other column's current fills as features, and writes the predictions into
   its gaps before the next column (Gauss–Seidel, as the R code does). It keeps the models
   of the round whose fills it keeps. `impute` starts the hidden and missing test cells at
   the training means and modes and runs the same rounds with those models on the test rows
   alone, under the same stopping rule. Two departures from the R package, both forced by
   the protocol:

   - **Every column gets a model, gaps or not.** The R package fits only columns with
     gaps, because it fills only the table it was given; here any test column can be
     hidden by the evaluation mask, and on a complete training fold (any `_00nan`
     variant) no column would get one. A unit test pins the step rule learnt from a
     complete fold.
   - **The fill replays fitted models instead of refitting on the test rows.** Refitting
     would make it transductive, see *Bias* below.

3. **The stopping rule is the R package's `stopCriterion`:** after each round, Δ_N (the
   numerical fills' squared change over their squared size) and Δ_F (the share of
   categorical fills that changed) are compared with the round before; the iteration goes
   on while either shrank, and the first round in which neither did is discarded, its
   predecessor kept. A kind with no gap has no Δ and never keeps it going. At most ten
   rounds (`maxiter`), the last one kept if all ten improve. Unit-tested as a pure
   function (`still_converging`) on eight cases, because "both grew" and "either grew"
   return different rounds that look equally plausible.

4. **The learners.**
   - `missforest`: scikit-learn's random forests set to the R `randomForest` defaults
     missForest runs with, 100 trees; regression `max_features = 1/3`,
     `min_samples_leaf = 5`; classification `max_features = "sqrt"`,
     `min_samples_leaf = 1`. scikit-learn's own regressor defaults (every feature, one row
     per leaf) would be a different forest. Categories are numbers to the forest, as most
     Python ports feed them (the R package splits factors by subsets).
   - `missforest_lgbm`: LightGBM's defaults (100 rounds, 31 leaves, learning rate 0.1),
     categories declared categorical. The user's request; it is in the spirit of
     `miceforest`, without that package's multiple imputation or mean matching.
   - Both seeded with `MISSFOREST_SEED = 0`, on eight threads (`THREADS`). Neither result
     depends on the thread count: each tree draws from the forest's seed, and LightGBM runs
     `deterministic=True, force_row_wise=True`. A unit test fills alike at one and at
     eight threads, which matters because the cache keys an entry on the machine, not its
     cores. Another pins every global stream (`random`, numpy, torch) untouched, as ADR
     0007 decision 6 requires.
   - A column with no observed value, one observed value or no feature falls back to a
     constant, as `hgb` does. A training categorical column with no observed category is
     refused by name.

5. **Keys, populations, best and gap: ADR 0007's, with two more names.**
   `impute/<population>/baseline/missforest/<metric>` and `.../missforest_lgbm/<metric>`,
   on the same populations, with the same per-column rows. The summary's best baseline is
   chosen among all six, so `baseline/best`, the tag `best_baseline/<population>` and both
   `gap_to_best_baseline` keys follow without a code change.

6. **The cache keeps a family of entries per module set.** `BaselineScorer` caches ADR
   0007's four under the key they always had (its source digest still reads
   `imputation_baselines.py` and `imputation_metrics.py`, unchanged), and the two
   missForests under a key that also reads `missforest.py` and the LightGBM version.
   Editing missForest misses only its own entries; editing the ADR 0007 module, which
   missForest imports, misses both. Checked on the real cache: the backfill's recomputed
   cells give, on four runs of 2026-10-05/06, the very keys their live runs wrote (40 of
   40 found), so the existing 1,274 entries stayed valid and the backfill fills the cache
   a later live run on the same fold reads.

7. **No flag.** ADR 0007 decision 5 stands: a baseline is a fixed bar. The cost above
   is paid once per fold and population, then read from the cache by
   every other arm, seed-mate and trial; `--no_baseline_cache` already computes
   everything every time for whoever needs that.

8. **`lightgbm>=4.7` is a dependency** (4.7.0 locked; nothing else in `uv.lock` moved). It
   ships its own type hints; the two random forests bring the hand-written scikit-learn
   stubs to eighteen symbols. The Windows checkout needs `uv sync` after pulling.

9. **The backfill, `scripts/backfill_missforest_baselines.py`.** For every finished
   imputation parent lacking either name, it recomputes the mean/mode baseline and the two
   missForests on the run's own folds and cells (`scripts/backfill_baseline_imputers.py`'s
   recomputation, now taking a pluggable fold scorer), and writes them where a live run
   would have. The check is the mean/mode baseline: every mean/mode mean the run logged, on
   every population and metric, must be reproduced within 1e-9, or the run is refused. The
   KNN and `hgb` baselines are not recomputed (an hour of KNN on electricity alone), so not
   checked again. Where a missForest becomes a population's best baseline, every statistic
   under `baseline/best` is rewritten as its own, the tag renamed, and the gap taken again
   fold by fold from the model's values in `metrics/raw_fold_metrics.csv`; they are logged
   again at step 0 (MLflow's latest value is the newer one; the history keeps both). The
   tree is tagged `missforest_backfilled=true` and re-mirrored. `cv_summary.json` is not
   rewritten, as no earlier backfill rewrote it: on a run whose best moved, that artifact
   names the old best. The scores go through the live cache, so runs sharing a fold pay
   once and dry runs per table can fill it in parallel before one `--apply`. An entry is
   per population: a run that also scored extra mask rates misses on those, and its fold's
   missForests are fit again to fill them. A unit test applies a moved best to a
   temporary store and reads the new value back from the run and its mirror: rewritten at
   step 0 days after the original, it is MLflow's latest.

## Bias

ADR 0007 argued its bar errs against the model, the safe direction for a claim that the
model beats it. An inductive missForest sees fewer rows than the canonical transductive
one: it never learns from the test fold's own observed cells. That weakens the bar, the
unsafe direction. The model is scored the same inductive way, which is why the protocol
stays; measured on fold 1 of seed 42, refitting on the training rows plus the test rows
(scored cells hidden) instead:

| variant | population | `missforest` inductive → transductive | `missforest_lgbm` inductive → transductive |
|---|---|---|---|
| `credit-g_20nan` | masked | 0.866 → 0.865 | 0.939 → 0.909 |
| | induced | 0.855 → 0.869 | 0.935 → 0.868 |
| `spambase_20nan` | masked | 0.963 → 0.952 | 0.996 → 0.958 |
| | induced | 0.887 → 0.879 | 0.904 → 0.911 |
| `kr-vs-kp_40nan` | masked | 0.820 → 0.820 | 0.878 → 0.829 |
| | induced | 0.816 → 0.832 | 0.817 → 0.819 |
| `pendigits_20nan` | masked | 0.420 → 0.418 | 0.489 → 0.489 |
| | induced | 0.320 → 0.316 | 0.381 → 0.377 |

For the forest, the transductive fit moves `impute_score` by 0.016 at most, in no
consistent direction: better on five rows by 0.001 to 0.011, worse on two by 0.014 and
0.016 (there the inductive bar is the stricter one), level on one. LightGBM gains up to 0.067 from the
extra rows on the small or categorical tables, where its leaves of at least twenty rows
are starved; but inductive, it scores above the forest on every row, and transductive it
at best draws level (`credit-g_20nan` induced, 0.868 against 0.869), so it does not set
the bar here. One fold of one seed, four tables: a size, not an interval.

## Considered options

Smoke-tested on gorgona8 before the choice, against the locked scikit-learn 1.9, numpy 2.4
and torch 2.5:

- **ReMasker** (Du, Melis and Wang, ICLR 2024), a transformer masked autoencoder: the
  architecture closest to the model's, and the comparison a reader of the thesis asks for
  next. Deferred, not rejected: no new dependency, but GPU training beside the model,
  about 40 GPU-hours to backfill, and determinism to engineer ([BACKLOG I3](../BACKLOG.md)).
- **HyperImpute** (Jarrett et al., ICML 2022). Runs, but `hyperimpute` 0.1.17 breaks on
  scikit-learn 1.9 (`LogisticRegression(multi_class=...)`, removed) unless logistic
  regression leaves its search space; it moves the global random streams; its `fit` does
  nothing and its `transform` fits on the frame it fills, so it would be transductive on
  the test fold alone; and it pulls catboost, xgboost, lightgbm and their GPU wheels.
  47 s per credit-g fold on 200 rows.
- **TabPFN v2.5 per column** (Hollmann et al., *Nature* 2025). Blocked: the weights need a
  Prior Labs licence acceptance and an API token, and its ~10,000-row context would
  subsample letter and electricity, the tables where the bar decides.
- **missForest through `IterativeImputer`.** Rejected: one regressor for every column
  regresses category indices as numbers.
- **The transductive missForest of the R package.** Rejected for the protocol, and
  quantified under *Bias*.
- **A flag to turn the new family off.** Rejected, decision 7.

## Consequences

- Fourteen more fold keys per scored population (`test/.../baseline/missforest*/...`),
  seventy more parent keys; two more per-column row groups.
- `best_baseline/<population>` may now name `missforest` or `missforest_lgbm`; every report
  of E30 to E37 that read the best baseline read the four-baseline bar. ADR 0007's
  *Outcome* table is that bar's, as of 2026-09-28.
- `src/mlflow_utils.MISSFOREST_BACKFILLED_TAG` marks the backfilled trees.
- A claim that the model beats a missForest bar by less than about 0.02 should say that
  the bar is the inductive one, and that a transductive forest moved by up to 0.016 on the
  four tables measured (*Bias*).

## Outcome

**Backfill applied 2026-10-06, 08:53 to 10:28 GMT-3.** 873 imputation parents now carry both
missForests, marked `missforest_backfilled=true` with their children and mirrors; on 476 of
them a missForest became the best baseline of some population, and `baseline/best`, its tag
and both gaps moved with it. Refused: the superseded electricity run of 2026-09-25 (its
mean/mode was scored before the spelling fix, as ADR 0007 found) and 53 runs that predate
ADR 0007 and log no baseline to check. The store was copied first with SQLite's backup API to
`/var/tmp/diogoneiss/backups/mlflow.db.before-missforest-backfill`. A first pass scanned the
whole mirror experiment once per tree (23 s each); it was stopped after 24 trees, the one in
flight re-mirrored, and the pass resumed with one shared index and the parent's mark written
last, so a pass cut short is planned again.

**The bar, before and after, on today's default** (E35: seeds 42, 7 and 13, five folds; the
model's score is the K readout ADR 0015 adopted; each cell the mean over the three seeds of
the logged CV means; lower is better; bold where the model is no longer below the bar):

| variant | induced: model | bar of 4 | bar of 6 | masked: model | bar of 4 | bar of 6 |
|---|---|---|---|---|---|---|
| `biodeg_20nan` | 0.587 | hgb 0.636 | missforest 0.618 | 0.544 | hgb 0.627 | missforest 0.570 |
| `biodeg_60nan` | 0.761 | hgb 0.866 | missforest 0.762 | **0.809** | hgb 0.873 | missforest 0.786 |
| `credit-g_20nan` | 0.882 | hgb 0.918 | missforest 0.885 | 0.880 | hgb 0.931 | missforest 0.896 |
| `credit-g_40nan` | **0.956** | hgb 0.964 | missforest 0.931 | **0.982** | hgb 0.997 | missforest 0.950 |
| `credit-g_60nan` | 0.992 | mean_mode 1.000 | mean_mode 1.000 | 0.980 | mean_mode 1.000 | mean_mode 1.000 |
| `credit-g_80nan` | 1.008 | mean_mode 1.000 | mean_mode 1.000 | 1.008 | mean_mode 1.000 | mean_mode 1.000 |
| `electricity_20nan` | 0.618 | hgb 0.606 | hgb 0.606 | 0.686 | hgb 0.690 | hgb 0.690 |
| `kc2_20nan` | 0.541 | knn5 0.567 | knn5 0.567 | 0.530 | knn5 0.559 | knn5 0.559 |
| `kc2_60nan` | 0.572 | knn5 0.717 | missforest 0.675 | 0.552 | knn10 0.717 | missforest 0.626 |
| `kr-vs-kp_20nan` | 0.586 | hgb 0.612 | hgb 0.612 | 0.649 | hgb 0.680 | hgb 0.680 |
| `kr-vs-kp_40nan` | 0.758 | hgb 0.773 | hgb 0.773 | 0.781 | hgb 0.780 | hgb 0.780 |
| `kr-vs-kp_60nan` | 0.865 | hgb 0.903 | hgb 0.903 | 0.887 | hgb 0.912 | hgb 0.912 |
| `kr-vs-kp_80nan` | 0.976 | mean_mode 1.000 | mean_mode 1.000 | 1.000 | mean_mode 1.000 | mean_mode 1.000 |
| `letter_20nan` | 0.459 | knn10 0.473 | missforest 0.462 | 0.496 | knn10 0.542 | missforest 0.535 |
| `pendigits_20nan` | 0.332 | knn10 0.350 | missforest 0.334 | 0.369 | knn10 0.401 | knn10 0.401 |
| `spambase_20nan` | 0.848 | hgb 0.866 | missforest 0.859 | 0.865 | hgb 0.887 | hgb 0.887 |
| `spambase_40nan` | 0.903 | hgb 0.937 | hgb 0.937 | 0.897 | hgb 0.939 | hgb 0.939 |
| `spambase_60nan` | 0.942 | hgb 0.984 | missforest 0.981 | 0.904 | hgb 0.983 | missforest 0.974 |
| `spambase_80nan` | 0.987 | mean_mode 1.000 | mean_mode 1.000 | 0.993 | mean_mode 1.000 | mean_mode 1.000 |
| `vehicle_20nan` | 0.418 | hgb 0.473 | missforest 0.452 | 0.420 | knn5 0.505 | missforest 0.472 |
| `vehicle_60nan` | 0.645 | hgb 0.701 | missforest 0.647 | **0.616** | hgb 0.671 | missforest 0.616 |

- **The model is below the bar on 18 of 21 variants induced (19 before) and 15 of 21 masked
  (18 before).** It loses `credit-g_40nan` on both populations, and `biodeg_60nan` and
  `vehicle_60nan` (a tie to three decimals) masked.
- **`missforest` sets the bar on 11 of 21 induced and 9 of 21 masked;** `hgb` keeps it on
  `kr-vs-kp`, `spambase_40nan` and `electricity`, KNN on `kc2_20nan`, the mean/mode fill at
  80% missingness and on `credit-g_60nan`. **`missforest_lgbm` sets it nowhere**: below the
  forest on every variant of both populations.
- **Six of the 18 induced wins are inside the 0.016 the *Bias* section measured** between
  inductive and transductive forests: `biodeg_60nan` (0.001), `pendigits_20nan` (0.002),
  `vehicle_60nan` (0.002), `credit-g_20nan` (0.003), `letter_20nan` (0.003) and
  `spambase_20nan` (0.011). They are ties to a canonical missForest, not wins.
- The table reads means over three seeds; the per-run gaps and their paired intervals are
  in the store (`gap_to_best_baseline*`), as for every run since ADR 0007 decision 9.
