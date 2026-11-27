# Does the default's standing on a wide table repeat on `nomao`? (E191, T68; pre-registered 2026-11-27 00:25 GMT-3)

## Why

E190 ([ticket 79](79-wide-table-standing.md)) read the imputation default on `gas-drift`, the development table of
the two wide tables a reviewer asked for: better than the bar's best baseline on 4 of 4 levels of gaps. One table
is not a claim about wide tables. `nomao` (OpenML 1486; 34,465 rows, 89 numerical and 29 categorical features, 2
classes, every row kept) is the confirmation table: nothing is tuned or chosen on it, and reading the unchanged
default there chooses nothing.

The user's go: 2026-11-26, "vamos no 1", to my proposal of the same day (the same study on `nomao`, 12 cells,
about 55 h, one cell at a time).

- **Hypothesis:** E190's reading repeats: on `nomao` the default decoder's fills are better than the best baseline
  of the bar's at every level of gaps.

## Design and rule (fixed before the first cell)

Ticket 79's, on the other table. Restated in full:

- **Table:** `nomao` as committed in `5d23f15`; variants `nomao_20nan`, `_40nan`, `_60nan`, `_80nan`, MCAR gaps cut
  by `generate_splits.py`'s `inject_nans` with its seeds (`datasets/build_wide_tables.py`).
- **Arm:** `K`, the imputation defaults of the checkout with no other flag. The baselines are scored by the run
  itself on the same cells.
- **Cells:** 4 variants × seeds 42/7/13, five folds each: 12 runs. These seeds have never run on this table.
- **Population and readout:** the induced cells; the default's output, which is the run's headline
  `impute/induced/impute_score` (calibrated from 40nan up by ADR 0027, the uncalibrated output at 20nan).
- **Primary:** per variant, the decoder's `impute_score` minus the bar's best baseline's
  (`report_common.opponent(frame, "bar")`: the best of mean/mode, kNN-5, kNN-10, HGB and LightGBM, chosen once per
  run on its cross-validation mean), paired per (seed, fold), 15 replicates. Called by `report_common.verdict`:
  **better** (or **worse**) when the plain 95% t interval of the pooled fold differences lies below (above) zero and
  every seed's mean difference agrees in sign; **equivalent** when neither holds and the interval lies within 1% of
  the opponent's mean score; otherwise no difference.
- **Study-level call:** **supported** with "better" on all four variants (E190's reading replicated on the second
  table), **refuted** with "better" on none, **mixed** otherwise.
- **Secondary, from E190's descriptive reading (same rule, no study-level call):** per variant, the decoder's
  `mae_num_z` minus missForest's on the same cells, margin 1% of missForest's mean. On `gas-drift` missForest had
  the lower mean absolute error at 20, 40 and 60nan and the decoder at 80nan, untested. Stated before the first
  cell: I expect the same pattern here. It covers the 89 numerical columns only.
- **Beside it, same rule, no study-level call:** the decoder's `impute_score` against the strong tier the runs log
  (`report_common.opponent(frame, "strong_live")`: the best of missForest, ICE, the conditional expectation and
  SoftImpute; ADR 0021).
- **Descriptive:** every scored imputer's mean `impute_score`, `rmse_num_z`, `mae_num_z`, `acc_cat` and
  `macro_f1_cat` per variant; the mean fill's `rmse_num_z` per fold; wall time per cell.
- **What this cannot say:** TabPFN v2 is not in it (the table is over its 10,000-row limit; how to cap it is the
  user's decision), so "strong tier" is only what the runs log. MCAR only, one seed set. 27 of the 29 categorical
  columns take three values, and whether one of them encodes "missing" in the original data was not checked. With
  E190 this makes two wide tables, both numerical or mostly numerical.
- **What follows:** no default change and no ADR either way.
- **Read once, only with all 12 runs done,** by
  `WIDE_TABLE=nomao QUEUE_STATE=<state> uv run --python 3.11 python -B scripts/experiments/wide_table_report.py`,
  which refuses a missing cell. The report is E190's with the table taken from `WIDE_TABLE`, the secondary above and
  the two categorical means added; the additions were tested with `--check` on two old `credit-g` runs from the
  store (20nan, uncalibrated, and 40nan, calibrated), never on a `nomao` cell.

## Prior looks

- **One cell on `nomao_40nan`, seed 3131** (the cost test, 2026-11-24 14:03 to 18:39 GMT-3, off the store, alone on
  gorgona8's GPU, 4 h 36): decoder's induced `impute_score` 0.155 against the best baseline's 0.241 (HGB), lower on
  5 of 5 folds. The roles of the two tables (the user's decision, 2026-11-25) were set after that cell. Nothing was
  run at 20, 60 or 80nan, and nothing on seeds 42, 7 or 13.
- **E190 itself,** on the other table, is why the hypothesis and the secondary are worded as they are.

## Run

- Runner `scripts/experiments/wide_table_run.py` with `WIDE_TABLE=nomao`, through `c15_queue.sh`. Tracked in
  MLflow on the leader (gorgona8): tag `experiment = wide-table-nomao-2026-11-27`, arm `K`.
- **gorgona8, one cell at a time:** the cost test saw 20 GB of the 24 GB card during scoring. State directory
  `/scratch2/diogoneiss/trident-2026-11-27-wide-nomao`. Seed by seed (42, then 7, then 13), each through its four
  levels, 16 threads a cell (`QUEUE_THREADS`, new in `c15_queue.sh`; E190's cells had 2).
- **medusa1, one cell at a time on GPU 1 until 2026-11-27 10:00 GMT-3** (the user's instruction of 2026-11-27,
  "voce pode colocar uma celula na medusa para acelerar tambem, rode algo la ate as 10h de amanha. a gpu 1 cabe +1
  com folga"): seed 13 from 20nan upward, as many cells as end by 10:00; no cell is started that cannot, and one
  still running at 10:00 is interrupted and rerun on gorgona8. State directory
  `/scratch/diogoneiss/trident-2026-11-27-wide-nomao`; a finished cell's run directory is copied into gorgona8's
  state directory, where the report reads it and gorgona8's seed-13 queue skips it. This splits seed 13 between the
  machines cell by cell, on the user's instruction; ADR 0035 covers a split by seed, not this one. Every comparison
  the study reads is within one run, so within one machine, and the runs carry the `host` tag. medusa1's variants are built by the same
  script and compared by hash before its first cell.
- A cell that fails for memory is rerun on gorgona8 alone, and the failure is recorded here.
