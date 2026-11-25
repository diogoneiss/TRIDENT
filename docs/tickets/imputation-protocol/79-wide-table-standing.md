# Where does the imputation default stand on a wide table? (E190, T67; pre-registered 2026-11-25 13:35 GMT-3)

## Why

A reviewer asked for a table with over 100 columns and over 10,000 rows. The widest table the project had read is
`spambase` (57 columns) and the widest with over 10,000 rows has 16 (`pendigits`, `letter`). The user adopted two
OpenML tables on 2026-11-25: `gas-drift` (128 numerical features, 6 classes) as the development table and `nomao` as
the confirmation table, on which nothing is tuned or chosen (`datasets/build_wide_tables.py`). No development on
`gas-drift` can be read without knowing where today's default stands there.

The user's go: 2026-11-25, "vamos no 1", to my proposal of the same day (the default against the reference imputers
on `gas-drift` at 20, 40, 60 and 80% gaps, 3 seeds, 12 cells).

- **Hypothesis:** on `gas-drift` the default decoder's fills are better than the best baseline of the bar's at every
  level of gaps.

## Design and rule (fixed before the first cell)

- **Table:** `gas-drift` as committed in `d5ab8ee`: 13,895 rows (OpenML 1476 without 15 rows, see Prior looks), 128
  numerical features, no categorical column. Variants `gas-drift_20nan`, `_40nan`, `_60nan`, `_80nan`: MCAR gaps cut
  by `generate_splits.py`'s `inject_nans` with its seeds, as for every other table.
- **Arm:** `K`, the imputation defaults of the checkout with no other flag (ADR 0013's defaults, ADR 0014's gap
  token, ADR 0015's checkpoint, ADR 0027's calibrated output). The baselines are scored by the run itself on the
  same cells.
- **Cells:** 4 variants × seeds 42/7/13, five folds each: 12 runs. These seeds have never run on this table.
- **Population and readout:** the induced cells (the variant's gaps, scored against the complete table); the
  default's output, read as `KC` by `report_common.model` (the run's headline `impute/induced/impute_score`).
- **Primary:** per variant, the decoder's `impute_score` minus the bar's best baseline's
  (`report_common.opponent(frame, "bar")`: the best of mean/mode, kNN-5, kNN-10, HGB and LightGBM, chosen once per
  run on its cross-validation mean), paired per (seed, fold), 15 replicates. Called by `report_common.verdict`, the
  project's rule: **better** (or **worse**) when the plain 95% t interval of the pooled fold differences lies below
  (above) zero and every seed's mean difference agrees in sign; **equivalent** when neither holds and the interval
  lies within 1% of the opponent's mean score; otherwise no difference.
- **Study-level call:** **supported** with "better" on all four variants, **refuted** with "better" on none, **mixed**
  otherwise.
- **Beside it, same rule, no study-level call:** the decoder against the strong tier the runs log
  (`report_common.opponent(frame, "strong_live")`: the best of missForest, ICE, the conditional expectation and
  SoftImpute; ADR 0021). The prior look below showed the decoder ahead of missForest by `impute_score` and behind it
  by absolute error, so both are printed.
- **Descriptive:** every scored imputer's mean `impute_score`, `rmse_num_z` and `mae_num_z` per variant; the mean
  fill's `rmse_num_z` per fold (the check that no fold's scale is carried by a few rows); wall time per cell.
- **What this cannot say:** TabPFN v2 (ADR 0028's member of the strong tier) is not in this study: the table is over
  its 10,000-row limit and how to cap it is the user's decision (E176 did it on `mimic-wide`). "Strong tier" here is
  only what the runs log. One table, MCAR only, one seed set: a first reading; a claim about wide tables needs
  `nomao` under its own pre-registration.
- **What follows:** no default change and no ADR either way. Nothing is read or chosen on `nomao`.
- **Read once, only with all 12 runs done;** the report refuses a missing cell. It is
  `scripts/experiments/wide_table_report.py`, written after this commit and tested only on the verification cell
  below (seed 3131, not a study seed); everything it computes is fixed by the functions named above.

## Prior looks (all on seed 3131, `gas-drift_40nan`, off the store)

- **The uncleaned table** (13,910 rows; 2026-11-25 00:40 to 04:20 GMT-3): `impute_score` 0.650 ± 0.296, per fold
  0.605, 0.784, 0.669, 1.0, 0.194; one fold's mean fill had `rmse_num_z` 55.7 and every method scored 1.0 there.
- **The row rule was chosen after that cell:** drop every row holding a cell more than 20 population standard
  deviations from its column's mean over the full table, one pass, 15 rows (the user's decision, 2026-11-25). I then
  compared eight rules by the mean fill's error on folds of my own; that comparison did not change the rule. After
  the pass 14 rows exceed 20 again when re-standardised (at most 50).
- **The cleaned table** (2026-11-25 09:28 to 13:13 GMT-3, 3 h 45): the mean fill's `rmse_num_z` 0.99 to 1.03 per
  fold. Mean `impute_score` over the 5 folds: decoder 0.1592 (0.131 to 0.197), missForest 0.1693, ICE 0.1905, kNN-5
  0.2025, LightGBM 0.2108, HGB 0.2133, kNN-10 0.2309, SoftImpute 0.2681, conditional expectation 0.3053. So at 40nan
  on that seed the decoder was below the bar's best (kNN-5) and below missForest on 5 of 5 folds by `impute_score`;
  by `mae_num_z` the decoder (0.0568) was behind missForest (0.0411) on 5 of 5.
- Nothing was run at 20, 60 or 80nan, and nothing on seeds 42, 7 or 13.

## Run

- Runner `scripts/experiments/wide_table_run.py` through `c15_queue.sh`; state directory
  `/scratch2/diogoneiss/trident-2026-11-25-wide-standing`. Tracked in MLflow on the leader (gorgona8): tag
  `experiment = wide-table-standing-2026-11-25`, arm `K`.
- Three queues on gorgona8's one GPU, one seed each (42, 7, 13), each running its four levels in order; starts
  staggered. This is cost only: a cell's scores do not depend on what runs beside it beyond GPU nondeterminism. The
  queue script caps each process at 2 CPU threads, and the 3 h 45 above was measured alone and uncapped, at one
  level, so the wall time is unknown until the first cells close. The GPU memory of the scoring stage was never
  measured on this table; **a cell that fails for memory is rerun alone on the same seed**, and the failure is
  recorded here.
- **Started 2026-11-25 13:36 GMT-3** (queue 1; queues 2 and 3 a few minutes apart), after the pre-registration
  commit `29aaff1`.

## Amendment (2026-11-25 13:37 GMT-3; no cell finished)

The report script `scripts/experiments/wide_table_report.py` is committed now, with no study cell finished. It was
run only with `--check` on the verification cell of the Prior looks (seed 3131), where it prints the numbers
already disclosed there (decoder 0.1592, kNN-5 0.2025, missForest 0.1693). Nothing in the design or the rule changes.
