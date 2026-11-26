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

## Amendment (2026-11-25 14:28 GMT-3; no cell finished, nothing read): three cells do not fit the GPU

- **The failure:** `gas-drift_20nan` on seed 7 failed at 14:26 GMT-3 with CUDA out of memory, in the decode stage of
  its first fold, while the seed-42 cell held 10.8 GB (its first fold's scoring stage) and the seed-13 cell 5.3 GB of
  the card's 24 GB. Its run is `FAILED` in the store.
- **What I did:** stopped queue 2 (seed 7) at 14:27 GMT-3 and interrupted the cell it had started a minute before
  (`gas-drift_40nan` seed 7, also `FAILED` in the store, nothing trained). Queues 1 and 3 (seeds 42 and 13) were not
  touched and go on, two cells at a time.
- **Seed 7** runs its four cells from the start, as queue 2 relaunched by a waiting process when the first of the
  other two queues ends, so at most two cells share the GPU. This replaces "rerun alone" for the failed cell; if a
  cell fails for memory again with two on the card, it is rerun alone.
- Cost only: the design, the rule and the cells are unchanged, and each cell is read from its one finished run.

## Amendment (2026-11-25 15:47 GMT-3; no cell finished, nothing read): seed 7 in two queues

- **Two cells fit:** with the seed-42 and seed-13 cells of `gas-drift_20nan` both in their third fold, the card's
  peak was 22,097 MiB of 24,564 (11,040 and 11,042 MiB per process), measured every 30 s from 14:28 to 15:46 GMT-3.
- **Scheduling of seed 7:** the two other queues end minutes apart, so one queue for seed 7 would run its four cells
  alone. The waiting process of the previous amendment was replaced at 15:47 GMT-3 by two: when queue 1 (seed 42)
  ends, `c15_queue.sh 2 6` runs seed 7's 20nan and 60nan; when queue 3 (seed 13) ends, `c15_queue.sh 5 6` runs its
  40nan and 80nan. Still at most two cells on the GPU, and a cell that fails for memory is rerun alone.
- Cost only: the design, the rule and the cells are unchanged.

## Amendment (2026-11-26 16:34 GMT-3; 12 runs done, no score read): the report's readout at 20nan

- **What happened:** all 12 runs finished (the last at 16:27 GMT-3 on 2026-11-26). The report, run at 16:29 GMT-3, stopped
  before printing anything: `KeyError: a 'induced' run carries no KC readout`. The default calibrates its output
  from 40nan up (ADR 0027), so the three `gas-drift_20nan` runs carry no calibrated column, and the script asked for
  `KC` on every variant. It had only been tested on the 40nan verification cell.
- **The fix:** the script reads `KC` where the run is calibrated and `K` where it is not, which in both cases is
  the run's headline `impute/induced/impute_score`, the quantity this ticket registered ("the default's output").
  An assert checks that equality on every run. The wall-time line's label was also corrected (two cells at a time,
  not three). Column names were listed to find the cause; no score was printed or opened.
- Design, rule and margin unchanged.

## Outcome (read once 2026-11-26 16:35 GMT-3; written 2026-11-26 16:37 GMT-3)

**Supported.** By the registered rule the default's output is better than the bar's best baseline on all four
variants, and the difference grows with the gaps.

| variant | decoder | bar's best | decoder − bar [95%] | by seed (42, 7, 13) | call |
|---|---|---|---|---|---|
| `gas-drift_20nan` | 0.1403 | 0.1645 | −0.0242 [−0.0313, −0.0170] | −0.0241, −0.0273, −0.0211 | better |
| `gas-drift_40nan` | 0.1593 | 0.2094 | −0.0501 [−0.0595, −0.0407] | −0.0467, −0.0552, −0.0484 | better |
| `gas-drift_60nan` | 0.1643 | 0.2451 | −0.0808 [−0.0884, −0.0732] | −0.0773, −0.0863, −0.0789 | better |
| `gas-drift_80nan` | 0.2461 | 0.4351 | −0.1890 [−0.1936, −0.1843] | −0.1894, −0.1890, −0.1885 | better |

Induced `impute_score`, lower is better, 15 fold pairs per variant. The bar's best is chosen per run, so it is not
one method: LightGBM at 20nan, kNN-5 or LightGBM at 40nan, kNN-5 at 60nan, HGB or LightGBM at 80nan.

- **The readout at 20nan is the uncalibrated output:** ADR 0027 calibrates from 40nan up, so the default's output at
  20nan is the run's own checkpoint without calibration. On every variant the number read is the run's headline
  `impute/induced/impute_score` (amendment of 16:34 GMT-3).
- **Beside it, the strong tier the runs log (no study-level call):**
  - 20nan: −0.0073 [−0.0155, +0.0010] against 0.1476 (ICE or missForest by run). No difference called, and not
    equivalent: the interval reaches past the margin of 1% of the opponent's mean (0.0015).
  - 40nan: −0.0106 [−0.0198, −0.0013] against missForest's 0.1699, "better", but thin: the upper end is −0.0013,
    nothing here is corrected for the four variants, and 40nan is the one level seen in the prior look.
  - 60nan: −0.0358 [−0.0410, −0.0306] against missForest's 0.2001, better.
  - 80nan: −0.1519 [−0.1554, −0.1484] against the conditional expectation's 0.3980, better.
- **Absolute error reads the other way at 20 to 60nan** (descriptive means, no interval, no call): missForest's
  `mae_num_z` is below the decoder's at 20nan (0.0337 against 0.0492), 40nan (0.0409 against 0.0581) and 60nan
  (0.0664 against 0.0713); at 80nan the decoder's is the lowest (0.1104 against missForest's 0.1995). Against the
  bar's members the decoder's `mae_num_z` is the lowest at every level.
- **The table's scale check:** the mean fill's `rmse_num_z` is 0.944 to 1.031 over the 60 folds, so no fold's scale
  is carried by a few rows, as it was before the 15 rows were dropped.
- **Runs:** 12 `FINISHED` parent runs in the store under `experiment = wide-table-standing-2026-11-25` (and their 12
  mirrors), plus the 2 `FAILED` runs of the memory failure (seed 7 at 20nan, and the 40nan cell interrupted a
  minute after it started). All on gorgona8. Wall time 3.48 to 5.22 h per cell; first cell started 2026-11-25 13:36
  GMT-3, last ended 2026-11-26 16:27 GMT-3.
- **Deviations, all in the amendments above:** three cells at a time did not fit the GPU, so the study ran two at a
  time and seed 7's cells ran after the other two seeds'; the report script was committed after the launch and
  fixed, before any score was read, to take the headline at 20nan. Its first invocation (16:29 GMT-3) printed only
  a traceback. Design, rule and margin as registered.

**What this does not say.**
- TabPFN v2 is not in it, so "the strong tier" is only missForest, ICE, the conditional expectation and SoftImpute.
- The lead over missForest is on a squared-error score, and by absolute error missForest is ahead at 20 to 60nan.
  The table still holds 14 rows beyond 20 standard deviations after the one pass of the row rule; whether the
  squared-error lead depends on such rows was not tested.
- One table, MCAR only, one seed set, and the row rule was chosen after a prior look at this table. A claim about
  wide tables needs `nomao` under its own pre-registration; nothing was read or chosen on `nomao`.
- No default change and no ADR.

**The printout, unedited** (`wide_table_report.py`, stdout and stderr, exit 0; nothing omitted):

```
gas-drift_20nan: decoder 0.1403 over 15 folds
  decoder - bar (0.1645, best by run: ['lgbm']): -0.0242 [-0.0313, -0.0170], by seed -0.0241, -0.0273, -0.0211 -> better
  decoder - strong_live (0.1476, best by run: ['ice', 'missforest']): -0.0073 [-0.0155, +0.0010], by seed -0.0041, -0.0093, -0.0084 -> none
  mean impute_score: decoder 0.1403, missforest 0.1487, ice 0.1500, lgbm 0.1645, hgb 0.1671, knn5 0.1879, condexp 0.1882, knn10 0.2092, softimpute 0.3118, mean_mode 1.0000
  mean rmse_num_z: decoder 0.1408, missforest 0.1494, ice 0.1504, lgbm 0.1651, hgb 0.1677, knn5 0.1886, condexp 0.1887, knn10 0.2100, softimpute 0.3126, mean_mode 1.0022
  mean mae_num_z: missforest 0.0337, decoder 0.0492, ice 0.0493, knn5 0.0521, lgbm 0.0544, hgb 0.0550, knn10 0.0601, condexp 0.0851, softimpute 0.1681, mean_mode 0.7095
  mean fill's rmse_num_z per fold: 0.9444 to 1.0305
gas-drift_40nan: decoder 0.1593 over 15 folds
  decoder - bar (0.2094, best by run: ['knn5', 'lgbm']): -0.0501 [-0.0595, -0.0407], by seed -0.0467, -0.0552, -0.0484 -> better
  decoder - strong_live (0.1699, best by run: ['missforest']): -0.0106 [-0.0198, -0.0013], by seed -0.0076, -0.0175, -0.0066 -> better
  mean impute_score: decoder 0.1593, missforest 0.1699, ice 0.1906, lgbm 0.2119, knn5 0.2122, hgb 0.2135, knn10 0.2390, softimpute 0.2705, condexp 0.3042, mean_mode 1.0000
  mean rmse_num_z: decoder 0.1597, missforest 0.1705, ice 0.1912, lgbm 0.2127, knn5 0.2131, hgb 0.2143, knn10 0.2400, softimpute 0.2714, condexp 0.3051, mean_mode 1.0033
  mean mae_num_z: missforest 0.0409, decoder 0.0581, knn5 0.0591, knn10 0.0690, ice 0.0723, lgbm 0.0790, hgb 0.0797, softimpute 0.1423, condexp 0.1616, mean_mode 0.7111
  mean fill's rmse_num_z per fold: 0.9504 to 1.0270
gas-drift_60nan: decoder 0.1643 over 15 folds
  decoder - bar (0.2451, best by run: ['knn5']): -0.0808 [-0.0884, -0.0732], by seed -0.0773, -0.0863, -0.0789 -> better
  decoder - strong_live (0.2001, best by run: ['missforest']): -0.0358 [-0.0410, -0.0306], by seed -0.0340, -0.0396, -0.0337 -> better
  mean impute_score: decoder 0.1643, missforest 0.2001, softimpute 0.2387, knn5 0.2451, ice 0.2665, hgb 0.2715, knn10 0.2721, lgbm 0.2722, condexp 0.3352, mean_mode 1.0000
  mean rmse_num_z: decoder 0.1645, missforest 0.2003, softimpute 0.2389, knn5 0.2454, ice 0.2668, hgb 0.2718, knn10 0.2725, lgbm 0.2726, condexp 0.3355, mean_mode 1.0009
  mean mae_num_z: missforest 0.0664, decoder 0.0713, knn5 0.0741, knn10 0.0872, softimpute 0.1192, hgb 0.1216, lgbm 0.1216, ice 0.1296, condexp 0.1868, mean_mode 0.7106
  mean fill's rmse_num_z per fold: 0.9495 to 1.0246
gas-drift_80nan: decoder 0.2461 over 15 folds
  decoder - bar (0.4351, best by run: ['hgb', 'lgbm']): -0.1890 [-0.1936, -0.1843], by seed -0.1894, -0.1890, -0.1885 -> better
  decoder - strong_live (0.3980, best by run: ['condexp']): -0.1519 [-0.1554, -0.1484], by seed -0.1516, -0.1523, -0.1518 -> better
  mean impute_score: decoder 0.2461, condexp 0.3980, missforest 0.4055, hgb 0.4351, lgbm 0.4353, softimpute 0.5033, knn10 0.5874, knn5 0.6296, ice 0.8016, mean_mode 1.0000
  mean rmse_num_z: decoder 0.2461, condexp 0.3979, missforest 0.4053, hgb 0.4350, lgbm 0.4352, softimpute 0.5031, knn10 0.5872, knn5 0.6294, ice 0.8009, mean_mode 0.9995
  mean mae_num_z: decoder 0.1104, missforest 0.1995, lgbm 0.2225, hgb 0.2227, condexp 0.2253, softimpute 0.2876, knn10 0.3063, knn5 0.3372, ice 0.5107, mean_mode 0.7105
  mean fill's rmse_num_z per fold: 0.9497 to 1.0187
AGAINST THE BAR over 4 variants: better 4 / no difference 0 / worse 0 (fold-pair t, seed agreement) -> supported
wall time per cell, hours (two cells sharing the GPU; three until 14:27 GMT-3 on the first day): K_gas-drift_20nan_s42 5.22, K_gas-drift_20nan_s7 4.92, K_gas-drift_20nan_s13 5.03, K_gas-drift_40nan_s7 4.74, K_gas-drift_40nan_s13 4.37, K_gas-drift_40nan_s42 4.48, K_gas-drift_60nan_s13 4.31, K_gas-drift_60nan_s42 4.16, K_gas-drift_80nan_s13 4.36, K_gas-drift_80nan_s42 4.36, K_gas-drift_80nan_s7 3.48, K_gas-drift_60nan_s7 3.70
```
