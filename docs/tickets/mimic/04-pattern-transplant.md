# Real gap patterns transplanted onto complete rows (E149; pre-registered 2026-11-05 as E151)

Written and committed before any table exists; blocked on [01](01-external-datasets-root.md) and
[02](02-tables.md). The second way: give the natural patterns a ground truth.

- **Why:** [E148](03-masked-cells-natural-gaps.md) scores uniform draws among measured values. The cohort's real gaps
  come in blocks (55 distinct patterns on the cardio table, five of them 81% of the rows) and fall on particular
  patients. Rows with everything measured can carry those patterns and keep the truth.
- **Hypothesis:** under gap patterns copied from real stays, the decoder's fills are better than the best baseline
  of the bar's.

## Design and rule (fixed before the first cell)

- **Donor tables:** `mimic-cardio-donor_00nan` and `mimic-sofa-donor_00nan`, the rows complete on every measured
  column ([02](02-tables.md)). A donor table under 1,000 rows is not run.
- **Variants, three per donor table, cut by one seeded script:**
  - `tmatch`: each donor row takes the gap pattern of its nearest incomplete stay of the cohort, by Euclidean
    distance on the standardised columns with under 1% of gaps. Where the gaps fall then depends on the row.
  - `trand`: the same patterns, permuted among the donor rows. The blocks stay, the dependence on the row goes.
  - `mcar`: in each column, the same number of cells as `tmatch`, at uniform positions.

  The three have the same count of gaps in every column. The `_NNnan` in their names is the nearest of 20/40/60/80
  to that rate, a label.
- **Cells:** 2 donor tables × 3 variants × seeds 42/7/13, five folds: 18 runs, imputation defaults (the
  donor table is the complete sibling, so ADR 0015's `induced` checkpoint applies).
- **Primary:** per donor table, on `tmatch`, the induced cells' `impute_score` minus the best baseline's, paired per
  (seed, fold), 15 replicates, plain 95% t interval; Holm over the two tables. Better, worse or no different as in
  E148.
- **Beside it, descriptive:** `tmatch − trand` and `trand − mcar` for the decoder's score and for its gap to the bar
  (what the dependence on the row costs, and what the blocks cost); TabPFN (ADR 0028) on `tmatch`.
- **Threats, stated:**
  - Donors are the patients who had everything measured, and those are sicker: deaths 33.5% among the 1,292 cardio
    donors and 25.4% among the 2,143 SOFA donors, against 16.5% in the cohort.
  - A transplant carries where the gaps fall, not what the missing values were: `tmatch` is at most missing at
    random given the matching columns. Nothing here is a measurement under the true mechanism.
- **Cost:** estimated on 2026-11-06 from E148's first cells, before the first cell here: 33 and 36 minutes for
  11,082 rows. A donor table has a fifth to an eighth of the rows and the same epochs, so 5 to 15 minutes a cell
  and one to two and a half hours for the 18 on two queues, more if the other project's ten trainers slow them.
- **What follows:** no default change and no ADR either way.
- **Read only with all 18 runs done.**

## Implementation (2026-11-06, before any cell)

`datasets/make_mimic_variants.py` cut the variants on 2026-11-06 07:27 GMT-3. It fixes what the design left open:

- The pool is every stay with a gap in any feature (9,790 on the cardio table, 8,939 on the SOFA one); the donors
  are the other stays.
- The matching columns are the numeric ones with under 1% of gaps, each standardised over the whole cohort. On the
  cardio table: heart rate, SpO2, potassium, hemoglobin, hematocrit, sodium, the four vasopressors and age. On the
  SOFA table: the lowest mean pressure, the four vasopressor rates, the lowest GCS, the SpO2/FiO2 ratio and age.
- A gap in a matching column counts as the column's mean. Of equally near stays, the first in the table's order.
- A pattern is a stay's whole row of gaps, so the matching columns receive the few gaps the pool has in them.
- One seed, 42, draws `trand`'s permutation and `mcar`'s positions.

| donor table | rows | stays the patterns came from | gaps | share of feature cells | row patterns in `tmatch` / `trand` / `mcar` |
|---|---|---|---|---|---|
| `mimic-cardio-donor` | 1,292 | 1,052 | 3,659 | 12.9% | 30 / 30 / 171 |
| `mimic-sofa-donor` | 2,143 | 1,513 | 3,830 | 10.5% | 29 / 29 / 74 |

- Both shares are nearest to 20 of 20/40/60/80, so the files are `<base>-donor_{tmatch,trand,mcar}_20nan.csv`.
  `<base>-donor_sources.csv` names, row by row, the stay each pattern came from.
- The donors' neighbours are not the average stay, so a column's share of gaps differs from the cohort's. Cardio:
  troponin 68.0% (cohort 65.8%), arterial systolic pressure 57.7% (60.3%), lactate 18.0% (28.5%). SOFA: bilirubin
  64.8% (47.0%), PaO2/FiO2 35.4% (56.5%), urine output 33.3% (28.6%). The script's report is `variants_report.txt`
  in the datasets root.
- No donor row is left complete under `tmatch` or `trand`; `mcar` leaves 21 cardio rows and 190 SOFA rows complete.
- Runner: `scripts/experiments/mimic_transplant_run.py`, one arm (`D`, the defaults), MLflow tag
  `experiment = mimic-pattern-transplant-2026-11-06`. `scripts/experiments/run_mimic_chain.sh` starts its two queues
  when E148's have ended; `scripts/experiments/mimic_report.py e149` reads it.
- Smoke on `mimic-sofa-donor_tmatch_20nan`, the table with a categorical column among the gaps (2026-11-06, CPU, two
  folds, 5 pre-training and 10 decode epochs, MLflow off): the gaps are scored against the donor table with the
  bar's baselines, and the `induced` checkpoint is the one kept (`decode/induced_checkpoint_epoch` is logged).

## Run

- **Started 2026-11-06 10:15 GMT-3** by `run_mimic_chain.sh`, when E148's queues ended: two queues, 18 cells, state
  directory `/scratch2/diogoneiss/trident-2026-11-06-mimic-e149`; MLflow tag
  `experiment = mimic-pattern-transplant-2026-11-06`, arm `D`. The GPU is shared with kg-impute's trainers and, from
  10:17, with E148's TabPFN pass.
- **One cell failed at its start and is run again** (2026-11-06 10:15 GMT-3): `mimic-cardio-donor_tmatch_20nan`,
  seed 42. Both queues opened with a cell of the cardio donor table in the same second, both asked the tracking
  server to create the experiment `TRIDENT/mimic-cardio-donor`, and the second was refused
  (`RESOURCE_ALREADY_EXISTS`). It stopped before its parent run existed and before any training; its log is kept as
  `logs/D_mimic-cardio-donor_tmatch_20nan_s42.failed-1015.log`. A detached waiter relaunches the two queues when the
  first pass ends, and a relaunch runs only the cells not done. The race is in `get_or_create_experiment`
  (`src/mlflow_utils.py`) and is left as it is until this study ends, so every cell runs on one version of `src/`.
  Fixed after the last cell, with a test (commit 1a7f674): a refused create looks the experiment up again. It
  reaches the main checkout when E152's cells, which import `src/` from there, have ended.
- **The relaunch ran that cell 2026-11-06 14:37 to 15:06 GMT-3,** on its second attempt, and the queues were
  complete. E152's chain ([06](06-sofa-positive-control.md)) waits for "no E149 cell running" and fired at 14:37:23,
  in the eight seconds between the first pass's end and the relaunched cell's start at 14:37:31: that cell ran its
  29 minutes beside E152's first three trainers. The fold's cells and seeds do not depend on the load; only its wall
  time does.

## Outcome (2026-11-06 15:14 GMT-3)

**Supported on both donor tables.** All 18 runs done: 17 on the first attempt, one on its second (see Run). 18
parent runs FINISHED on the leader under the study's tag, none from the failed attempt. The report, as
`mimic_report.py e149` prints it:

```
mimic-cardio-donor_tmatch_20nan: decoder 0.8606, bar 0.9061 (best baseline by run: ['hgb', 'knn10', 'lgbm'])
  mean impute_score over 15 folds: decoder 0.8606, condexp 0.8616, missforest 0.8964, hgb 0.9080, lgbm 0.9104, knn10 0.9122, knn5 0.9395, ice 0.9992, mean_mode 1.0000, softimpute 1.2209
mimic-cardio-donor_trand_20nan: decoder 0.8491, bar 0.9078 (best baseline by run: ['knn10'])
  mean impute_score over 15 folds: decoder 0.8491, condexp 0.8657, knn10 0.9078, missforest 0.9089, lgbm 0.9122, hgb 0.9127, knn5 0.9430, mean_mode 1.0000, ice 1.0983, softimpute 1.1978
mimic-cardio-donor_mcar_20nan: decoder 0.7831, bar 0.8257 (best baseline by run: ['hgb', 'lgbm'])
  mean impute_score over 15 folds: decoder 0.7831, condexp 0.7930, missforest 0.8004, ice 0.8259, hgb 0.8263, lgbm 0.8288, knn10 0.8771, knn5 0.9171, mean_mode 1.0000, softimpute 1.0124
mimic-sofa-donor_tmatch_20nan: decoder 0.9209, bar 0.9481 (best baseline by run: ['knn10', 'lgbm'])
  mean impute_score over 15 folds: decoder 0.9209, condexp 0.9305, knn10 0.9482, hgb 0.9579, lgbm 0.9629, missforest 0.9822, knn5 0.9943, mean_mode 1.0000, softimpute 1.2129, ice 1.9002
mimic-sofa-donor_trand_20nan: decoder 0.9339, bar 0.9730 (best baseline by run: ['knn10', 'lgbm'])
  mean impute_score over 15 folds: decoder 0.9339, condexp 0.9423, missforest 0.9711, knn10 0.9754, hgb 0.9902, lgbm 0.9908, mean_mode 1.0000, knn5 1.0042, softimpute 1.0711, ice 1.5812
mimic-sofa-donor_mcar_20nan: decoder 0.9237, bar 0.9473 (best baseline by run: ['hgb', 'knn10'])
  mean impute_score over 15 folds: missforest 0.9151, decoder 0.9237, condexp 0.9318, hgb 0.9482, lgbm 0.9521, knn10 0.9546, ice 0.9615, knn5 0.9856, mean_mode 1.0000, softimpute 1.0529
PRIMARY mimic-cardio-donor_tmatch_20nan (n=15): -0.0455 [-0.0604, -0.0307], Holm p 0.0000 -> better
PRIMARY mimic-sofa-donor_tmatch_20nan (n=15): -0.0272 [-0.0423, -0.0122], Holm p 0.0017 -> better
TabPFN 0.9206; mimic-cardio-donor_tmatch_20nan, decoder - TabPFN (n=15): -0.0600 [-0.0698, -0.0503]
  decoder against TabPFN: rmse_num_z 0.8580 against 0.9176, mae_num_z 0.5521 against 0.5776
TabPFN 0.8944; mimic-sofa-donor_tmatch_20nan, decoder - TabPFN (n=15): +0.0265 [+0.0155, +0.0375]
  decoder against TabPFN: rmse_num_z 0.8158 against 0.7936, mae_num_z 0.5484 against 0.5173, acc_cat 0.7386 against 0.7470
mimic-cardio-donor tmatch - trand, decoder's score (n=15): +0.0115 [-0.0083, +0.0314]
mimic-cardio-donor tmatch - trand, gap to the bar (n=15): +0.0132 [-0.0061, +0.0325]
mimic-cardio-donor trand - mcar, decoder's score (n=15): +0.0660 [+0.0465, +0.0854]
mimic-cardio-donor trand - mcar, gap to the bar (n=15): -0.0161 [-0.0374, +0.0052]
mimic-sofa-donor tmatch - trand, decoder's score (n=15): -0.0130 [-0.0286, +0.0027]
mimic-sofa-donor tmatch - trand, gap to the bar (n=15): +0.0119 [-0.0130, +0.0368]
mimic-sofa-donor trand - mcar, decoder's score (n=15): +0.0102 [-0.0020, +0.0224]
mimic-sofa-donor trand - mcar, gap to the bar (n=15): -0.0155 [-0.0401, +0.0092]
```

- **Primary, `tmatch`'s induced cells, decoder minus the bar's best baseline, 15 replicates:**
  - `mimic-cardio-donor`: 0.8606 against 0.9061, −0.0455 [−0.0604, −0.0307], Holm p below 0.0001, **better**.
  - `mimic-sofa-donor`: 0.9209 against 0.9481, −0.0272 [−0.0423, −0.0122], Holm p 0.0017, **better**.

  About 5% and 3% of the bar's score. The bar's best baseline changes with the run here (`hgb`, `knn10` or `lgbm`),
  where E148's was `hgb` in every run.
- **Every imputer the runs scored, levels only, no test.** On `tmatch` the decoder has the lowest mean on both
  tables. The nearest is the strong tier's conditional expectation (`condexp`): 0.8616 against 0.8606 on the cardio
  table, which these levels do not tell apart, and 0.9305 against 0.9209 on the SOFA one. Over the six variants the
  decoder's mean is the lowest on five; on `mimic-sofa-donor_mcar` missForest's is lower (0.9151 against 0.9237).
  ICE and SoftImpute are above the mean/mode fill on every transplanted variant (ICE 1.90 on the SOFA `tmatch`).
- **`trand − mcar`, what the blocks cost, descriptive.** On the cardio table the decoder's score is higher under
  the permuted real patterns than under uniform gaps, +0.0660 [+0.0465, +0.0854], and its gap to the bar does not
  move with it (−0.0161 [−0.0374, +0.0052]): the bar's score rises as much. On the SOFA table both intervals hold
  zero (+0.0102 [−0.0020, +0.0224] and −0.0155 [−0.0401, +0.0092]).
- **`tmatch − trand`, what the dependence on the row costs, descriptive.** Every interval holds zero. The decoder's
  score moves in opposite directions on the two tables (+0.0115 [−0.0083, +0.0314] cardio, −0.0130 [−0.0286,
  +0.0027] SOFA); its gap to the bar, +0.0132 [−0.0061, +0.0325] and +0.0119 [−0.0130, +0.0368]. No reading of a
  mechanism follows from either contrast (the second threat above).
- **TabPFN on `tmatch`, descriptive** (added 2026-11-06 15:44 GMT-3; the block above was printed again with it). The
  offline pass (`tabpfn_run.py`, ADR 0028's settings) ran 2026-11-06 15:07 to 15:43 GMT-3, 30 folds in
  `<state>/tabpfn/`. Its induced cells are the runs': the rebuilt folds score the mean/mode fill as the runs did on
  all 30 folds (120 values, largest difference 2e-16). No test in the rule. The two tables disagree:
  - `mimic-cardio-donor`: the decoder's score is lower, −0.0600 [−0.0698, −0.0503] (TabPFN 0.9206, above the bar's
    0.9061 too); lower in RMSE (0.858 against 0.918) and in MAE (0.552 against 0.578).
  - `mimic-sofa-donor`: TabPFN's score is lower, +0.0265 [+0.0155, +0.0375] (TabPFN 0.8944, the lowest of every
    imputer on this variant); lower in RMSE (0.794 against 0.816) and in MAE (0.517 against 0.548), and its
    categorical accuracy is higher (0.747 against 0.739).

  So the claim of this ticket is against the bar, as registered. Against the strong tier's TabPFN the decoder
  leads on one donor table and trails on the other; on E148's hidden observed cells it led on both.
- **What this cannot say** stands as the threats above: the donors are the sicker patients, and a transplant
  carries where the gaps fall, not what the missing values were. With [E148](03-masked-cells-natural-gaps.md) this
  makes two readings on one cohort where the decoder is better than the bar; neither scores a true gap.
- **Cost, measured, against the estimate of 5 to 15 minutes a cell:** 16 of the 18 cells took 19 to 33 minutes and
  two took 57 and 59 (both from about 12:27 to 13:26 GMT-3; what else loaded the GPU then was not looked into).
  The estimate scaled E148's cells by the rows; the time did not fall with them. 4 h 21 min of wall time for the
  first pass, 4 h 51 min with the relaunched cell.
- No default change and no ADR, as fixed above. One seed set on two views of one cohort.
