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
