# T29 · Why the boosters win on electricity

Written 2026-10-20, 23:12 GMT-3, on the user's go for T29 (2026-10-20). It is read-only. Every
number below comes from `scripts/experiments/electricity_report.py`, whose printout is at the end.
The one test this report leads to is E54 ([imputation-protocol/09](../tickets/imputation-protocol/09-electricity.md)),
pre-registered before any of its subsets was computed.

## In five lines

- **The limit is against the two boosters (hgb, lgbm), not the strong tier.**
  - Against the bar of 5, today's K is worse at 20nan (+2.3%, E51 on fresh seeds) and at 60nan
    (+1.4%, E44).
  - It shows no difference at 40nan (+6.0%, a wide interval) and at 80nan.
  - Against missForest it is no different at 20 and 40nan, and 9.8% and 10.5% better at 60 and
    80nan.
- **E44's "loss" at 40nan is D4's.** K is worse than the tuned bar there (+0.0496), while against
  the untuned bar it shows no difference.
- **The model trails the boosters on almost every column.** The gap is widest on `day` (+0.107 of
  the mean/mode ratio at 20nan) and `nswdemand` (+0.068), then `transfer`, `vicdemand` and
  `date`. The ranking holds in MAE.
  - It wins only on `vicprice` and `nswprice`, a heavy-tailed pair. The `vicprice` win shrinks
    from −0.649 in RMSE to −0.100 in MAE.
- **The table has two structures a tree reads directly:**
  - `date` is a 933-level key: 48 rows per date, one per half-hour, and `day` is constant within
    a date on 922 of 933.
  - The first 17,424 rows (38.5%, date ≤ 0.4423) hold `vicprice`, `vicdemand` and `transfer` at
    one constant each, the source's fill before Victoria's data begins.
- **The leading hypothesis:** the boosters look values up by `date`, from the same day's other
  rows in the training fold. The model reads `date` as one scalar through a small MLP, which
  cannot resolve 933 keys. E54 tests it by splitting K − bar by whether a cell's row has its
  `date`.

## Reading the per-column tables

- **`day`** is a 7-way category. Within a date it never changes (bar 11 dates where two days
  round to one date value). A tree that splits finely on `date` recovers it from the date's other
  training rows. The boosters reach 0.705 at 20nan; the model 0.812. The gap closes with
  missingness: +0.045 at 40nan, +0.020 at 60nan, +0.002 at 80nan, as fewer rows keep their date.
- **`nswdemand`** follows the day and the half-hour. The boosters' lead is +0.068 at 20nan, falling
  to +0.011 at 80nan.
- **The constant regime is not where the lead is.**
  - The boosters lead on `vicdemand` and `transfer` by only 0.03 at 20nan.
  - On `vicprice` the model wins, because the boosters' RMSE ratios there exceed 1 (1.38 and 1.67
    at 20nan) under the column's tail: sd 0.010, maximum 1.0.
  - E54 reports the regime split as a secondary measure, without a rule.
- **The strong tier at 40–80nan is missForest alone in this printout.** E44's fold tables lack
  E41's linear arms, which live in offline CSVs. E44's own report, with them, has K − strong
  better at 60 and 80nan.

## Printout

```text
## The table
- 45312 rows, 8 features; the constant regime: 17424 rows (38.5%), rows 0–17423, date 0.0000–0.4423; none after row 17423
  - vicprice: constant 0.003467 on the regime; elsewhere sd 0.0130, and 0.01% equal to it
  - vicdemand: constant 0.422915 on the regime; elsewhere sd 0.1542, and 0.00% equal to it
  - transfer: constant 0.414912 on the regime; elsewhere sd 0.1754, and 0.20% equal to it

## electricity_20nan (default-refresh-2026-10-20, seeds 1001/2002/3003)
- K 0.6130 − bar (hgb/lgbm): +0.0138 (+2.3% of it) [+0.0023, +0.0254] worse
- K 0.6130 − strong (missforest): +0.0003 (+0.0% of it) [-0.0089, +0.0095] none

Per column (electricity_20nan; ratio to the mean/mode fill, mean over 15 folds; lower is better):

| column | share | model | hgb | lgbm | missforest | condexp | softimpute | ice | knn10 | model − best booster | MAE model − best booster |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `date` | 0.125 | 0.547 | 0.530 | 0.528 | 0.558 | 0.887 | 0.941 | 0.889 | 0.720 | 0.018 | 0.024 |
| `day` | 0.125 | 0.812 | 0.706 | 0.705 | 0.711 | 1.000 | 1.000 | 0.907 | 0.935 | 0.107 | nan |
| `nswdemand` | 0.125 | 0.470 | 0.401 | 0.402 | 0.476 | 0.708 | 0.760 | 0.707 | 0.841 | 0.068 | 0.069 |
| `nswprice` | 0.125 | 0.578 | 0.601 | 0.599 | 0.590 | 0.919 | 1.091 | 0.909 | 0.908 | -0.021 | 0.038 |
| `period` | 0.125 | 0.809 | 0.804 | 0.803 | 0.830 | 0.909 | 1.021 | 0.925 | 1.019 | 0.005 | -0.003 |
| `transfer` | 0.125 | 0.494 | 0.458 | 0.458 | 0.498 | 0.761 | 0.828 | 0.762 | 0.761 | 0.036 | 0.028 |
| `vicdemand` | 0.125 | 0.388 | 0.355 | 0.354 | 0.405 | 0.664 | 0.691 | 0.658 | 0.729 | 0.034 | 0.016 |
| `vicprice` | 0.125 | 0.727 | 1.376 | 1.674 | 1.111 | 0.816 | 1.430 | 1.079 | 1.080 | -0.649 | -0.100 |

## electricity_40nan (confirmation-2026-10-17, seeds 707/808/909)
- K 0.7745 − bar (hgb/lgbm): +0.0436 (+6.0% of it) [-0.0013, +0.0885] none
- K 0.7745 − strong (missforest): -0.0164 (-2.1% of it) [-0.0614, +0.0286] none

Per column (electricity_40nan; ratio to the mean/mode fill, mean over 15 folds; lower is better):

| column | share | model | hgb | lgbm | missforest | condexp | softimpute | ice | knn10 | model − best booster | MAE model − best booster |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `date` | 0.125 | 0.612 | 0.599 | 0.596 | 0.704 | nan | nan | nan | 0.847 | 0.015 | 0.022 |
| `day` | 0.125 | 0.867 | 0.821 | 0.824 | 0.866 | nan | nan | nan | 0.961 | 0.045 | nan |
| `nswdemand` | 0.125 | 0.585 | 0.546 | 0.544 | 0.660 | nan | nan | nan | 0.927 | 0.041 | 0.042 |
| `nswprice` | 0.125 | 0.711 | 0.741 | 0.722 | 0.757 | nan | nan | nan | 1.003 | -0.012 | 0.009 |
| `period` | 0.125 | 0.871 | 0.872 | 0.871 | 0.958 | nan | nan | nan | 1.042 | 0.001 | -0.003 |
| `transfer` | 0.125 | 0.601 | 0.572 | 0.570 | 0.681 | nan | nan | nan | 0.893 | 0.030 | 0.018 |
| `vicdemand` | 0.125 | 0.500 | 0.494 | 0.491 | 0.589 | nan | nan | nan | 0.892 | 0.009 | -0.023 |
| `vicprice` | 0.125 | 1.513 | 1.334 | 1.565 | 1.135 | nan | nan | nan | 1.065 | 0.179 | -0.008 |

## electricity_60nan (confirmation-2026-10-17, seeds 707/808/909)
- K 0.8298 − bar (hgb/lgbm): +0.0112 (+1.4% of it) [+0.0059, +0.0164] worse
- K 0.8298 − strong (missforest): -0.0902 (-9.8% of it) [-0.1056, -0.0748] better

Per column (electricity_60nan; ratio to the mean/mode fill, mean over 15 folds; lower is better):

| column | share | model | hgb | lgbm | missforest | condexp | softimpute | ice | knn10 | model − best booster | MAE model − best booster |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `date` | 0.125 | 0.702 | 0.693 | 0.688 | 0.868 | nan | nan | nan | 0.846 | 0.013 | 0.023 |
| `day` | 0.125 | 0.921 | 0.901 | 0.902 | 0.937 | nan | nan | nan | 0.942 | 0.020 | nan |
| `nswdemand` | 0.125 | 0.715 | 0.691 | 0.690 | 0.829 | nan | nan | nan | 0.920 | 0.025 | 0.029 |
| `nswprice` | 0.125 | 0.833 | 0.830 | 0.825 | 0.903 | nan | nan | nan | 1.000 | 0.008 | -0.002 |
| `period` | 0.125 | 0.926 | 0.930 | 0.929 | 1.078 | nan | nan | nan | 1.037 | -0.003 | -0.003 |
| `transfer` | 0.125 | 0.723 | 0.697 | 0.697 | 0.873 | nan | nan | nan | 0.894 | 0.027 | 0.015 |
| `vicdemand` | 0.125 | 0.658 | 0.653 | 0.652 | 0.776 | nan | nan | nan | 0.890 | 0.006 | -0.033 |
| `vicprice` | 0.125 | 0.956 | 0.997 | 1.043 | 1.060 | nan | nan | nan | 1.090 | -0.041 | 0.029 |

## electricity_80nan (confirmation-2026-10-17, seeds 707/808/909)
- K 0.9002 − bar (hgb/lgbm): +0.0004 (+0.0% of it) [-0.0093, +0.0100] none
- K 0.9002 − strong (missforest): -0.1058 (-10.5% of it) [-0.1184, -0.0931] better

Per column (electricity_80nan; ratio to the mean/mode fill, mean over 15 folds; lower is better):

| column | share | model | hgb | lgbm | missforest | condexp | softimpute | ice | knn10 | model − best booster | MAE model − best booster |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `date` | 0.125 | 0.838 | 0.836 | 0.834 | 1.005 | nan | nan | nan | 0.890 | 0.004 | 0.013 |
| `day` | 0.125 | 0.962 | 0.961 | 0.959 | 0.973 | nan | nan | nan | 0.981 | 0.002 | nan |
| `nswdemand` | 0.125 | 0.855 | 0.846 | 0.844 | 0.930 | nan | nan | nan | 0.929 | 0.011 | 0.013 |
| `nswprice` | 0.125 | 0.894 | 0.922 | 0.920 | 0.982 | nan | nan | nan | 0.985 | -0.026 | -0.002 |
| `period` | 0.125 | 0.969 | 0.983 | 0.982 | 1.198 | nan | nan | nan | 1.022 | -0.013 | -0.006 |
| `transfer` | 0.125 | 0.865 | 0.858 | 0.855 | 0.985 | nan | nan | nan | 0.923 | 0.010 | 0.009 |
| `vicdemand` | 0.125 | 0.836 | 0.838 | 0.837 | 0.904 | nan | nan | nan | 0.921 | -0.000 | -0.020 |
| `vicprice` | 0.125 | 1.231 | 1.076 | 1.231 | 1.360 | nan | nan | nan | 1.225 | 0.156 | -0.029 |
```
