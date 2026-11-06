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
- **Cost:** not estimated; from E148's first cells (ticket 01's item (c)), recorded here before the first cell.
- **What follows:** no default change and no ADR either way.
- **Read only with all 18 runs done.**
