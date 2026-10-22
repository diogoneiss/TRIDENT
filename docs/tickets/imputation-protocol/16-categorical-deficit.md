# Where does K lose to TabPFN on kr-vs-kp and credit-g? A per-column read (E61; pre-registered 2026-10-22)

This ticket was written and committed before the study's first cell. One smoke pair ran first,
and its disclosure is under "Before the ticket".

- **Why:** this is T10's first step. The user's go was "vamos no 3 agora, dispare" (2026-10-22,
  about 06:10 GMT-3), for the per-column categorical loss weighting.
  - On 404/505/606, TabPFN beats K on all six kr-vs-kp and credit-g variants from 40nan (E58).
    These are the two most categorical tables.
- **The plain lever has nothing to act on.**
  - Under MCAR every feature column has the same count of observed cells: 1,278 on each of
    kr-vs-kp_60nan's columns and 400 on each of credit-g_60nan's. The dynamic mask therefore hides
    about equal counts per column.
  - So averaging the cross-entropy per column, then across columns, is about the cell-pooled term
    the loss already uses.
  - F-02-3 (the critique) names only constant columns as an artifact. No shipped table has one;
    kr-vs-kp's `spcop` is the nearest, at 99.97% one class.
- **What a weighting could still act on:** columns where K learned only the marginal, while TabPFN
  learned the column's conditional structure. This read finds whether such columns carry K's
  deficit.
- **Offline:** `scripts/experiments/categorical_deficit.py`. Nothing is written to the store or to
  `src/`, and no training runs.

## Design (fixed before the first cell)

- **Cells:** E40's runs of kr-vs-kp and credit-g at 40, 60 and 80nan, seeds 404/505/606. That is 18
  pairs on two GPU lanes, about 30 minutes.
- **Per fold and per column of the ledger's induced cells:**
  - **the model's errors,** from the ledger (the headline K's fills);
  - **TabPFN's errors,** from TabPFN v2 refitted on the same training fold, as E60 did, and asked
    for exactly the ledger's cells;
  - **the naive fill's errors** (mode or mean on the training fold), as `impute_score` uses.
  - A categorical column's error is its count of wrong cells, and a numerical one's its sum of
    squared errors.
  - The categorical columns also get their class count, mode share and entropy, from the training
    fold.
- **Integrity:**
  - the model's fold `impute_score` must equal the run's K;
  - TabPFN's must lie within 2e-3 of E58's CSV;
  - otherwise the pair stops.

## Reading, fixed now (`categorical_deficit.py --tally`)

- **Per variant:**
  - **K − TabPFN split into its categorical and numerical terms.** The split is exact, since
    `impute_score` is the cell-weighted sum of the two ratios.
    - A variant is **counted** if K is behind and the categorical term carries at least half of
      the gap.
  - **Pooled over its 15 folds, per categorical column:** K's gain over the mode, TabPFN's, and the
    excess errors (K's minus TabPFN's).
  - **A column is "stuck at the marginal"** if K's gain over the mode is at most 5% while TabPFN's
    is at least 20%.
  - **Descriptive only:**
    - the share of the net excess held by the top 20% of columns, beside the same share of
      TabPFN's own errors (what a proportional deficit would give);
    - the correlation of K's and TabPFN's per-column gains.
- **Primary, over the counted variants (at least 3 are needed):**
  - **a target:** stuck columns hold 50% or more of the net excess on 3 or more;
  - **no target:** on 1 or none;
  - **mixed:** otherwise.
- **What each leads to:**
  - **a target:** a flag-gated weighting aimed at those columns, its form chosen from their
    properties. It needs its own pre-registered study, with the old behaviour as the default and an
    MLflow tag.
  - **no target:** K's deficit is spread over columns it already learns. No column weighting can
    move it, and T10's weighting half closes.
  - **mixed:** reported, and the user chooses.

## Before the ticket (disclosed)

- **One smoke pair, credit-g_40nan at s404, ran at 06:19 GMT-3** to test the script. Both
  integrity checks passed, with TabPFN within 6.6e-6 of E58.
- **The thresholds (5%, 20%, 50%) were in the script before it ran.**
- **The tally's code paths were tested on that pair copied to three seeds.** It printed one
  variant's line, with no stuck column, so I have seen that line.
  - The study reruns that pair, and a TabPFN refit with the same seed gives the same fills.

## Outcome (2026-10-22, 06:40 GMT-3)

**The run:** 18 of 18 pairs, 06:20–06:32 GMT-3, with none failed.

- On every fold, the model's `impute_score` from the ledger equals the run's K.
- TabPFN's lies within 1.0e-3 of E58's (the largest gap, at credit-g_60nan s505), under the 2e-3
  tolerance.

**The printout** (`scripts/experiments/categorical_deficit.py --tally`, unedited):

```text
## `kr-vs-kp_40nan`: K − TabPFN +0.0110 = categorical +0.0110 + numerical +0.0000
  - net excess errors 282 over 36 columns; the top 8 hold 89% (TabPFN's own errors, top 8: 47%)
  - stuck at the marginal (model gain ≤ 5%, TabPFN gain ≥ 20%): 0 columns, 0% of the net excess
  | column | cells | K errors | TabPFN errors | mode errors | K gain | TabPFN gain | classes | mode share |
  |---|---|---|---|---|---|---|---|---|
  | rkxwp | 3834 | 458 | 420 | 735 | +38% | +43% | 2 | 0.79 |
  | wtoeg | 3834 | 881 | 843 | 984 | +10% | +14% | 2 | 0.76 |
  | dwipd | 3834 | 1067 | 1031 | 1155 | +8% | +11% | 2 | 0.69 |
  | dsopp | 3834 | 421 | 392 | 417 | -1% | +6% | 2 | 0.90 |
  | r2ar8 | 3834 | 723 | 694 | 1170 | +38% | +41% | 2 | 0.68 |
  | wkpos | 3834 | 658 | 629 | 984 | +33% | +36% | 2 | 0.73 |
  | katri | 3834 | 768 | 740 | 807 | +5% | +8% | 3 | 0.79 |
  | bxqsq | 3834 | 975 | 950 | 1167 | +16% | +19% | 2 | 0.70 |
  - correlation of K's and TabPFN's per-column gains: +0.94

## `kr-vs-kp_60nan`: K − TabPFN +0.0434 = categorical +0.0434 + numerical +0.0000
  - net excess errors 1691 over 36 columns; the top 8 hold 53% (TabPFN's own errors, top 8: 45%)
  - stuck at the marginal (model gain ≤ 5%, TabPFN gain ≥ 20%): 0 columns, 0% of the net excess
  | column | cells | K errors | TabPFN errors | mode errors | K gain | TabPFN gain | classes | mode share |
  |---|---|---|---|---|---|---|---|---|
  | bkxbq | 5754 | 2574 | 2401 | 2631 | +2% | +9% | 2 | 0.53 |
  | simpl | 5754 | 2151 | 2025 | 2166 | +1% | +7% | 2 | 0.61 |
  | bkspr | 5754 | 1785 | 1663 | 1926 | +7% | +14% | 2 | 0.67 |
  | wknck | 5754 | 2084 | 1973 | 2151 | +3% | +8% | 2 | 0.61 |
  | bkxwp | 5754 | 949 | 853 | 1242 | +24% | +31% | 2 | 0.78 |
  | dwipd | 5754 | 1771 | 1678 | 1770 | -0% | +5% | 2 | 0.69 |
  | wkpos | 5754 | 1326 | 1241 | 1587 | +16% | +22% | 2 | 0.75 |
  | katri | 5754 | 1323 | 1241 | 1233 | -7% | -1% | 3 | 0.80 |
  - correlation of K's and TabPFN's per-column gains: +0.94

## `kr-vs-kp_80nan`: K − TabPFN +0.0365 = categorical +0.0365 + numerical +0.0000
  - net excess errors 1885 over 36 columns; the top 8 hold 73% (TabPFN's own errors, top 8: 44%)
  - stuck at the marginal (model gain ≤ 5%, TabPFN gain ≥ 20%): 0 columns, 0% of the net excess
  | column | cells | K errors | TabPFN errors | mode errors | K gain | TabPFN gain | classes | mode share |
  |---|---|---|---|---|---|---|---|---|
  | blxwp | 7671 | 2547 | 2264 | 2940 | +13% | +23% | 2 | 0.63 |
  | cntxt | 7671 | 2813 | 2537 | 3357 | +16% | +24% | 2 | 0.60 |
  | wkpos | 7671 | 2060 | 1886 | 1977 | -4% | +5% | 2 | 0.70 |
  | wknck | 7671 | 3026 | 2884 | 2892 | -5% | +0% | 2 | 0.61 |
  | bkxcr | 7671 | 2559 | 2421 | 2811 | +9% | +14% | 2 | 0.64 |
  | wkovl | 7671 | 2632 | 2494 | 2808 | +6% | +11% | 2 | 0.61 |
  | r2ar8 | 7671 | 2258 | 2140 | 2373 | +5% | +10% | 2 | 0.67 |
  | bkxwp | 7671 | 1603 | 1489 | 1677 | +4% | +11% | 2 | 0.79 |
  - correlation of K's and TabPFN's per-column gains: +0.76

## `credit-g_40nan`: K − TabPFN +0.0219 = categorical +0.0155 + numerical +0.0065
  - net excess errors 156 over 13 columns; the top 3 hold 71% (TabPFN's own errors, top 3: 39%)
  - stuck at the marginal (model gain ≤ 5%, TabPFN gain ≥ 20%): 0 columns, 0% of the net excess
  | column | cells | K errors | TabPFN errors | mode errors | K gain | TabPFN gain | classes | mode share |
  |---|---|---|---|---|---|---|---|---|
  | credit_history | 1200 | 484 | 437 | 531 | +9% | +18% | 5 | 0.51 |
  | property_magnitude | 1200 | 691 | 656 | 837 | +17% | +22% | 4 | 0.35 |
  | housing | 1200 | 286 | 257 | 333 | +14% | +23% | 3 | 0.70 |
  | job | 1200 | 445 | 419 | 426 | -4% | +2% | 4 | 0.62 |
  | employment | 1200 | 742 | 719 | 813 | +9% | +12% | 5 | 0.35 |
  | personal_status | 1200 | 514 | 492 | 549 | +6% | +10% | 4 | 0.55 |
  | savings_status | 1200 | 516 | 510 | 510 | -1% | +0% | 5 | 0.62 |
  | other_payment_plans | 1200 | 230 | 225 | 225 | -2% | +0% | 3 | 0.82 |
  - correlation of K's and TabPFN's per-column gains: +0.90

## `credit-g_60nan`: K − TabPFN +0.0391 = categorical +0.0324 + numerical +0.0067
  - net excess errors 474 over 13 columns; the top 3 hold 55% (TabPFN's own errors, top 3: 38%)
  - stuck at the marginal (model gain ≤ 5%, TabPFN gain ≥ 20%): 0 columns, 0% of the net excess
  | column | cells | K errors | TabPFN errors | mode errors | K gain | TabPFN gain | classes | mode share |
  |---|---|---|---|---|---|---|---|---|
  | property_magnitude | 1800 | 1167 | 1045 | 1203 | +3% | +13% | 4 | 0.33 |
  | employment | 1800 | 1190 | 1097 | 1197 | +1% | +8% | 5 | 0.35 |
  | own_telephone | 1800 | 670 | 624 | 705 | +5% | +11% | 2 | 0.58 |
  | housing | 1800 | 502 | 459 | 504 | +0% | +9% | 3 | 0.70 |
  | credit_history | 1800 | 788 | 747 | 840 | +6% | +11% | 5 | 0.53 |
  | personal_status | 1800 | 818 | 777 | 801 | -2% | +3% | 4 | 0.54 |
  | job | 1800 | 659 | 631 | 639 | -3% | +1% | 4 | 0.60 |
  | purpose | 1800 | 1294 | 1270 | 1278 | -1% | +1% | 10 | 0.26 |
  - correlation of K's and TabPFN's per-column gains: +0.81

## `credit-g_80nan`: K − TabPFN +0.0139 = categorical +0.0113 + numerical +0.0025
  - net excess errors 225 over 13 columns; the top 3 hold 66% (TabPFN's own errors, top 3: 38%)
  - stuck at the marginal (model gain ≤ 5%, TabPFN gain ≥ 20%): 0 columns, 0% of the net excess
  | column | cells | K errors | TabPFN errors | mode errors | K gain | TabPFN gain | classes | mode share |
  |---|---|---|---|---|---|---|---|---|
  | employment | 2400 | 1640 | 1578 | 1626 | -1% | +3% | 5 | 0.33 |
  | purpose | 2400 | 1792 | 1746 | 1733 | -3% | -1% | 10 | 0.27 |
  | own_telephone | 2400 | 1006 | 965 | 996 | -1% | +3% | 2 | 0.64 |
  | checking_status | 2400 | 1510 | 1473 | 1473 | -3% | +0% | 4 | 0.43 |
  | property_magnitude | 2400 | 1622 | 1599 | 1600 | -1% | +0% | 4 | 0.33 |
  | credit_history | 2400 | 1132 | 1122 | 1119 | -1% | -0% | 5 | 0.52 |
  | personal_status | 2400 | 1099 | 1089 | 1098 | -0% | +1% | 4 | 0.57 |
  | housing | 2400 | 699 | 699 | 699 | +0% | +0% | 3 | 0.72 |
  - correlation of K's and TabPFN's per-column gains: +0.09

Primary: stuck columns hold ≥ 50% of the net excess on 0 of 6 counted variants (none) -> no target
```

**Verdict: no target.** All six variants are counted: K is behind TabPFN on each, and the
categorical term carries 71–100% of the gap. No column on any of them is stuck at the marginal, so
stuck columns hold 0% of the excess on 0 of 6.

**Reading:**

- **K learns every column TabPFN learns, a few points less.**
  - At 40 and 60nan, K's gain over the mode trails TabPFN's by about 3 to 10 points, column by
    column.
  - The columns where K gains most are the ones where TabPFN gains most. The per-column gains
    correlate at +0.76 to +0.94.
  - **An example, kr-vs-kp_40nan:** K +38% against TabPFN +43% on `rkxwp`, and +38% against +41%
    on `r2ar8`.
- **The excess is more concentrated than TabPFN's own errors (descriptive).** The top 20% of
  columns hold 53–89% of it, against 38–47% of TabPFN's errors.
  - Those columns are ones K has learned (`rkxwp`, `r2ar8`, `wkpos`, `blxwp`, `cntxt`), not ones
    it missed.
  - A weighting moves the gradient between columns. It cannot make the model learn each column's
    conditional structure better.
- **credit-g_80nan:** both methods are at the marginal, with per-column gains between −3% and +3%
  and a gain correlation of +0.09.
  - K is a little below the mode on most columns there.
  - That is what ADR 0027's calibration addresses: KC falls back to the mode where the model's
    category does worse on the validation gaps. This read is on K.
- **Limit:** a single seed set (404/505/606), on E40's runs of the default as it was then. The
  categorical path has not changed since.

**What it leads to (fixed in the ticket):**

- **T10's weighting half closes.** K's deficit to TabPFN on these tables is spread over columns it
  already learns, so no per-column weighting has a target.
- T10's other half, an unbiased draw of the evaluation mask, stays in the backlog.
- No default changes, and nothing in `src/` was touched.
