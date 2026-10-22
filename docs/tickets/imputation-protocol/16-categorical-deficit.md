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
