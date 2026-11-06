# Intervals calibrated per bin of a row's gaps (E151, T43; pre-registered 2026-11-06)

This ticket was written and committed before its reader was run on any ledger.

- **Why:** [ADR 0030](../../adr/0030-an-opt-in-per-cell-uncertainty-signal-from-re-masking.md) keeps the
  re-masking spread as a rank, not an interval. [E77](30-remask-conformal.md) found a per-column constant
  width narrower than the spread-scaled one at the same 90% coverage on 15 of 15 variants. E80
  (imputation-interpretability/03) found why one scale fails: the spread *falls* as a row's gaps grow, while
  the error *rises* when a cell's top sources are missing. Conformal prediction with missing values (Zaffran
  et al., ICML 2023) answers this by calibrating per missingness pattern (Mondrian conformal). Two questions:
  1. **Coverage:** does E77's per-column constant under-cover the rows that miss the most, and does
     calibrating per bin of the row's gaps fix it?
  2. **Width:** once the bin is fixed, does the spread adapt a width better than a constant does? E80's
     confound, spread against the row's gaps, is held fixed inside a bin.
- **The user's go:** "vou na sua sugestão, vá" (2026-11-06, after 12:21 GMT-3), on the proposal that put T43
  first among the lines that need no GPU.
- **Offline, no training:** E75's and E76's fold ledgers (and E84's, descriptive), nothing in `src/` or the
  store. Reader: `scripts/experiments/conformal_row_gaps.py`.
- **Disclosure:** E77's and E80's results are known and motivate the design. No per-bin coverage or width has
  been computed on any ledger.

## Design (fixed now)

- **Cells:**
  - **Primary:** E75's 15 confirmation variants (MCAR, seeds 707/808/909, five folds), numerical gaps only,
    the set E77 read.
  - **Development, same rules, reported beside:** E76's 21 variants at 101/202/303 with numerical gaps
    (kr-vs-kp has none, so 17).
  - **Descriptive:** E84's MNAR and MAR variants at 60nan, 101/202/303.
- **A cell's row gaps:** b = the number of gaps in its row in the fold's ledger (`induced` rows, which hold
  every gap of the test split; the cell itself included), as E80 reads them.
- **Bins:** for held-out fold k of a seed, the tertiles of b over the fit cells (the other four folds of the
  same seed, numerical columns pooled) cut three bins: b ≤ e1, e1 < b ≤ e2, b > e2. Coinciding edges merge
  bins. The **top bin** is the highest one present.
- **Four 90% intervals per numerical cell, cross-fitted as E77** (fit on the same seed's other four folds):
  - **C**, per column, constant: fill ± w, w the conformal quantile of |error| over the column's fit cells;
  - **S**, per column, spread-scaled: fill ± q × spread, q the conformal quantile of |error| / spread (E77's);
  - **MC** and **MS**, the same two fitted on the column's fit cells *in the cell's bin*. A column-and-bin
    group with fewer than 30 fit cells falls back to its column's C or S; the share of such cells is printed.
- **Quantile:** the split-conformal level, the ⌈(n + 1) × 0.9⌉-th smallest of the n fit scores (the largest
  if that exceeds n), for all four arms; E77 used the plain 90% quantile, so C and S are recomputed here.
- **Spread floor:** E77's amendment, per column at the smallest positive spread on the fit cells (1 if none),
  in the column's group and in each bin's.
- **Centres:** K primary, as E77: the ledger's `induced` fill (KC on these calibrated runs) with the spread of
  `induced_remask`. H descriptive: `induced_remask_loss_checkpoint`'s fill and spread.

## Measures

- **Q1, coverage:** per fold, an arm's worst-bin under-coverage u = the largest of 0.90 − coverage over the
  fold's bins, floored at 0. The fold difference is u(MC) − u(C); lower is better. SESOI 0.01, one coverage
  point.
- **Q2, width:** per fold, (mean width of MS − mean width of MC) / mean width of MC; lower is better. SESOI
  0.01 of MC's width, as E77.
- **Per variant:** the project's verdict rule (`report_common.verdict`: the 95% t interval over the 15 fold
  differences, every seed agreeing), better / no difference / equivalent / worse.
- **Printed beside, not ruled on:** each arm's overall coverage and its coverage in each bin; u(C) alone (does
  the per-column constant under-cover the top bin at all?); MS against C, the width of a binned spread
  interval against E77's constant; the fallback share; H.

## Reading, fixed now (on the primary 15, under K)

For each question separately:
- **supported:** better on 10 or more of the 15 and worse on at most 1;
- **refuted:** better on 5 or fewer, or worse on 4 or more;
- **mixed:** otherwise.

The development set is read by the same rule over its 17 and reported as agreeing or not; it does not change
the primary reading.

**What each reading leads to:**
- **Q1 supported:** an interval for the fill, calibrated per bin of the row's gaps, is proposed as an
  amendment to ADR 0030, for the user's decision. It is a conformal wrapper around the fill: it uses the
  spread only if Q2 is supported too.
- **Q1 refuted, with u(C) near 0:** a per-column constant already covers every bin; binning is not needed.
- **Q2 supported:** ADR 0030's "a rank, not an interval" is proposed for amendment: the spread, scaled per
  bin, becomes an interval.
- **Q2 otherwise:** ADR 0030 stays a rank; E77's conclusion extends to binned intervals.

**Limit, stated now:** the calibration cells are other test folds' gaps, whose truth a deployed model would
not have (an oracle reading, as E77). A deployable reading needs observed validation cells hidden on purpose,
which these ledgers do not hold; it would be a follow-up with new runs.
