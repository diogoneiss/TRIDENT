# ADR 0030's binned interval under real gap patterns (E153; pre-registered 2026-11-06)

This ticket was written and committed before its reader was run on any ledger.

- **Why:** [ADR 0030](../../adr/0030-an-opt-in-per-cell-uncertainty-signal-from-re-masking.md) decision 8 (amended
  2026-11-06 from [E151](../imputation-protocol/54-conformal-by-row-gaps.md)) makes the fill's 90% interval
  split-conformal per column and per tertile of the row's gap count, because one width per column under-covers the
  rows that miss the most. E151 read that on injected MCAR gaps (and, descriptively, E84's MAR and self-masking MNAR).
  [E149](04-pattern-transplant.md)'s ledgers carry gap patterns copied from real ICU stays onto complete rows, with
  the truth kept: blocks of gaps that fall on particular patients. Does decision 8 hold there?
- **The user's go:** "vamos no 1, faça pull antes" (2026-11-06), on the proposal to read decision 8's interval on
  E149's ledgers, offline.
- **Offline, no training:** E149's 18 runs, 90 fold ledgers in `/scratch2/diogoneiss/trident-2026-11-06-mimic-e149`,
  nothing in `src/` or the store. Reader: `scripts/experiments/conformal_real_patterns.py`.
- **Disclosure:** E149's scores and E151's results are known. Before this ticket, one fold ledger per donor table
  (`tmatch`, seed 42, fold 1) was opened to see what it holds: its `induced` rows (every transplanted gap, with the
  truth), no `spread` (E149 ran without `--score_remask`), and the count of gaps per row and per column (SOFA: 59% of
  rows with one gap, the numerical gaps mostly bilirubin, PaO2/FiO2 and urine output; cardio: one to five gaps a row,
  mostly troponin and the arterial pressures). No coverage or width has been computed on any of them.

## Design (fixed now)

- **Cells:** the `induced` numerical cells of each fold ledger (the transplanted gaps; categoricals left out, as
  E151), the fill the ledger holds for them (the run's checkpoint, arm `D`), and b, the count of `induced` cells in
  the cell's row, itself included.
- **Variants:** per donor table (`mimic-cardio-donor`, `mimic-sofa-donor`), `tmatch` (each row takes its nearest
  incomplete stay's pattern), `trand` (the same patterns permuted among rows) and `mcar` (the same count per column at
  uniform positions); seeds 42/7/13, five folds.
- **Two arms, E151's code unchanged** (`conformal_row_gaps.fold_arms`, cross-fitted on the same seed's other four
  folds, the split-conformal level):
  - **C**, per column, one width;
  - **MC**, per column and per tertile of b cut on the fit cells; coinciding edges merge bins; a column-and-bin group
    with fewer than 30 fit cells falls back to its column's width, and the share of such cells is printed.
  - E151's spread arms (S, MS) need `spread`, which these ledgers lack, so Q2 is not read.
- **Measure (E151's Q1):** per fold, u, the worst bin's under-coverage of 0.90 floored at 0; the fold difference
  u(MC) − u(C), lower better; SESOI 0.01. Per variant, `report_common.verdict` over the 15 fold differences, every
  seed agreeing: better, no difference, equivalent or worse.

## Reading, fixed now (on `tmatch`, the two donor tables)

- **supported:** better on both tables;
- **mixed:** better on one, worse on neither;
- **refuted:** better on neither, or worse on either.

**Printed beside, decide nothing:** u(C) alone and each arm's overall, bottom-bin and top-bin coverage; the width
(MC − C) / C with its verdict; the fallback share; `trand` and `mcar` by the same measure (do the blocks, or their
dependence on the row, change what one width per column misses?).

**What each reading leads to:**
- **supported:** decision 8's scope is proposed to include real gap patterns transplanted from MIMIC-IV stays, as an
  ADR 0030 amendment for the user's decision.
- **refuted with u(C) near 0 on `tmatch`:** one width per column already covers every tertile on these tables;
  decision 8 is proposed to state it.
- **refuted with MC worse:** decision 8 is proposed to be limited to injected gaps, for the user's decision.
- **mixed:** reported as such, no amendment proposed.

**Limits, stated now:** the calibration is the oracle one of E151 (other test folds' gaps). The donors are the
patients with everything measured, so a transplanted pattern is at most missing at random given the matching
columns (E149's threat). The rates are near 20nan (10.5% and 12.9% of feature cells), with few gaps in most columns,
so many column-and-bin groups may fall back.
