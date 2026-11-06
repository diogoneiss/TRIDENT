# ADR 0030's binned interval on the natural gaps' rows (E154; pre-registered 2026-11-06)

This ticket was written and committed before its reader was run on any ledger.

- **Why:** [ADR 0030](../../adr/0030-an-opt-in-per-cell-uncertainty-signal-from-re-masking.md) decision 8 calibrates
  the fill's 90% interval per column and per tertile of the row's gaps. [E153](07-conformal-real-patterns.md) found
  no gain on E149's transplanted patterns, where a row has one to five gaps. [E148](03-masked-cells-natural-gaps.md)'s
  ledgers hold the whole cohort, 11,082 rows whose natural gaps reach most of a row's sparse columns, with observed
  cells hidden and their truth kept. That is the spread of row gaps where E151 found one width per column failing.
- **The user's go:** "ok va" (2026-11-06), on the proposal that put this reading first.
- **Offline, no training:** E148's 6 runs, 30 fold ledgers and their partitions in
  `/scratch2/diogoneiss/trident-2026-11-06-mimic-e148`, the natural tables in `/scratch/diogoneiss/trident-data`.
  Reader: `scripts/experiments/conformal_natural_gaps.py`.
- **Disclosure:** E148's Outcome is known, including the decoder's error by the row's count of natural gaps: it is
  highest in the rows with none (RMSE 1.15 z on cardio, 1.05 on SOFA) and lower in the rows with more. So if one
  width per column fails a tertile here, it may be the bottom one, not the top. E153's and E151's results are known.
  No coverage or width has been computed on these ledgers.

## Design (fixed now)

- **Cells:** each fold ledger's `masked` numerical cells (observed cells hidden for scoring, drawn uniformly among
  measured values), with the fill the ledger holds (the run's checkpoint, `--decode_checkpoint masked`, arm `M`).
- **A cell's row gaps:** b = the row's natural gaps in the table file plus the row's `masked` cells in that fold's
  ledger, the cell itself included: every gap the model saw in the row. The ledger numbers a row by its position among
  the fold's test rows, mapped to the file through the fold's partition, as `mimic_report.by_row_gaps` does.
- **Tables:** `mimic-cardio_natural`, `mimic-sofa_natural`; seeds 42/7/13, five folds.
- **Arms and measure:** E153's, unchanged: C and MC from `conformal_row_gaps.fold_arms`, cross-fitted on the same
  seed's other four folds; Q1, u(MC) − u(C) per fold, u the worst bin's under-coverage of 0.90 (whichever bin it
  is), SESOI 0.01, `report_common.verdict` over 15 fold differences with seed agreement. No `spread`, so no Q2.

## Reading, fixed now (the two tables)

- **supported:** better on both;
- **mixed:** better on one, worse on neither;
- **refuted:** better on neither, or worse on either.

**Printed beside, decide nothing:** u(C), each arm's overall, bottom-bin and top-bin coverage, the worst bin's
position under C; the width (MC − C) / C with its verdict; the fallback share.

**What each reading leads to:**
- **supported:** decision 8's scope is proposed to include MIMIC-IV's natural gaps (on hidden observed cells), as an
  ADR 0030 amendment for the user's decision, with the bin C fails named.
- **refuted with u(C) near 0:** with E153, decision 8 is proposed to state that on MIMIC-IV one width per column
  already covers, for the user's decision.
- **refuted with MC worse:** decision 8 is proposed to be limited to injected gaps, for the user's decision.
- **mixed:** reported as such, no amendment proposed.

**Limits, stated now:** the oracle calibration of E151 (other test folds' cells). The hidden cells are measured
values, so the interval's coverage on a true natural gap is not read (E148's own limit: an unmeasured lactate is not
a measured one).
