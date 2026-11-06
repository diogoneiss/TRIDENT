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

## Outcome (2026-11-06 17:13 GMT-3): refuted, one width per column already covers

The reader ran once, at 17:12 GMT-3, on all 90 ledgers (no ledger missing; fallback shares 0.01 to 0.07). The
self-check's line is the first.

**By the rule, refuted:** on `tmatch` MC is equivalent to C on both tables (cardio −0.0016 [−0.0075, +0.0044], SOFA
+0.0009 [−0.0052, +0.0070]), better on neither, worse on neither.

- **Which branch:** "refuted with u(C) near 0". One width per column covers the top tertile of the row's gaps on both
  tables (0.900 cardio, 0.909 SOFA); its worst bin is the bottom one (0.894, 0.889), u(C) 0.027 and 0.018. That is
  the range E151 read as "C already covers" on biodeg, kc2, spambase and credit-g (u(C) 0.006 to 0.037). MC is not
  worse, so the "MC worse" branch does not apply.
- **The width:** MC costs 0.6% (SOFA) to 3.0% (cardio) more, every verdict worse beyond the SESOI; no coverage is
  bought with it here.
- **`trand` and `mcar`, descriptive:** the same, no variant better or worse; the blocks and their dependence on the
  row do not change it.
- **A reading, not tested:** at these rates a row has one to five gaps (59% of SOFA rows one), so the tertiles
  separate rows that differ little; E151's under-coverage appeared on large tables at 40 to 80nan.
- **Leads to, as fixed:** decision 8 is proposed to state that on MIMIC-IV's transplanted patterns, near 20nan, one
  width per column already covers every tertile, for the user's decision. ADR 0030 is not edited.

**The printout** (`scripts/experiments/conformal_real_patterns.py`, unedited):

```text
check passed: {'u_C': 0.17, 'top_C': 0.73, 'width_C': 6.872, 'u_S': 0.007, 'top_S': 0.9, 'width_S': 6.001, 'u_MC': 0.008, 'top_MC': 0.899, 'width_MC': 6.783, 'u_MS': 0.008, 'top_MS': 0.899, 'width_MS': 5.998}

## tmatch (primary)

- `mimic-cardio-donor_tmatch_20nan`: Q1 u(MC) − u(C) -0.0016 [-0.0075, +0.0044] equivalent, u(C) 0.027, u(MC) 0.026; width (MC − C) / C +0.030 [+0.023, +0.037] worse; coverage C 0.899 (low bin 0.894, top 0.900), MC 0.902 (low bin 0.904, top 0.899); fallback 0.05
- `mimic-sofa-donor_tmatch_20nan`: Q1 u(MC) − u(C) +0.0009 [-0.0052, +0.0070] equivalent, u(C) 0.018, u(MC) 0.019; width (MC − C) / C +0.006 [+0.001, +0.010] worse; coverage C 0.900 (low bin 0.889, top 0.909), MC 0.900 (low bin 0.901, top 0.905); fallback 0.01
- Q1: better 0 / no difference 0 / equivalent 2 / worse 0 (fold-pair t, seed agreement)

## trand (descriptive)

- `mimic-cardio-donor_trand_20nan`: Q1 u(MC) − u(C) -0.0044 [-0.0112, +0.0023] none, u(C) 0.024, u(MC) 0.019; width (MC − C) / C +0.017 [+0.013, +0.022] worse; coverage C 0.900 (low bin 0.912, top 0.893), MC 0.902 (low bin 0.902, top 0.895); fallback 0.04
- `mimic-sofa-donor_trand_20nan`: Q1 u(MC) − u(C) -0.0015 [-0.0056, +0.0026] equivalent, u(C) 0.022, u(MC) 0.020; width (MC − C) / C +0.014 [+0.011, +0.017] worse; coverage C 0.901 (low bin 0.904, top 0.894), MC 0.903 (low bin 0.902, top 0.901); fallback 0.01
- Q1: better 0 / no difference 1 / equivalent 1 / worse 0 (fold-pair t, seed agreement)

## mcar (descriptive)

- `mimic-cardio-donor_mcar_20nan`: Q1 u(MC) − u(C) -0.0062 [-0.0144, +0.0021] none, u(C) 0.034, u(MC) 0.027; width (MC − C) / C +0.020 [+0.012, +0.027] worse; coverage C 0.901 (low bin 0.906, top 0.888), MC 0.901 (low bin 0.901, top 0.904); fallback 0.06
- `mimic-sofa-donor_mcar_20nan`: Q1 u(MC) − u(C) +0.0053 [-0.0082, +0.0188] none, u(C) 0.022, u(MC) 0.028; width (MC − C) / C +0.008 [+0.002, +0.013] worse; coverage C 0.901 (low bin 0.908, top 0.902), MC 0.901 (low bin 0.902, top 0.894); fallback 0.07
- Q1: better 0 / no difference 2 / worse 0 (fold-pair t, seed agreement)

Primary (E153, Q1 on tmatch): {'mimic-cardio-donor': 'equivalent', 'mimic-sofa-donor': 'equivalent'} -> refuted (fold-pair t, seed agreement)
```
