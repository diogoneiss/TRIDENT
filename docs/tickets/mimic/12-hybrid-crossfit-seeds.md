# The cross-fit side pick on the MIMIC-IV transplant variants (E171, T55; pre-registered 2026-11-09 21:18 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E168's cross-fit pick ([imputation-protocol/63](../imputation-protocol/63-crossfit-side-pick.md)) chooses
  TabPFN or missForest per fold by each side's held-out validation score. It was supported on E167's 12 OpenML variants.
  On the MIMIC variants E166 ([11](11-hybrid-transplant-seeds.md)) printed it, descriptively, worse than the accepted
  TabPFN-side hybrid (`table_tabpfn`, ADR 0031 decision 6, extended to these variants on 2026-11-09) on the SOFA
  `tmatch` (+0.0054 [+0.0007, +0.0102]). This study asks whether that loss is real before the cross-fit pick goes near
  clinical tables.
- **The user's go:** "Os dois em sequência" (2026-11-09, after 20:47 GMT-3), to use gorgona8's idle GPU.
- **Disclosure:** E166's printout (seeds 4545/5656/6767) is on the record, the loss above among it; E163's (42/7/13)
  predates the cross-fit code and has never printed it. Seeds 101/202/303 have no run on these variants.

## Design

- **Variants:** E149's six MIMIC-IV transplant variants at 20nan (cardio and SOFA; `tmatch`, `trand`, `mcar`), the two
  `tmatch` as primaries, as E163.
- **Cells:** 6 × 101/202/303 = 18 runs of H under `experiment = mimic-hybrid-crossfit-2026-11-09`, state
  `/scratch/diogoneiss/trident-2026-11-09-mimic-crossfit` on gorgona8, runner
  `scripts/experiments/mimic_hybrid.py --study e171`. Offline per pair: TabPFN refit (H's ledger asserted equal to the
  run's H), then missForest refit (asserted equal to the run's logged missForest), into `cells-mf`.
- **Machine (ADR 0035):** everything on gorgona8, H, TabPFN and missForest alike. Its trainings log to medusa1's store
  through the tunnel (ADR 0033); the cells read gorgona8's store copy, so a lane starts a pair only 15 minutes after its
  run is done. Three GPU queues, then two offline lanes taking each pair once its run is in. E172
  ([imputation-protocol/66](../imputation-protocol/66-crossfit-off-mcar-seeds.md)) follows in the same queues.
- **Cost:** about 1–2 h for the 18 runs and their cells.

## Measures and rule (fixed now)

Fold-pair verdict on 15 (seed, fold) differences, 1% SESOI of the opponent's mean.
- **Primary, `crossfit` on the two `tmatch` variants:**
  - **supported** if worse than H on neither and worse than `table_tabpfn` on neither;
  - **refuted** if worse than H on either, or worse than `table_tabpfn` on both;
  - **mixed** otherwise (worse than `table_tabpfn` on one).
- **Secondary:** the same two counts on all six; E163's rule on `table_tabpfn`, descriptive here (a third seed set).
- Three seeds, fixed; no extension.
- **Read** once with all 18 in, on gorgona8:
  `TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data uv run --python 3.11 python -B scripts/experiments/mimic_hybrid.py --study e171 --tally`.
- **What it leads to:** with E169 and E172, whether the cross-fit pick goes to the user as an ADR 0031 amendment, and
  whether that amendment may cover clinical tables.
