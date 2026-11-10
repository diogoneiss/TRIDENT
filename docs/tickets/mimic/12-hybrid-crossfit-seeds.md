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

## Outcome (read once 2026-11-10 04:00 GMT-3: mixed; written 2026-11-10 04:02 GMT-3)

Read on gorgona8 at commit cfc6470, after all 18 pairs' cells were in (no FAILED line; the H-ledger and missForest
asserts held on every pair).

- **E171 primary, mixed.** The cross-fit pick is worse than H on neither `tmatch` (cardio −0.0055, SOFA −0.0088, both no
  call). It is worse than the accepted TabPFN-side hybrid (`table_tabpfn`) on the SOFA `tmatch`, +0.0091
  [+0.0013, +0.0169], and equivalent to it on the cardio `tmatch` (−0.0007). E166's descriptive loss (+0.0054) recurs,
  larger. On the SOFA `tmatch` the cross-fit pick took missForest on 7 of 15 folds, where missForest alone scores 0.9744
  against TabPFN's 0.8923.
- **Secondary, all six:** `crossfit` worse than H on none; worse than `table_tabpfn` on the SOFA `tmatch` and the SOFA
  `trand` (+0.0103 [+0.0004, +0.0202]), equivalent on the other four.
- **E163's rule on `table_tabpfn`, descriptive (a third seed set):** better than H on the SOFA `tmatch` (−0.0179), no call
  on the cardio `tmatch` (−0.0049), worse than TabPFN on neither: supported again. Off the primaries it is worse than
  TabPFN on the SOFA `trand` (+0.0074, as on both earlier seed sets) and the SOFA `mcar` (+0.0103).
- **What it leads to:** on these clinical tables the cross-fit pick loses to the accepted hybrid it would replace on
  SOFA, so an ADR 0031 amendment for it, if the user wants one, should not cover clinical tables. ADR 0031 decision 6's
  extension to these variants stands; this printout is a third seed set in its favour on both `tmatch`.

The printout, unedited:

```
- `mimic-cardio-donor_tmatch_20nan` (primary): H 0.8610, tabpfn 0.9197, missforest 0.8954, table_tabpfn 0.8561, table_missforest 0.8525, pick 0.8562, picked_missforest 0.1333, crossfit 0.8555, crossfit_missforest 0.1333
  - table_missforest − H -0.0085 [-0.0127, -0.0043] better
  - table_missforest − missforest -0.0429 [-0.0512, -0.0346] better
  - table_tabpfn − H -0.0049 [-0.0109, +0.0012] none
  - table_tabpfn − tabpfn -0.0636 [-0.0767, -0.0505] better
  - pick − H -0.0048 [-0.0108, +0.0012] none
  - pick − table_tabpfn +0.0000 [-0.0012, +0.0013] equivalent
  - pick − table_missforest +0.0037 [-0.0042, +0.0116] none
  - crossfit − H -0.0055 [-0.0118, +0.0007] none
  - crossfit − table_tabpfn -0.0007 [-0.0016, +0.0003] equivalent
  - the pick took missForest on 2 of 15 folds, the cross-fit pick on 2
- `mimic-cardio-donor_trand_20nan`: H 0.8522, tabpfn 0.9009, missforest 0.9079, table_tabpfn 0.8426, table_missforest 0.8487, pick 0.8430, picked_missforest 0.4000, crossfit 0.8429, crossfit_missforest 0.4667
  - table_missforest − H -0.0035 [-0.0078, +0.0009] equivalent
  - table_missforest − missforest -0.0592 [-0.0784, -0.0401] better
  - table_tabpfn − H -0.0096 [-0.0186, -0.0006] better
  - table_tabpfn − tabpfn -0.0583 [-0.0692, -0.0473] better
  - pick − H -0.0092 [-0.0176, -0.0008] better
  - pick − table_tabpfn +0.0004 [-0.0040, +0.0047] equivalent
  - pick − table_missforest -0.0057 [-0.0123, +0.0008] none
  - crossfit − H -0.0093 [-0.0177, -0.0009] better
  - crossfit − table_tabpfn +0.0003 [-0.0040, +0.0047] equivalent
  - the pick took missForest on 6 of 15 folds, the cross-fit pick on 7
- `mimic-cardio-donor_mcar_20nan`: H 0.7988, tabpfn 0.7691, missforest 0.7985, table_tabpfn 0.7708, table_missforest 0.7819, pick 0.7717, picked_missforest 0.2000, crossfit 0.7737, crossfit_missforest 0.3333
  - table_missforest − H -0.0169 [-0.0243, -0.0095] better
  - table_missforest − missforest -0.0166 [-0.0243, -0.0090] better
  - table_tabpfn − H -0.0280 [-0.0376, -0.0184] better
  - table_tabpfn − tabpfn +0.0017 [-0.0032, +0.0066] equivalent
  - pick − H -0.0271 [-0.0375, -0.0168] better
  - pick − table_tabpfn +0.0009 [-0.0013, +0.0030] equivalent
  - pick − table_missforest -0.0102 [-0.0162, -0.0043] better
  - crossfit − H -0.0251 [-0.0354, -0.0149] better
  - crossfit − table_tabpfn +0.0028 [-0.0013, +0.0070] equivalent
  - the pick took missForest on 3 of 15 folds, the cross-fit pick on 5
- `mimic-sofa-donor_tmatch_20nan` (primary): H 0.9147, tabpfn 0.8923, missforest 0.9744, table_tabpfn 0.8967, table_missforest 0.9137, pick 0.9006, picked_missforest 0.2000, crossfit 0.9059, crossfit_missforest 0.4667
  - table_missforest − H -0.0009 [-0.0064, +0.0046] equivalent
  - table_missforest − missforest -0.0607 [-0.0784, -0.0430] better
  - table_tabpfn − H -0.0179 [-0.0231, -0.0127] better
  - table_tabpfn − tabpfn +0.0044 [-0.0022, +0.0111] none
  - pick − H -0.0140 [-0.0228, -0.0052] better
  - pick − table_tabpfn +0.0039 [-0.0021, +0.0099] none
  - pick − table_missforest -0.0131 [-0.0209, -0.0053] better
  - crossfit − H -0.0088 [-0.0168, -0.0007] none
  - crossfit − table_tabpfn +0.0091 [+0.0013, +0.0169] worse
  - the pick took missForest on 3 of 15 folds, the cross-fit pick on 7
- `mimic-sofa-donor_trand_20nan`: H 0.9275, tabpfn 0.8994, missforest 0.9708, table_tabpfn 0.9068, table_missforest 0.9288, pick 0.9176, picked_missforest 0.3333, crossfit 0.9171, crossfit_missforest 0.3333
  - table_missforest − H +0.0013 [-0.0094, +0.0121] none
  - table_missforest − missforest -0.0420 [-0.0485, -0.0356] better
  - table_tabpfn − H -0.0207 [-0.0275, -0.0139] better
  - table_tabpfn − tabpfn +0.0074 [+0.0016, +0.0132] worse
  - pick − H -0.0099 [-0.0243, +0.0046] none
  - pick − table_tabpfn +0.0108 [+0.0007, +0.0210] none
  - pick − table_missforest -0.0112 [-0.0186, -0.0038] better
  - crossfit − H -0.0104 [-0.0242, +0.0034] none
  - crossfit − table_tabpfn +0.0103 [+0.0004, +0.0202] worse
  - the pick took missForest on 5 of 15 folds, the cross-fit pick on 5
- `mimic-sofa-donor_mcar_20nan`: H 0.9206, tabpfn 0.8940, missforest 0.9213, table_tabpfn 0.9043, table_missforest 0.9150, pick 0.9059, picked_missforest 0.3333, crossfit 0.9049, crossfit_missforest 0.2667
  - table_missforest − H -0.0056 [-0.0099, -0.0012] better
  - table_missforest − missforest -0.0063 [-0.0164, +0.0038] none
  - table_tabpfn − H -0.0163 [-0.0210, -0.0115] better
  - table_tabpfn − tabpfn +0.0103 [+0.0038, +0.0169] worse
  - pick − H -0.0147 [-0.0201, -0.0092] better
  - pick − table_tabpfn +0.0016 [-0.0011, +0.0043] equivalent
  - pick − table_missforest -0.0091 [-0.0155, -0.0027] better
  - crossfit − H -0.0157 [-0.0205, -0.0108] better
  - crossfit − table_tabpfn +0.0006 [-0.0012, +0.0024] equivalent
  - the pick took missForest on 5 of 15 folds, the cross-fit pick on 4

E163 rule (descriptive in E171) `table` (TabPFN side) on mimic-cardio-donor_tmatch_20nan, mimic-sofa-donor_tmatch_20nan: better than H on ['mimic-sofa-donor_tmatch_20nan'], worse than H on none, worse than TabPFN on none -> supported
E171 PRIMARY `crossfit` on mimic-cardio-donor_tmatch_20nan, mimic-sofa-donor_tmatch_20nan: worse than H on none, worse than `table_tabpfn` on ['mimic-sofa-donor_tmatch_20nan'] -> mixed
E171 SECONDARY on all six: `crossfit` worse than H on none, worse than `table_tabpfn` on ['mimic-sofa-donor_tmatch_20nan', 'mimic-sofa-donor_trand_20nan']
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```
