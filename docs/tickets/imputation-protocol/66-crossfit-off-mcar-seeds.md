# The cross-fit side pick on E93's off-MCAR variants, fresh seeds (E172, T54; pre-registered 2026-11-09 21:18 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** ADR 0031 decision 6 stays MCAR-only for synthetic missingness because E93 found the accepted TabPFN-side
  hybrid (`table_tabpfn`) worse than H on `spambase_probit_60nan` (+0.0090), repeated in E167 (+0.0082). E170
  ([65](65-crossfit-off-mcar.md)) asks whether the cross-fit pick removes that, but only on 6 variants, where one loss
  reads mixed. This study takes all 12 of E93's off-MCAR variants on a seed set no hybrid study used, so the scope
  question has its own test.
- **The user's go:** "Os dois em sequência" (2026-11-09, after 20:47 GMT-3), to use gorgona8's idle GPU.
- **Disclosure:** the cross-fit pick against H on these 12 is known on two seed sets: E168's development printout
  (4545/5656/6767) has it worse on `spambase_probit_60nan` (+0.0062); E167 (42/7/13, 6 of the 12) on none. E169's 6
  are unread at this commit. Seeds 101/202/303 ran on some of these variants in the gap-reliance, occlusion and
  spread studies of 2026-10-27/28 (no hybrid); none is read here.

## Design

- **Variants (12), `hybrid_fresh_seeds.E93`:** self-masking MNAR `biodeg_probit0p25_60nan`, `biodeg_probit_60nan`,
  `biodeg_probit1p0_60nan`, `biodeg_probit2p0_60nan`, `vehicle_probit_60nan`, `kc2_probit_60nan`,
  `spambase_probit_60nan`; MAR `biodeg_marf_60nan`, `kc2_marf_60nan`, `vehicle_marf_60nan`, `spambase_marf_60nan`,
  `credit-g_marf_60nan`.
- **Cells:** 12 × 101/202/303 = 36 runs of H under `experiment = hybrid-crossfit-offmcar-2026-11-09`, state
  `/scratch/diogoneiss/trident-2026-11-09-hybrid-crossfit-offmcar` on gorgona8, runner
  `scripts/experiments/hybrid_fresh_seeds.py --study e172`. Offline per pair as E167: TabPFN refit (H's ledger asserted),
  then missForest refit (logged score asserted), into `cells-mf`.
- **Machine (ADR 0035):** everything on gorgona8, after E171 ([mimic/12](../mimic/12-hybrid-crossfit-seeds.md)) in the
  same three GPU queues; two offline lanes, a pair 15 minutes after its run is done (the store copy's lag).
- **Cost:** the two spambase variants dominate (TabPFN on spambase took up to 1h50 a pair on gorgona8 on 2026-11-09);
  about 6–9 h in all, so E171 and E172 end about 06:00–09:00 GMT-3.

## Measures and rule (fixed now)

Fold-pair verdict on 15 (seed, fold) differences, 1% SESOI of the opponent's mean.
- **Primary, `crossfit` against H on the 12:** **supported** if worse on none; **refuted** if worse on 2 or more;
  **mixed** if worse on 1.
- **Secondary:** `crossfit` better than H on which; `table_tabpfn` worse than H on which (E93's finding, a third seed
  set); `crossfit` against `table_tabpfn`, worse and better.
- Three seeds, fixed; no extension.
- **Read** once with all 36 in, on gorgona8: `uv run --python 3.11 python -B scripts/experiments/hybrid_fresh_seeds.py --study e172 --tally`.
- **What it leads to:** with E169 and E170, whether an ADR 0031 amendment for the cross-fit pick can drop the MCAR-only
  scope for synthetic MAR and MNAR. The user's call.

## Outcome (read once 2026-11-10 04:01 GMT-3: mixed; written 2026-11-10 04:02 GMT-3)

Read on gorgona8 at commit cfc6470, after all 36 pairs' cells were in (no FAILED line; the H-ledger and missForest
asserts held on every pair).

- **Primary, mixed.** The cross-fit pick is worse than H on one of 12, `biodeg_probit2p0_60nan` (+0.0066
  [+0.0043, +0.0090]). There it took missForest on 0 of 15 folds and equals `table_tabpfn`, which is worse than H by the
  same amount; the missForest-side hybrid is worse than H too (+0.0011 [+0.0002, +0.0021]). Both sides lose to H on
  that variant, the steepest self-masking one, so no side choice avoids the loss. E168's development printout had no
  call there (+0.0052).
- **`spambase_probit_60nan`:** `table_tabpfn` is worse than H (+0.0067 [+0.0032, +0.0101]), E93's finding on a third
  seed set; the cross-fit pick is equivalent to H (+0.0007) and better than `table_tabpfn` (−0.0059).
- **Secondary:** `crossfit` better than H on 7 (`biodeg_probit0p25`, `biodeg_probit`, `vehicle_probit`, and the marf
  variants of biodeg, vehicle, spambase and credit-g, all `_60nan`); `table_tabpfn` worse than H on 2
  (`biodeg_probit2p0_60nan`, `spambase_probit_60nan`); `crossfit` worse than `table_tabpfn` on none and better on 4.
- **What it leads to:** the cross-fit pick removes the accepted hybrid's spambase_probit loss but not a loss both sides
  share under strong self-masking. Whether ADR 0031's MCAR-only scope moves for synthetic MAR and MNAR stays the user's
  call, with E169 and E170 still to read.

The printout, unedited:

```
- `biodeg_probit0p25_60nan` (trails): H 0.9106, tabpfn 0.9152, missforest 0.8881, table_tabpfn 0.9064, table_missforest 0.8851, pick 0.8935, picked_missforest 0.7333, crossfit 0.8946, crossfit_missforest 0.6667
  - table_missforest − H -0.0255 [-0.0310, -0.0200] better
  - table_missforest − missforest -0.0030 [-0.0076, +0.0017] equivalent
  - table_tabpfn − H -0.0042 [-0.0127, +0.0042] none
  - table_tabpfn − tabpfn -0.0089 [-0.0134, -0.0043] better
  - pick − H -0.0171 [-0.0284, -0.0059] better
  - pick − table_tabpfn -0.0129 [-0.0176, -0.0082] better
  - pick − table_missforest +0.0083 [-0.0003, +0.0169] none
  - crossfit − H -0.0160 [-0.0265, -0.0055] better
  - crossfit − table_tabpfn -0.0118 [-0.0168, -0.0068] better
  - crossfit − pick +0.0011 [-0.0013, +0.0035] equivalent
- `biodeg_probit_60nan` (trails): H 0.9178, tabpfn 0.9257, missforest 0.8975, table_tabpfn 0.9170, table_missforest 0.8972, pick 0.9031, picked_missforest 0.6667, crossfit 0.9038, crossfit_missforest 0.6667
  - table_missforest − H -0.0207 [-0.0250, -0.0163] better
  - table_missforest − missforest -0.0003 [-0.0033, +0.0027] equivalent
  - table_tabpfn − H -0.0008 [-0.0079, +0.0062] equivalent
  - table_tabpfn − tabpfn -0.0087 [-0.0127, -0.0047] better
  - pick − H -0.0147 [-0.0236, -0.0059] better
  - pick − table_tabpfn -0.0139 [-0.0212, -0.0067] better
  - pick − table_missforest +0.0059 [+0.0003, +0.0115] none
  - crossfit − H -0.0140 [-0.0230, -0.0051] better
  - crossfit − table_tabpfn -0.0132 [-0.0202, -0.0062] better
  - crossfit − pick +0.0007 [-0.0036, +0.0050] equivalent
- `biodeg_probit1p0_60nan`: H 0.9305, tabpfn 0.9375, missforest 0.9361, table_tabpfn 0.9298, table_missforest 0.9269, pick 0.9288, picked_missforest 0.2667, crossfit 0.9285, crossfit_missforest 0.3333
  - table_missforest − H -0.0035 [-0.0064, -0.0007] better
  - table_missforest − missforest -0.0091 [-0.0121, -0.0062] better
  - table_tabpfn − H -0.0007 [-0.0032, +0.0018] equivalent
  - table_tabpfn − tabpfn -0.0077 [-0.0102, -0.0052] better
  - pick − H -0.0016 [-0.0045, +0.0013] equivalent
  - pick − table_tabpfn -0.0009 [-0.0023, +0.0004] equivalent
  - pick − table_missforest +0.0019 [+0.0007, +0.0031] worse
  - crossfit − H -0.0020 [-0.0049, +0.0010] equivalent
  - crossfit − table_tabpfn -0.0013 [-0.0028, +0.0002] equivalent
  - crossfit − pick -0.0004 [-0.0011, +0.0004] equivalent
- `biodeg_probit2p0_60nan`: H 0.9311, tabpfn 0.9543, missforest 0.9515, table_tabpfn 0.9377, table_missforest 0.9322, pick 0.9378, picked_missforest 0.0667, crossfit 0.9377, crossfit_missforest 0.0000
  - table_missforest − H +0.0011 [+0.0002, +0.0021] worse
  - table_missforest − missforest -0.0193 [-0.0218, -0.0168] better
  - table_tabpfn − H +0.0066 [+0.0043, +0.0090] worse
  - table_tabpfn − tabpfn -0.0166 [-0.0193, -0.0138] better
  - pick − H +0.0067 [+0.0044, +0.0090] worse
  - pick − table_tabpfn +0.0000 [-0.0000, +0.0001] equivalent
  - pick − table_missforest +0.0056 [+0.0032, +0.0079] worse
  - crossfit − H +0.0066 [+0.0043, +0.0090] worse
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick -0.0000 [-0.0001, +0.0000] equivalent
- `vehicle_probit_60nan` (trails): H 0.6284, tabpfn 0.6210, missforest 0.6192, table_tabpfn 0.6124, table_missforest 0.6081, pick 0.6080, picked_missforest 0.6667, crossfit 0.6080, crossfit_missforest 0.6667
  - table_missforest − H -0.0203 [-0.0241, -0.0164] better
  - table_missforest − missforest -0.0111 [-0.0166, -0.0057] better
  - table_tabpfn − H -0.0160 [-0.0203, -0.0116] better
  - table_tabpfn − tabpfn -0.0086 [-0.0134, -0.0038] better
  - pick − H -0.0204 [-0.0244, -0.0164] better
  - pick − table_tabpfn -0.0044 [-0.0085, -0.0004] better
  - pick − table_missforest -0.0001 [-0.0020, +0.0018] equivalent
  - crossfit − H -0.0204 [-0.0244, -0.0164] better
  - crossfit − table_tabpfn -0.0044 [-0.0085, -0.0004] better
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `kc2_probit_60nan`: H 0.7143, tabpfn 0.7719, missforest 0.7382, table_tabpfn 0.7248, table_missforest 0.7170, pick 0.7251, picked_missforest 0.2667, crossfit 0.7257, crossfit_missforest 0.2000
  - table_missforest − H +0.0027 [-0.0163, +0.0217] none
  - table_missforest − missforest -0.0212 [-0.0362, -0.0061] better
  - table_tabpfn − H +0.0105 [-0.0065, +0.0276] none
  - table_tabpfn − tabpfn -0.0470 [-0.0628, -0.0313] better
  - pick − H +0.0108 [-0.0091, +0.0307] none
  - pick − table_tabpfn +0.0003 [-0.0055, +0.0060] equivalent
  - pick − table_missforest +0.0081 [-0.0025, +0.0186] none
  - crossfit − H +0.0114 [-0.0082, +0.0310] none
  - crossfit − table_tabpfn +0.0009 [-0.0047, +0.0064] equivalent
  - crossfit − pick +0.0006 [-0.0007, +0.0019] equivalent
- `spambase_probit_60nan`: H 0.9589, tabpfn 0.9954, missforest 0.9605, table_tabpfn 0.9656, table_missforest 0.9530, pick 0.9613, picked_missforest 0.4667, crossfit 0.9596, crossfit_missforest 0.5333
  - table_missforest − H -0.0059 [-0.0083, -0.0036] better
  - table_missforest − missforest -0.0075 [-0.0104, -0.0046] better
  - table_tabpfn − H +0.0067 [+0.0032, +0.0101] worse
  - table_tabpfn − tabpfn -0.0298 [-0.0357, -0.0239] better
  - pick − H +0.0024 [-0.0028, +0.0076] equivalent
  - pick − table_tabpfn -0.0042 [-0.0070, -0.0014] better
  - pick − table_missforest +0.0084 [+0.0022, +0.0145] worse
  - crossfit − H +0.0007 [-0.0038, +0.0053] equivalent
  - crossfit − table_tabpfn -0.0059 [-0.0099, -0.0019] better
  - crossfit − pick -0.0017 [-0.0053, +0.0019] equivalent
- `biodeg_marf_60nan`: H 0.6838, tabpfn 0.7002, missforest 0.7161, table_tabpfn 0.6614, table_missforest 0.6658, pick 0.6615, picked_missforest 0.0667, crossfit 0.6620, crossfit_missforest 0.3333
  - table_missforest − H -0.0180 [-0.0230, -0.0131] better
  - table_missforest − missforest -0.0503 [-0.0602, -0.0403] better
  - table_tabpfn − H -0.0224 [-0.0289, -0.0160] better
  - table_tabpfn − tabpfn -0.0389 [-0.0522, -0.0255] better
  - pick − H -0.0223 [-0.0287, -0.0159] better
  - pick − table_tabpfn +0.0002 [-0.0002, +0.0005] equivalent
  - pick − table_missforest -0.0043 [-0.0080, -0.0005] better
  - crossfit − H -0.0218 [-0.0284, -0.0152] better
  - crossfit − table_tabpfn +0.0006 [-0.0007, +0.0020] equivalent
  - crossfit − pick +0.0005 [-0.0009, +0.0018] equivalent
- `kc2_marf_60nan`: H 0.5776, tabpfn 0.6521, missforest 0.7046, table_tabpfn 0.5829, table_missforest 0.6068, pick 0.5960, picked_missforest 0.4000, crossfit 0.5950, crossfit_missforest 0.3333
  - table_missforest − H +0.0293 [+0.0033, +0.0552] worse
  - table_missforest − missforest -0.0978 [-0.1380, -0.0575] better
  - table_tabpfn − H +0.0053 [-0.0137, +0.0243] none
  - table_tabpfn − tabpfn -0.0692 [-0.1027, -0.0357] better
  - pick − H +0.0184 [-0.0030, +0.0398] none
  - pick − table_tabpfn +0.0131 [-0.0012, +0.0275] none
  - pick − table_missforest -0.0108 [-0.0219, +0.0002] none
  - crossfit − H +0.0175 [-0.0013, +0.0363] none
  - crossfit − table_tabpfn +0.0122 [-0.0013, +0.0256] none
  - crossfit − pick -0.0009 [-0.0091, +0.0072] none
- `vehicle_marf_60nan`: H 0.4558, tabpfn 0.4540, missforest 0.4943, table_tabpfn 0.4393, table_missforest 0.4406, pick 0.4379, picked_missforest 0.5333, crossfit 0.4397, crossfit_missforest 0.5333
  - table_missforest − H -0.0152 [-0.0215, -0.0088] better
  - table_missforest − missforest -0.0537 [-0.0698, -0.0376] better
  - table_tabpfn − H -0.0165 [-0.0227, -0.0104] better
  - table_tabpfn − tabpfn -0.0147 [-0.0483, +0.0188] none
  - pick − H -0.0179 [-0.0244, -0.0113] better
  - pick − table_tabpfn -0.0013 [-0.0073, +0.0046] none
  - pick − table_missforest -0.0027 [-0.0065, +0.0011] none
  - crossfit − H -0.0161 [-0.0232, -0.0090] better
  - crossfit − table_tabpfn +0.0004 [-0.0051, +0.0060] none
  - crossfit − pick +0.0018 [-0.0010, +0.0045] none
- `spambase_marf_60nan`: H 0.9018, tabpfn 0.9349, missforest 0.9627, table_tabpfn 0.8950, table_missforest 0.8970, pick 0.8956, picked_missforest 0.2000, crossfit 0.8955, crossfit_missforest 0.3333
  - table_missforest − H -0.0049 [-0.0073, -0.0024] better
  - table_missforest − missforest -0.0657 [-0.0754, -0.0560] better
  - table_tabpfn − H -0.0069 [-0.0098, -0.0040] better
  - table_tabpfn − tabpfn -0.0400 [-0.0493, -0.0306] better
  - pick − H -0.0062 [-0.0089, -0.0035] better
  - pick − table_tabpfn +0.0007 [-0.0002, +0.0016] equivalent
  - pick − table_missforest -0.0013 [-0.0028, +0.0002] equivalent
  - crossfit − H -0.0063 [-0.0091, -0.0035] better
  - crossfit − table_tabpfn +0.0006 [-0.0003, +0.0015] equivalent
  - crossfit − pick -0.0001 [-0.0003, +0.0001] equivalent
- `credit-g_marf_60nan`: H 0.9420, tabpfn 0.9087, missforest 0.9781, table_tabpfn 0.9149, table_missforest 0.9404, pick 0.9149, picked_missforest 0.0000, crossfit 0.9296, crossfit_missforest 0.4667
  - table_missforest − H -0.0016 [-0.0029, -0.0003] better
  - table_missforest − missforest -0.0377 [-0.0576, -0.0177] better
  - table_tabpfn − H -0.0271 [-0.0429, -0.0113] better
  - table_tabpfn − tabpfn +0.0061 [+0.0000, +0.0122] none
  - pick − H -0.0271 [-0.0429, -0.0113] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0256 [-0.0408, -0.0103] better
  - crossfit − H -0.0124 [-0.0210, -0.0039] better
  - crossfit − table_tabpfn +0.0147 [-0.0013, +0.0307] none
  - crossfit − pick +0.0147 [-0.0013, +0.0307] none

E172 PRIMARY `crossfit` on E93's 12 off-MCAR variants: worse than H on ['biodeg_probit2p0_60nan'] -> mixed
E172 SECONDARY `crossfit` better than H on ['biodeg_probit0p25_60nan', 'biodeg_probit_60nan', 'vehicle_probit_60nan', 'biodeg_marf_60nan', 'vehicle_marf_60nan', 'spambase_marf_60nan', 'credit-g_marf_60nan']; `table_tabpfn` worse than H on ['biodeg_probit2p0_60nan', 'spambase_probit_60nan']; `crossfit` worse than `table_tabpfn` on none, better on ['biodeg_probit0p25_60nan', 'biodeg_probit_60nan', 'vehicle_probit_60nan', 'spambase_probit_60nan']
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```
