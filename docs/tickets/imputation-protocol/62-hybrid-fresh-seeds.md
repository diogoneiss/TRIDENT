# E164 and E165 on fresh seeds (E167, T54; pre-registered 2026-11-09 00:57 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E164 ([60](60-missforest-hybrid-generality.md)), the missForest-side `table` hybrid, and E165
  ([61](61-hybrid-side-pick.md)), the per-fold side pick, were both mixed on E90/E91/E93/E95's cell tables (seeds
  4545/5656/6767). The variants that decided them are named below; this study re-runs them on seeds 42/7/13, never used
  by a hybrid study on these variants (E82/E89 used 1212/2323/3434), the E89-to-E90 pattern.
- **The user's go:** "B: E164/E165 confirmatório" (2026-11-09 00:40 GMT-3).
- **Disclosure:** the variants were chosen from E164/E165's printouts, so their dev numbers are known; the fresh seeds are
  the test. Runs of these variants on 42/7/13 exist in other studies (K readouts, no hybrid); none is read here.

## Design

- **Variants (12), named now from E164/E165's printouts:**
  - the 7 where H trails missForest (E164's TRAILS): `biodeg_80nan`, `vehicle_80nan`, `credit-g_40nan`,
    `biodeg_probit0p25_60nan`, `biodeg_probit_60nan`, `vehicle_probit_60nan`, `electricity_20nan`;
  - E165's losses: `spambase_probit_60nan`, `kc2_marf_60nan` (against H), `kc2_probit_60nan` (against `table_tabpfn`);
  - E165's other wins over `table_tabpfn`: `letter_20nan`, `spambase_20nan` (its three probit wins are TRAILS).
- **Cells:** 12 × 42/7/13 = 36 runs of H under `experiment = hybrid-fresh-seeds-2026-11-09`, state
  `/scratch/diogoneiss/trident-2026-11-09-hybrid-fresh-seeds`, runner `scripts/experiments/hybrid_fresh_seeds.py`.
  Imputation defaults with H's flags; `letter_20nan` therefore trains under ADR 0036's P600 and ADR 0018's encoder,
  unlike E95's runs of it (2026-10-29), disclosed. Offline per pair: TabPFN refit (H's ledger asserted equal to the
  run's H), then missForest refit (asserted equal to the run's logged missForest), into `cells-mf`.
- **Machine (ADR 0035):** all on medusa1, four GPU queues (two per GPU, beside the other user's job), longest first;
  offline cells in two lanes, one per GPU, each pair as soon as its run is done.
- **Cost:** about 15 queue-hours (letter ~118 min, electricity ~75–97, spambase ~40–55 a run), so 4–5 h on four
  queues, then E166's 18 MIMIC runs (~40 min). Expected end about 06:00–07:00 GMT-3.

## Measures and rules (fixed now)

Fold-pair verdict on 15 (seed, fold) differences, 1% SESOI of the opponent's mean (E90's rule).
- **E167 primary, `table_missforest`** (E164's hypothesis: no harm):
  - **supported** if worse than H on none of 12 and worse than missForest on at most 1;
  - **refuted** if worse than H on 2 or more, or worse than missForest on 3 or more;
  - **mixed** otherwise.
  - Secondary: better than H on how many of 12, and which of the 7 TRAILS.
- **E167 pick** (E165's rule, unchanged): **supported** if worse than H on none, worse than `table_tabpfn` on at most 1
  and better on at least 2; **refuted** if worse than `table_tabpfn` on 3 or more, or better on none; **mixed** otherwise.
- **Read** once with all 36 in: `uv run --python 3.11 python -B scripts/experiments/hybrid_fresh_seeds.py --tally` on
  medusa1; the same printout carries E168's primary.
- **What it leads to:** the user's calls on extending ADR 0031 decision 6 to missForest and on a side-picking rule.

## Amendment

- **Amendment (2026-11-09 12:00 GMT-3), stalled, nothing read:** all 54 runs trained by 04:07 GMT-3. During the offline pass TabPFN ran out of GPU memory on the spambase pairs at 04:21 GMT-3. Another user's job held 26.4 GB of each card, and TabPFN keeps one model per column. At about 04:48 GMT-3 medusa1's GPU 0 failed ("Unknown Error"), and CUDA has been unusable on both cards since. After that TabPFN fell back to the CPU: `kc2_marf_60nan` s7, `kc2_probit_60nan` s42/s13 and E166's nine cardio cells were computed there, and `electricity_20nan` s7 is uncertain. The pass refused the SOFA tables (over 1000 rows). Pairs without cells: E167's letter_20nan s7/s13 and all six spambase pairs; E166's nine SOFA cells. Recomputing on a GPU, and where, is the user's call.
- **Amendment (2026-11-09 12:23 GMT-3), the offline pass moves to gorgona8 (the user's decision, "rode o tabpfn na gorgona8 mesmo"):** H's runs stay medusa1's. Their selection files and ledgers were copied to the same paths on gorgona8, and the store copy there holds all 54 parents. The TabPFN and missForest refits are redone on gorgona8's GPU and CPU for all 54 pairs, not only the missing ones, so that every pair's other side comes from one machine and one device. The H-ledger and missForest asserts still apply. This departs from ADR 0035, which wants every arm of a pair on one machine: H is from medusa1 and TabPFN from gorgona8, the same split in every pair. medusa1's partial cells are kept under `medusa1-partial/`. Comparing their TabPFN fills with gorgona8's is a device check only, not a read.

## Outcome (read once 2026-11-09 19:01 GMT-3: E167 supported, the pick mixed; written 2026-11-09 19:20 GMT-3)

Read on gorgona8 at commit 14fa1e0, after all 36 pairs' cells were in (gorgona8's offline pass, amendment of 12:23 GMT-3,
no FAILED line, the H-ledger and missForest asserts held on every pair).

- **E167 primary, supported.** The missForest-side hybrid (`table_missforest`) is worse than H on none of 12 and worse
  than missForest on none. It is better than H on 10 of 12, on all 7 TRAILS among them; the two kc2 losses of E165
  (`kc2_marf_60nan`, `kc2_probit_60nan`) read no call. It is better than missForest on 9.
- **E167 pick, mixed.** The in-sample pick is worse than H on `spambase_probit_60nan` (+0.0066 [+0.0022, +0.0110]),
  E165's loss again; against `table_tabpfn` it is better on 6 and worse on none.
- **E168 primary** ([63](63-crossfit-side-pick.md)), in the same printout: **supported**. The cross-fit pick is worse
  than H on none (`spambase_probit_60nan` reads equivalent, +0.0018), better than `table_tabpfn` on 7 and worse on
  none. `crossfit` − `pick` has no better or worse call. Share of folds that took missForest: 54% cross-fit, 47% pick.
- **What it leads to:** extending ADR 0031 decision 6 to missForest, and a side-picking rule, are the user's calls.
  E169 ([64](64-crossfit-generality.md)) tests the cross-fit pick on the other 26 variants.

The printout, unedited:

```
- `biodeg_80nan` (trails): H 0.9775, tabpfn 0.9189, missforest 0.9243, table_tabpfn 0.9141, table_missforest 0.9108, pick 0.9111, picked_missforest 0.3333, crossfit 0.9118, crossfit_missforest 0.2667
  - table_missforest − H -0.0667 [-0.0767, -0.0566] better
  - table_missforest − missforest -0.0135 [-0.0204, -0.0066] better
  - table_tabpfn − H -0.0634 [-0.0706, -0.0562] better
  - table_tabpfn − tabpfn -0.0048 [-0.0077, -0.0018] better
  - pick − H -0.0664 [-0.0752, -0.0575] better
  - pick − table_tabpfn -0.0030 [-0.0069, +0.0010] equivalent
  - pick − table_missforest +0.0003 [-0.0041, +0.0047] equivalent
  - crossfit − H -0.0657 [-0.0743, -0.0572] better
  - crossfit − table_tabpfn -0.0024 [-0.0062, +0.0015] equivalent
  - crossfit − pick +0.0006 [-0.0007, +0.0020] equivalent
- `vehicle_80nan` (trails): H 0.8157, tabpfn 0.7507, missforest 0.7679, table_tabpfn 0.7502, table_missforest 0.7475, pick 0.7468, picked_missforest 0.5333, crossfit 0.7468, crossfit_missforest 0.5333
  - table_missforest − H -0.0681 [-0.0790, -0.0572] better
  - table_missforest − missforest -0.0203 [-0.0278, -0.0129] better
  - table_tabpfn − H -0.0655 [-0.0720, -0.0589] better
  - table_tabpfn − tabpfn -0.0006 [-0.0044, +0.0033] equivalent
  - pick − H -0.0689 [-0.0771, -0.0606] better
  - pick − table_tabpfn -0.0034 [-0.0078, +0.0010] none
  - pick − table_missforest -0.0007 [-0.0063, +0.0049] equivalent
  - crossfit − H -0.0689 [-0.0771, -0.0606] better
  - crossfit − table_tabpfn -0.0034 [-0.0078, +0.0010] none
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `credit-g_40nan` (trails): H 0.9301, tabpfn 0.9006, missforest 0.9306, table_tabpfn 0.9037, table_missforest 0.9201, pick 0.9037, picked_missforest 0.0000, crossfit 0.9107, crossfit_missforest 0.2667
  - table_missforest − H -0.0100 [-0.0181, -0.0019] better
  - table_missforest − missforest -0.0104 [-0.0201, -0.0007] better
  - table_tabpfn − H -0.0265 [-0.0398, -0.0132] better
  - table_tabpfn − tabpfn +0.0031 [-0.0032, +0.0093] none
  - pick − H -0.0265 [-0.0398, -0.0132] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0165 [-0.0270, -0.0060] better
  - crossfit − H -0.0194 [-0.0310, -0.0078] better
  - crossfit − table_tabpfn +0.0071 [-0.0028, +0.0169] none
  - crossfit − pick +0.0071 [-0.0028, +0.0169] none
- `biodeg_probit0p25_60nan` (trails): H 0.9066, tabpfn 0.9135, missforest 0.8832, table_tabpfn 0.9030, table_missforest 0.8805, pick 0.8850, picked_missforest 0.8667, crossfit 0.8880, crossfit_missforest 0.7333
  - table_missforest − H -0.0261 [-0.0325, -0.0197] better
  - table_missforest − missforest -0.0027 [-0.0063, +0.0008] equivalent
  - table_tabpfn − H -0.0035 [-0.0095, +0.0024] none
  - table_tabpfn − tabpfn -0.0105 [-0.0144, -0.0065] better
  - pick − H -0.0216 [-0.0321, -0.0111] better
  - pick − table_tabpfn -0.0180 [-0.0234, -0.0127] better
  - pick − table_missforest +0.0045 [-0.0029, +0.0119] none
  - crossfit − H -0.0186 [-0.0294, -0.0078] better
  - crossfit − table_tabpfn -0.0150 [-0.0213, -0.0088] better
  - crossfit − pick +0.0030 [-0.0014, +0.0074] equivalent
- `biodeg_probit_60nan` (trails): H 0.9185, tabpfn 0.9254, missforest 0.8934, table_tabpfn 0.9148, table_missforest 0.8955, pick 0.9047, picked_missforest 0.6000, crossfit 0.9032, crossfit_missforest 0.6667
  - table_missforest − H -0.0231 [-0.0309, -0.0152] better
  - table_missforest − missforest +0.0021 [-0.0015, +0.0057] equivalent
  - table_tabpfn − H -0.0037 [-0.0075, +0.0001] equivalent
  - table_tabpfn − tabpfn -0.0105 [-0.0145, -0.0066] better
  - pick − H -0.0138 [-0.0206, -0.0071] better
  - pick − table_tabpfn -0.0101 [-0.0159, -0.0043] better
  - pick − table_missforest +0.0092 [+0.0017, +0.0168] worse
  - crossfit − H -0.0153 [-0.0219, -0.0087] better
  - crossfit − table_tabpfn -0.0116 [-0.0174, -0.0058] better
  - crossfit − pick -0.0015 [-0.0046, +0.0017] equivalent
- `vehicle_probit_60nan` (trails): H 0.6232, tabpfn 0.6184, missforest 0.6188, table_tabpfn 0.6088, table_missforest 0.6044, pick 0.6056, picked_missforest 0.5333, crossfit 0.6063, crossfit_missforest 0.4667
  - table_missforest − H -0.0188 [-0.0226, -0.0150] better
  - table_missforest − missforest -0.0143 [-0.0188, -0.0099] better
  - table_tabpfn − H -0.0144 [-0.0173, -0.0115] better
  - table_tabpfn − tabpfn -0.0096 [-0.0138, -0.0054] better
  - pick − H -0.0177 [-0.0212, -0.0142] better
  - pick − table_tabpfn -0.0033 [-0.0054, -0.0012] better
  - pick − table_missforest +0.0011 [-0.0017, +0.0039] equivalent
  - crossfit − H -0.0169 [-0.0199, -0.0139] better
  - crossfit − table_tabpfn -0.0025 [-0.0042, -0.0008] better
  - crossfit − pick +0.0008 [-0.0009, +0.0024] equivalent
- `electricity_20nan` (trails): H 0.6126, tabpfn 0.6263, missforest 0.6194, table_tabpfn 0.6044, table_missforest 0.5910, pick 0.5938, picked_missforest 0.7333, crossfit 0.5933, crossfit_missforest 0.8000
  - table_missforest − H -0.0216 [-0.0275, -0.0156] better
  - table_missforest − missforest -0.0284 [-0.0398, -0.0170] better
  - table_tabpfn − H -0.0082 [-0.0151, -0.0013] better
  - table_tabpfn − tabpfn -0.0218 [-0.0354, -0.0082] better
  - pick − H -0.0189 [-0.0249, -0.0128] better
  - pick − table_tabpfn -0.0107 [-0.0179, -0.0034] better
  - pick − table_missforest +0.0027 [-0.0027, +0.0081] none
  - crossfit − H -0.0193 [-0.0255, -0.0131] better
  - crossfit − table_tabpfn -0.0111 [-0.0182, -0.0040] better
  - crossfit − pick -0.0004 [-0.0014, +0.0005] equivalent
- `spambase_probit_60nan`: H 0.9575, tabpfn 0.9953, missforest 0.9616, table_tabpfn 0.9657, table_missforest 0.9524, pick 0.9641, picked_missforest 0.2000, crossfit 0.9593, crossfit_missforest 0.5333
  - table_missforest − H -0.0051 [-0.0061, -0.0041] better
  - table_missforest − missforest -0.0093 [-0.0120, -0.0066] better
  - table_tabpfn − H +0.0082 [+0.0050, +0.0114] worse
  - table_tabpfn − tabpfn -0.0296 [-0.0333, -0.0260] better
  - pick − H +0.0066 [+0.0022, +0.0110] worse
  - pick − table_tabpfn -0.0016 [-0.0035, +0.0003] equivalent
  - pick − table_missforest +0.0118 [+0.0072, +0.0163] worse
  - crossfit − H +0.0018 [-0.0033, +0.0069] equivalent
  - crossfit − table_tabpfn -0.0064 [-0.0105, -0.0024] better
  - crossfit − pick -0.0048 [-0.0090, -0.0006] equivalent
- `kc2_marf_60nan`: H 0.6115, tabpfn 0.6566, missforest 0.7006, table_tabpfn 0.6084, table_missforest 0.6183, pick 0.6102, picked_missforest 0.2000, crossfit 0.6092, crossfit_missforest 0.3333
  - table_missforest − H +0.0068 [-0.0067, +0.0203] none
  - table_missforest − missforest -0.0823 [-0.1166, -0.0480] better
  - table_tabpfn − H -0.0031 [-0.0146, +0.0084] none
  - table_tabpfn − tabpfn -0.0481 [-0.0823, -0.0140] better
  - pick − H -0.0013 [-0.0170, +0.0144] none
  - pick − table_tabpfn +0.0018 [-0.0087, +0.0122] none
  - pick − table_missforest -0.0081 [-0.0164, +0.0003] none
  - crossfit − H -0.0023 [-0.0181, +0.0135] none
  - crossfit − table_tabpfn +0.0007 [-0.0100, +0.0115] none
  - crossfit − pick -0.0010 [-0.0032, +0.0012] equivalent
- `kc2_probit_60nan`: H 0.6987, tabpfn 0.7500, missforest 0.7153, table_tabpfn 0.6940, table_missforest 0.6910, pick 0.6949, picked_missforest 0.0667, crossfit 0.6942, crossfit_missforest 0.4000
  - table_missforest − H -0.0077 [-0.0288, +0.0133] none
  - table_missforest − missforest -0.0244 [-0.0511, +0.0024] none
  - table_tabpfn − H -0.0047 [-0.0257, +0.0163] none
  - table_tabpfn − tabpfn -0.0560 [-0.0766, -0.0354] better
  - pick − H -0.0038 [-0.0249, +0.0174] none
  - pick − table_tabpfn +0.0009 [-0.0010, +0.0028] equivalent
  - pick − table_missforest +0.0039 [-0.0025, +0.0104] none
  - crossfit − H -0.0045 [-0.0257, +0.0166] none
  - crossfit − table_tabpfn +0.0001 [-0.0023, +0.0026] equivalent
  - crossfit − pick -0.0007 [-0.0021, +0.0006] equivalent
- `letter_20nan`: H 0.4224, tabpfn 0.4539, missforest 0.4622, table_tabpfn 0.4082, table_missforest 0.4053, pick 0.4053, picked_missforest 1.0000, crossfit 0.4053, crossfit_missforest 1.0000
  - table_missforest − H -0.0171 [-0.0180, -0.0162] better
  - table_missforest − missforest -0.0569 [-0.0595, -0.0543] better
  - table_tabpfn − H -0.0142 [-0.0150, -0.0133] better
  - table_tabpfn − tabpfn -0.0457 [-0.0485, -0.0429] better
  - pick − H -0.0171 [-0.0180, -0.0162] better
  - pick − table_tabpfn -0.0029 [-0.0034, -0.0025] better
  - pick − table_missforest +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − H -0.0171 [-0.0180, -0.0162] better
  - crossfit − table_tabpfn -0.0029 [-0.0034, -0.0025] better
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `spambase_20nan`: H 0.8524, tabpfn 0.9378, missforest 0.8594, table_tabpfn 0.8534, table_missforest 0.8417, pick 0.8471, picked_missforest 0.5333, crossfit 0.8471, crossfit_missforest 0.5333
  - table_missforest − H -0.0106 [-0.0134, -0.0079] better
  - table_missforest − missforest -0.0177 [-0.0267, -0.0087] better
  - table_tabpfn − H +0.0011 [-0.0039, +0.0060] equivalent
  - table_tabpfn − tabpfn -0.0844 [-0.0980, -0.0708] better
  - pick − H -0.0052 [-0.0096, -0.0009] better
  - pick − table_tabpfn -0.0063 [-0.0113, -0.0013] better
  - pick − table_missforest +0.0054 [+0.0013, +0.0095] worse
  - crossfit − H -0.0052 [-0.0096, -0.0009] better
  - crossfit − table_tabpfn -0.0063 [-0.0113, -0.0013] better
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent

E167 PRIMARY `table_missforest` on 12: worse than H on none, worse than missForest on none -> supported
E167 SECONDARY better than H on 10 of 12; on the 7 trails: ['biodeg_80nan', 'vehicle_80nan', 'credit-g_40nan', 'biodeg_probit0p25_60nan', 'biodeg_probit_60nan', 'vehicle_probit_60nan', 'electricity_20nan']
E167 PICK `pick`: worse than H on ['spambase_probit_60nan']; against `table_tabpfn` better on ['biodeg_probit0p25_60nan', 'biodeg_probit_60nan', 'vehicle_probit_60nan', 'electricity_20nan', 'letter_20nan', 'spambase_20nan'], worse on none -> mixed
E168 PRIMARY `crossfit`: worse than H on none; against `table_tabpfn` better on ['biodeg_probit0p25_60nan', 'biodeg_probit_60nan', 'vehicle_probit_60nan', 'electricity_20nan', 'spambase_probit_60nan', 'letter_20nan', 'spambase_20nan'], worse on none -> supported
E168 SECONDARY `crossfit` − `pick`: better on none, worse on none
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```
