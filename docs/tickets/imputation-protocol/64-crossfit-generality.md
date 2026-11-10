# The cross-fit side pick on the other 26 variants (E169, T54; pre-registered 2026-11-09 19:17 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E168 ([63](63-crossfit-side-pick.md)) supported the cross-fit pick on E167's 12 fresh variants
  ([62](62-hybrid-fresh-seeds.md)): worse than H on none, better than the accepted TabPFN-side hybrid (`table_tabpfn`,
  ADR 0031 decision 6) on 7 and worse on none. Those 12 were chosen because they decided E164/E165, so the reading says
  nothing about the rest of E164's 38 variants. This study asks whether the rule is safe there.
- **The user's go:** "E169 generalidade" (2026-11-09, just after 19:01 GMT-3), medusa1 back after its GPU failure.
- **Disclosure:** E168's development printout already covers these 26 on seeds 4545/5656/6767. There `crossfit` is worse
  than H on none of the 26 and better than `table_tabpfn` on only one (`biodeg_probit1p0_60nan`), worse on none. So
  E168's "better on at least 2" clause would almost decide the study before it runs; the question here is no harm, and
  the rule below is shaped for it. Runs of these variants on 42/7/13 exist in other studies (K readouts, no hybrid);
  none is read here.

## Design

- **Variants (26):** E164's 38 minus E167's 12, listed in `hybrid_fresh_seeds.REST`:
  electricity 40/80nan; kr-vs-kp 20/40/60/80nan; credit-g 20/60/80nan and `credit-g_marf_60nan`; biodeg 20/40nan,
  `biodeg_probit1p0_60nan`, `biodeg_probit2p0_60nan`, `biodeg_marf_60nan`; kc2 20/40/80nan; pendigits 20/40/60/80nan;
  vehicle 20/40nan and `vehicle_marf_60nan`; `spambase_marf_60nan`.
- **Cells:** 26 × 42/7/13 = 78 runs of H under `experiment = hybrid-crossfit-generality-2026-11-09`, state
  `/scratch/diogoneiss/trident-2026-11-09-hybrid-crossfit-generality`, runner
  `scripts/experiments/hybrid_fresh_seeds.py --study e169`. Imputation defaults with H's flags; pendigits therefore
  trains under ADR 0036's P600 (20–40nan) and ADR 0037's bigger encoder (40–60nan), unlike E91/E95's parents, so its
  development and fresh levels are not directly comparable (disclosed). Offline per pair, as E167: TabPFN refit (H's
  ledger asserted equal to the run's H), then missForest refit (asserted equal to the run's logged missForest), into
  `cells-mf`.
- **Machine (ADR 0035):** everything on medusa1, H, TabPFN and missForest alike: four GPU queues (two per GPU), longest
  first, then one offline lane per GPU once that GPU's two queues are done.
- **Cost:** E167's runs took about a fifth of E82's minutes on medusa1 (letter 20–25 min against 118), so the 78 runs
  should take about 1.5–2.5 h on four queues; the offline pass, with six electricity pairs, about 4–6 h. State dir:
  E167's 36 pairs took 545 MB, against 174 GB free on medusa1's `/scratch`.

## Measures and rules (fixed now)

Fold-pair verdict on 15 (seed, fold) differences, 1% SESOI of the opponent's mean (E90's rule).
- **E169 primary, `crossfit`** (E168's rule, the no-harm half):
  - **supported** if worse than H on none of 26 and worse than `table_tabpfn` on at most 1;
  - **refuted** if worse than H on 2 or more, or worse than `table_tabpfn` on 3 or more;
  - **mixed** otherwise.
- **Secondary:** `crossfit` better than `table_tabpfn` on which; better than H on how many; `crossfit` − `pick`;
  `table_missforest` against H and missForest (E164's question on fresh seeds).
- Three seeds, fixed; no extension.
- **Read** once with all 78 in: `uv run --python 3.11 python -B scripts/experiments/hybrid_fresh_seeds.py --study e169 --tally`
  on medusa1.
- **What it leads to:** with E168, whether the cross-fit pick goes to the user as an ADR 0031 amendment for all 38
  variants.

## Amendment

- **Amendment (2026-11-09 22:44 GMT-3), stalled, nothing read:** all 78 runs of H trained on medusa1 by 22:28 GMT-3
  with no FAILED line. The offline pass had finished 4 pairs (`electricity_40nan` s7/s42, `electricity_80nan` s13/s42, the
  last at 22:38:44) when medusa1's GPU 0 failed again ("Unknown Error", the first failed cell at 22:40:07). As at 04:48 the
  same day, CUDA has since been unusable on both cards: a restart on GPU 1 alone failed at CUDA init, and TabPFN fell back
  to the CPU and refused tables over 1000 rows. Every partial output is kept: the first attempt in `cells.gpu0/` and
  `cells-mf.gpu0/`, the GPU 1 attempt's logs as `logs/gpu1_cells_*`. No cell was scored. Where the offline pass is redone
  is the user's call; E170 (65) waits on it.
- **Amendment (2026-11-09 23:20 GMT-3), the offline pass moves to gorgona8 (the user's decision, "gorgona8 após
  E171/E172"), as E167's did:** H's runs stay medusa1's. Their selection files and ledgers (78 and 390) were copied to the
  same paths on gorgona8, whose store copy holds all 78 parents. The TabPFN and missForest refits are redone on gorgona8's
  GPU and CPU for all 78 pairs, the 4 done on medusa1 among them, so every pair's other side comes from one machine and one
  device. Two lanes (`cells.sh` in the state dir on gorgona8) start once E171/E172's offline lanes are complete, about
  06:00–09:00 GMT-3 on 2026-11-10. The H-ledger and missForest asserts still apply. This departs from ADR 0035, which wants
  every arm of a pair on one machine: H is from medusa1 and TabPFN from gorgona8, the same split in every pair. The read
  moves to gorgona8 with the same command.

## Outcome (read once 2026-11-10 12:41 GMT-3: mixed; written 2026-11-10 12:44 GMT-3)

Read on gorgona8 at commit c11458c, after all 78 pairs' cells were in (gorgona8's offline pass of the amendment above;
no FAILED line; the H-ledger and missForest asserts held on every pair).

- **E169 primary, mixed.** The cross-fit pick is worse than H on 1 of 26, `biodeg_probit2p0_60nan` (+0.0095
  [+0.0051, +0.0139]), where the accepted TabPFN-side hybrid (`table_tabpfn`) is worse than H too (+0.0106). It is worse
  than `table_tabpfn` on 2: `credit-g_marf_60nan` (+0.0115 [+0.0020, +0.0210]; missForest's side on 60% of folds,
  where missForest alone is the worst arm) and `biodeg_20nan` (+0.0062 [+0.0010, +0.0114]). Not refuted (that needed 2
  against H or 3 against `table_tabpfn`).
- **Secondary:** better than `table_tabpfn` on none of 26 (E168 found 7 on the 12 that motivated it); better than H on
  19 of 26; `crossfit` − `pick` worse on `credit-g_marf_60nan`, better on none. `table_missforest` worse than H on
  `credit-g_80nan` (+0.0121 [+0.0016, +0.0227]), worse than missForest on none.
- **What it leads to:** outside the 12 variants that motivated it, the cross-fit pick gains nothing over the accepted
  hybrid and loses to it twice. With E171 (worse than `table_tabpfn` on the MIMIC SOFA `tmatch`) and E172, the evidence
  does not favour an ADR 0031 amendment replacing `table_tabpfn` with it. That, and any narrower use where H trails
  missForest (E167/E168), is the user's call.

The printout, unedited (E170's lines, [65](65-crossfit-off-mcar.md), are its last two):

```
- `electricity_40nan`: H 0.7479, tabpfn 0.7430, missforest 0.7912, table_tabpfn 0.7367, table_missforest 0.7468, pick 0.7364, picked_missforest 0.1333, crossfit 0.7367, crossfit_missforest 0.0667
  - table_missforest − H -0.0010 [-0.0074, +0.0054] equivalent
  - table_missforest − missforest -0.0444 [-0.0632, -0.0256] better
  - table_tabpfn − H -0.0112 [-0.0215, -0.0008] better
  - table_tabpfn − tabpfn -0.0063 [-0.0188, +0.0061] none
  - pick − H -0.0114 [-0.0217, -0.0011] better
  - pick − table_tabpfn -0.0002 [-0.0009, +0.0004] equivalent
  - pick − table_missforest -0.0104 [-0.0204, -0.0004] better
  - crossfit − H -0.0111 [-0.0215, -0.0008] better
  - crossfit − table_tabpfn +0.0001 [-0.0001, +0.0002] equivalent
  - crossfit − pick +0.0003 [-0.0003, +0.0009] equivalent
- `electricity_80nan`: H 0.8984, tabpfn 0.8905, missforest 1.0060, table_tabpfn 0.8899, table_missforest 0.8989, pick 0.8899, picked_missforest 0.0000, crossfit 0.8899, crossfit_missforest 0.0000
  - table_missforest − H +0.0005 [-0.0001, +0.0011] equivalent
  - table_missforest − missforest -0.1070 [-0.1237, -0.0903] better
  - table_tabpfn − H -0.0084 [-0.0102, -0.0067] better
  - table_tabpfn − tabpfn -0.0006 [-0.0012, -0.0000] better
  - pick − H -0.0084 [-0.0102, -0.0067] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0090 [-0.0107, -0.0073] better
  - crossfit − H -0.0084 [-0.0102, -0.0067] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `kr-vs-kp_40nan`: H 0.7299, tabpfn 0.7119, missforest 0.8007, table_tabpfn 0.7114, table_missforest 0.7299, pick 0.7114, picked_missforest 0.0000, crossfit 0.7156, crossfit_missforest 0.2667
  - table_missforest − H +0.0000 [+0.0000, +0.0000] equivalent
  - table_missforest − missforest -0.0708 [-0.0822, -0.0595] better
  - table_tabpfn − H -0.0185 [-0.0255, -0.0114] better
  - table_tabpfn − tabpfn -0.0004 [-0.0024, +0.0015] equivalent
  - pick − H -0.0185 [-0.0255, -0.0114] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0185 [-0.0255, -0.0114] better
  - crossfit − H -0.0142 [-0.0211, -0.0074] better
  - crossfit − table_tabpfn +0.0042 [-0.0023, +0.0107] none
  - crossfit − pick +0.0042 [-0.0023, +0.0107] none
- `kr-vs-kp_60nan`: H 0.8750, tabpfn 0.8271, missforest 0.9355, table_tabpfn 0.8271, table_missforest 0.8826, pick 0.8271, picked_missforest 0.0000, crossfit 0.8271, crossfit_missforest 0.0000
  - table_missforest − H +0.0076 [-0.0036, +0.0188] none
  - table_missforest − missforest -0.0529 [-0.0674, -0.0384] better
  - table_tabpfn − H -0.0479 [-0.0547, -0.0411] better
  - table_tabpfn − tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − H -0.0479 [-0.0547, -0.0411] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0555 [-0.0698, -0.0412] better
  - crossfit − H -0.0479 [-0.0547, -0.0411] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `kr-vs-kp_80nan`: H 1.0047, tabpfn 0.9419, missforest 1.0211, table_tabpfn 0.9419, table_missforest 1.0112, pick 0.9419, picked_missforest 0.0000, crossfit 0.9462, crossfit_missforest 0.0667
  - table_missforest − H +0.0065 [-0.0004, +0.0134] none
  - table_missforest − missforest -0.0099 [-0.0200, +0.0001] none
  - table_tabpfn − H -0.0628 [-0.0742, -0.0514] better
  - table_tabpfn − tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − H -0.0628 [-0.0742, -0.0514] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0693 [-0.0799, -0.0587] better
  - crossfit − H -0.0585 [-0.0730, -0.0440] better
  - crossfit − table_tabpfn +0.0043 [-0.0049, +0.0136] none
  - crossfit − pick +0.0043 [-0.0049, +0.0136] none
- `credit-g_60nan`: H 0.9958, tabpfn 0.9535, missforest 1.0209, table_tabpfn 0.9539, table_missforest 0.9930, pick 0.9539, picked_missforest 0.0000, crossfit 0.9602, crossfit_missforest 0.2000
  - table_missforest − H -0.0028 [-0.0104, +0.0048] none
  - table_missforest − missforest -0.0278 [-0.0409, -0.0148] better
  - table_tabpfn − H -0.0420 [-0.0580, -0.0259] better
  - table_tabpfn − tabpfn +0.0004 [-0.0042, +0.0049] equivalent
  - pick − H -0.0420 [-0.0580, -0.0259] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0392 [-0.0528, -0.0256] better
  - crossfit − H -0.0357 [-0.0512, -0.0201] better
  - crossfit − table_tabpfn +0.0063 [-0.0055, +0.0181] none
  - crossfit − pick +0.0063 [-0.0055, +0.0181] none
- `credit-g_80nan`: H 1.0181, tabpfn 1.0007, missforest 1.0928, table_tabpfn 1.0045, table_missforest 1.0302, pick 1.0092, picked_missforest 0.2000, crossfit 1.0100, crossfit_missforest 0.4000
  - table_missforest − H +0.0121 [+0.0016, +0.0227] worse
  - table_missforest − missforest -0.0626 [-0.0758, -0.0493] better
  - table_tabpfn − H -0.0136 [-0.0216, -0.0055] better
  - table_tabpfn − tabpfn +0.0039 [-0.0033, +0.0110] none
  - pick − H -0.0088 [-0.0223, +0.0046] none
  - pick − table_tabpfn +0.0047 [-0.0039, +0.0134] none
  - pick − table_missforest -0.0210 [-0.0350, -0.0070] better
  - crossfit − H -0.0080 [-0.0215, +0.0055] none
  - crossfit − table_tabpfn +0.0055 [-0.0031, +0.0141] none
  - crossfit − pick +0.0008 [-0.0004, +0.0021] equivalent
- `biodeg_40nan`: H 0.6770, tabpfn 0.7145, missforest 0.6937, table_tabpfn 0.6628, table_missforest 0.6572, pick 0.6582, picked_missforest 0.8000, crossfit 0.6582, crossfit_missforest 0.8000
  - table_missforest − H -0.0198 [-0.0322, -0.0074] better
  - table_missforest − missforest -0.0366 [-0.0479, -0.0252] better
  - table_tabpfn − H -0.0142 [-0.0269, -0.0015] better
  - table_tabpfn − tabpfn -0.0517 [-0.0707, -0.0328] better
  - pick − H -0.0188 [-0.0335, -0.0041] better
  - pick − table_tabpfn -0.0047 [-0.0122, +0.0029] none
  - pick − table_missforest +0.0010 [-0.0028, +0.0048] equivalent
  - crossfit − H -0.0188 [-0.0335, -0.0041] better
  - crossfit − table_tabpfn -0.0047 [-0.0122, +0.0029] none
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `kc2_40nan`: H 0.5265, tabpfn 0.5278, missforest 0.6147, table_tabpfn 0.5071, table_missforest 0.5231, pick 0.5152, picked_missforest 0.4667, crossfit 0.5170, crossfit_missforest 0.6000
  - table_missforest − H -0.0034 [-0.0274, +0.0206] none
  - table_missforest − missforest -0.0916 [-0.1358, -0.0474] better
  - table_tabpfn − H -0.0194 [-0.0372, -0.0016] better
  - table_tabpfn − tabpfn -0.0207 [-0.0641, +0.0227] none
  - pick − H -0.0114 [-0.0384, +0.0157] none
  - pick − table_tabpfn +0.0080 [-0.0074, +0.0234] none
  - pick − table_missforest -0.0080 [-0.0177, +0.0017] none
  - crossfit − H -0.0096 [-0.0364, +0.0173] none
  - crossfit − table_tabpfn +0.0098 [-0.0055, +0.0252] none
  - crossfit − pick +0.0018 [-0.0010, +0.0047] equivalent
- `kc2_80nan`: H 0.7066, tabpfn 0.7351, missforest 0.7306, table_tabpfn 0.7034, table_missforest 0.6986, pick 0.7003, picked_missforest 0.4000, crossfit 0.6990, crossfit_missforest 0.4667
  - table_missforest − H -0.0080 [-0.0262, +0.0102] none
  - table_missforest − missforest -0.0320 [-0.0804, +0.0164] none
  - table_tabpfn − H -0.0032 [-0.0180, +0.0116] none
  - table_tabpfn − tabpfn -0.0317 [-0.0780, +0.0146] none
  - pick − H -0.0063 [-0.0247, +0.0121] none
  - pick − table_tabpfn -0.0031 [-0.0114, +0.0053] none
  - pick − table_missforest +0.0017 [-0.0013, +0.0047] equivalent
  - crossfit − H -0.0077 [-0.0261, +0.0107] none
  - crossfit − table_tabpfn -0.0045 [-0.0132, +0.0043] none
  - crossfit − pick -0.0014 [-0.0044, +0.0016] equivalent
- `pendigits_40nan`: H 0.3926, tabpfn 0.4367, missforest 0.4963, table_tabpfn 0.3790, table_missforest 0.3804, pick 0.3790, picked_missforest 0.0667, crossfit 0.3790, crossfit_missforest 0.0667
  - table_missforest − H -0.0122 [-0.0134, -0.0110] better
  - table_missforest − missforest -0.1159 [-0.1230, -0.1088] better
  - table_tabpfn − H -0.0136 [-0.0148, -0.0125] better
  - table_tabpfn − tabpfn -0.0577 [-0.0618, -0.0536] better
  - pick − H -0.0137 [-0.0148, -0.0125] better
  - pick − table_tabpfn -0.0000 [-0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0014 [-0.0023, -0.0006] better
  - crossfit − H -0.0137 [-0.0148, -0.0125] better
  - crossfit − table_tabpfn -0.0000 [-0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `pendigits_60nan`: H 0.5756, tabpfn 0.6035, missforest 0.7071, table_tabpfn 0.5553, table_missforest 0.5583, pick 0.5553, picked_missforest 0.0000, crossfit 0.5553, crossfit_missforest 0.0000
  - table_missforest − H -0.0173 [-0.0182, -0.0164] better
  - table_missforest − missforest -0.1488 [-0.1553, -0.1423] better
  - table_tabpfn − H -0.0203 [-0.0215, -0.0190] better
  - table_tabpfn − tabpfn -0.0482 [-0.0513, -0.0451] better
  - pick − H -0.0203 [-0.0215, -0.0190] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0030 [-0.0042, -0.0018] better
  - crossfit − H -0.0203 [-0.0215, -0.0190] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `pendigits_80nan`: H 0.8328, tabpfn 0.8144, missforest 0.9375, table_tabpfn 0.8099, table_missforest 0.8213, pick 0.8099, picked_missforest 0.0000, crossfit 0.8099, crossfit_missforest 0.0000
  - table_missforest − H -0.0115 [-0.0128, -0.0102] better
  - table_missforest − missforest -0.1162 [-0.1237, -0.1087] better
  - table_tabpfn − H -0.0229 [-0.0249, -0.0209] better
  - table_tabpfn − tabpfn -0.0046 [-0.0057, -0.0034] better
  - pick − H -0.0229 [-0.0249, -0.0209] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0114 [-0.0126, -0.0103] better
  - crossfit − H -0.0229 [-0.0249, -0.0209] better
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `vehicle_40nan`: H 0.4910, tabpfn 0.4967, missforest 0.5194, table_tabpfn 0.4777, table_missforest 0.4802, pick 0.4785, picked_missforest 0.6000, crossfit 0.4790, crossfit_missforest 0.7333
  - table_missforest − H -0.0108 [-0.0197, -0.0019] better
  - table_missforest − missforest -0.0392 [-0.0598, -0.0186] better
  - table_tabpfn − H -0.0133 [-0.0220, -0.0046] better
  - table_tabpfn − tabpfn -0.0190 [-0.0317, -0.0064] better
  - pick − H -0.0124 [-0.0228, -0.0020] better
  - pick − table_tabpfn +0.0009 [-0.0041, +0.0058] none
  - pick − table_missforest -0.0016 [-0.0041, +0.0008] equivalent
  - crossfit − H -0.0119 [-0.0219, -0.0020] better
  - crossfit − table_tabpfn +0.0013 [-0.0038, +0.0065] none
  - crossfit − pick +0.0005 [-0.0010, +0.0019] equivalent
- `biodeg_probit1p0_60nan`: H 0.9317, tabpfn 0.9378, missforest 0.9337, table_tabpfn 0.9310, table_missforest 0.9258, pick 0.9306, picked_missforest 0.2667, crossfit 0.9302, crossfit_missforest 0.4667
  - table_missforest − H -0.0059 [-0.0083, -0.0035] better
  - table_missforest − missforest -0.0080 [-0.0115, -0.0045] better
  - table_tabpfn − H -0.0006 [-0.0041, +0.0028] equivalent
  - table_tabpfn − tabpfn -0.0068 [-0.0097, -0.0039] better
  - pick − H -0.0011 [-0.0048, +0.0026] equivalent
  - pick − table_tabpfn -0.0004 [-0.0014, +0.0005] equivalent
  - pick − table_missforest +0.0048 [+0.0020, +0.0076] worse
  - crossfit − H -0.0014 [-0.0049, +0.0021] equivalent
  - crossfit − table_tabpfn -0.0008 [-0.0017, +0.0001] equivalent
  - crossfit − pick -0.0003 [-0.0014, +0.0008] equivalent
- `biodeg_probit2p0_60nan`: H 0.9291, tabpfn 0.9557, missforest 0.9526, table_tabpfn 0.9397, table_missforest 0.9336, pick 0.9395, picked_missforest 0.0667, crossfit 0.9386, crossfit_missforest 0.2000
  - table_missforest − H +0.0045 [-0.0001, +0.0091] equivalent
  - table_missforest − missforest -0.0190 [-0.0244, -0.0135] better
  - table_tabpfn − H +0.0106 [+0.0065, +0.0147] worse
  - table_tabpfn − tabpfn -0.0161 [-0.0213, -0.0109] better
  - pick − H +0.0104 [+0.0064, +0.0144] worse
  - pick − table_tabpfn -0.0002 [-0.0006, +0.0002] equivalent
  - pick − table_missforest +0.0059 [+0.0017, +0.0101] worse
  - crossfit − H +0.0095 [+0.0051, +0.0139] worse
  - crossfit − table_tabpfn -0.0011 [-0.0024, +0.0003] equivalent
  - crossfit − pick -0.0009 [-0.0022, +0.0005] equivalent
- `biodeg_marf_60nan`: H 0.6731, tabpfn 0.7053, missforest 0.7165, table_tabpfn 0.6526, table_missforest 0.6539, pick 0.6538, picked_missforest 0.2667, crossfit 0.6536, crossfit_missforest 0.4000
  - table_missforest − H -0.0193 [-0.0238, -0.0147] better
  - table_missforest − missforest -0.0626 [-0.0748, -0.0505] better
  - table_tabpfn − H -0.0205 [-0.0278, -0.0133] better
  - table_tabpfn − tabpfn -0.0527 [-0.0689, -0.0364] better
  - pick − H -0.0194 [-0.0266, -0.0121] better
  - pick − table_tabpfn +0.0012 [-0.0005, +0.0029] equivalent
  - pick − table_missforest -0.0001 [-0.0037, +0.0035] equivalent
  - crossfit − H -0.0196 [-0.0267, -0.0124] better
  - crossfit − table_tabpfn +0.0010 [-0.0009, +0.0028] equivalent
  - crossfit − pick -0.0002 [-0.0010, +0.0006] equivalent
- `vehicle_marf_60nan`: H 0.4551, tabpfn 0.4554, missforest 0.4945, table_tabpfn 0.4406, table_missforest 0.4440, pick 0.4400, picked_missforest 0.2667, crossfit 0.4400, crossfit_missforest 0.3333
  - table_missforest − H -0.0111 [-0.0165, -0.0057] better
  - table_missforest − missforest -0.0505 [-0.0801, -0.0209] better
  - table_tabpfn − H -0.0144 [-0.0197, -0.0091] better
  - table_tabpfn − tabpfn -0.0148 [-0.0578, +0.0283] none
  - pick − H -0.0151 [-0.0208, -0.0093] better
  - pick − table_tabpfn -0.0006 [-0.0034, +0.0021] equivalent
  - pick − table_missforest -0.0040 [-0.0094, +0.0015] none
  - crossfit − H -0.0151 [-0.0208, -0.0093] better
  - crossfit − table_tabpfn -0.0006 [-0.0034, +0.0021] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `spambase_marf_60nan`: H 0.9046, tabpfn 0.9367, missforest 0.9632, table_tabpfn 0.8975, table_missforest 0.8977, pick 0.8969, picked_missforest 0.1333, crossfit 0.8961, crossfit_missforest 0.4000
  - table_missforest − H -0.0069 [-0.0095, -0.0043] better
  - table_missforest − missforest -0.0654 [-0.0708, -0.0601] better
  - table_tabpfn − H -0.0071 [-0.0105, -0.0037] better
  - table_tabpfn − tabpfn -0.0392 [-0.0486, -0.0298] better
  - pick − H -0.0077 [-0.0109, -0.0045] better
  - pick − table_tabpfn -0.0006 [-0.0019, +0.0008] equivalent
  - pick − table_missforest -0.0008 [-0.0034, +0.0019] equivalent
  - crossfit − H -0.0086 [-0.0116, -0.0055] better
  - crossfit − table_tabpfn -0.0014 [-0.0038, +0.0009] equivalent
  - crossfit − pick -0.0009 [-0.0029, +0.0012] equivalent
- `credit-g_marf_60nan`: H 0.9352, tabpfn 0.9011, missforest 0.9642, table_tabpfn 0.9122, table_missforest 0.9333, pick 0.9122, picked_missforest 0.0000, crossfit 0.9237, crossfit_missforest 0.6000
  - table_missforest − H -0.0019 [-0.0033, -0.0006] better
  - table_missforest − missforest -0.0310 [-0.0448, -0.0171] better
  - table_tabpfn − H -0.0230 [-0.0322, -0.0137] better
  - table_tabpfn − tabpfn +0.0111 [+0.0018, +0.0204] worse
  - pick − H -0.0230 [-0.0322, -0.0137] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0210 [-0.0309, -0.0111] better
  - crossfit − H -0.0115 [-0.0201, -0.0029] better
  - crossfit − table_tabpfn +0.0115 [+0.0020, +0.0210] worse
  - crossfit − pick +0.0115 [+0.0020, +0.0210] worse
- `pendigits_20nan`: H 0.3032, tabpfn 0.3079, missforest 0.3344, table_tabpfn 0.2851, table_missforest 0.2867, pick 0.2854, picked_missforest 0.2667, crossfit 0.2854, crossfit_missforest 0.2667
  - table_missforest − H -0.0165 [-0.0176, -0.0154] better
  - table_missforest − missforest -0.0478 [-0.0528, -0.0428] better
  - table_tabpfn − H -0.0180 [-0.0186, -0.0174] better
  - table_tabpfn − tabpfn -0.0228 [-0.0265, -0.0191] better
  - pick − H -0.0177 [-0.0185, -0.0170] better
  - pick − table_tabpfn +0.0003 [-0.0000, +0.0006] equivalent
  - pick − table_missforest -0.0012 [-0.0021, -0.0004] better
  - crossfit − H -0.0177 [-0.0185, -0.0170] better
  - crossfit − table_tabpfn +0.0003 [-0.0000, +0.0006] equivalent
  - crossfit − pick +0.0000 [+0.0000, +0.0000] equivalent
- `kr-vs-kp_20nan`: H 0.5980, tabpfn 0.5828, missforest 0.6578, table_tabpfn 0.5895, table_missforest 0.5980, pick 0.5895, picked_missforest 0.0000, crossfit 0.5933, crossfit_missforest 0.3333
  - table_missforest − H +0.0000 [+0.0000, +0.0000] equivalent
  - table_missforest − missforest -0.0598 [-0.0723, -0.0472] better
  - table_tabpfn − H -0.0085 [-0.0169, -0.0002] better
  - table_tabpfn − tabpfn +0.0067 [-0.0009, +0.0143] none
  - pick − H -0.0085 [-0.0169, -0.0002] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0085 [-0.0169, -0.0002] better
  - crossfit − H -0.0047 [-0.0124, +0.0031] none
  - crossfit − table_tabpfn +0.0039 [-0.0009, +0.0086] none
  - crossfit − pick +0.0039 [-0.0009, +0.0086] none
- `biodeg_20nan`: H 0.5862, tabpfn 0.6164, missforest 0.6183, table_tabpfn 0.5646, table_missforest 0.5736, pick 0.5681, picked_missforest 0.5333, crossfit 0.5708, crossfit_missforest 0.6667
  - table_missforest − H -0.0126 [-0.0241, -0.0012] better
  - table_missforest − missforest -0.0447 [-0.0619, -0.0276] better
  - table_tabpfn − H -0.0216 [-0.0342, -0.0090] better
  - table_tabpfn − tabpfn -0.0518 [-0.0719, -0.0318] better
  - pick − H -0.0180 [-0.0316, -0.0045] better
  - pick − table_tabpfn +0.0035 [-0.0006, +0.0077] none
  - pick − table_missforest -0.0054 [-0.0122, +0.0013] none
  - crossfit − H -0.0154 [-0.0290, -0.0018] better
  - crossfit − table_tabpfn +0.0062 [+0.0010, +0.0114] worse
  - crossfit − pick +0.0027 [-0.0016, +0.0070] none
- `credit-g_20nan`: H 0.8855, tabpfn 0.8717, missforest 0.8846, table_tabpfn 0.8677, table_missforest 0.8814, pick 0.8728, picked_missforest 0.2000, crossfit 0.8745, crossfit_missforest 0.2667
  - table_missforest − H -0.0041 [-0.0139, +0.0057] none
  - table_missforest − missforest -0.0033 [-0.0128, +0.0062] none
  - table_tabpfn − H -0.0178 [-0.0261, -0.0095] better
  - table_tabpfn − tabpfn -0.0041 [-0.0091, +0.0010] none
  - pick − H -0.0127 [-0.0234, -0.0020] better
  - pick − table_tabpfn +0.0051 [-0.0025, +0.0128] none
  - pick − table_missforest -0.0086 [-0.0163, -0.0008] better
  - crossfit − H -0.0110 [-0.0215, -0.0004] better
  - crossfit − table_tabpfn +0.0068 [-0.0012, +0.0148] none
  - crossfit − pick +0.0017 [-0.0016, +0.0050] equivalent
- `vehicle_20nan`: H 0.4206, tabpfn 0.4200, missforest 0.4516, table_tabpfn 0.4073, table_missforest 0.4156, pick 0.4077, picked_missforest 0.2667, crossfit 0.4098, crossfit_missforest 0.4000
  - table_missforest − H -0.0050 [-0.0098, -0.0001] better
  - table_missforest − missforest -0.0360 [-0.0508, -0.0212] better
  - table_tabpfn − H -0.0133 [-0.0193, -0.0073] better
  - table_tabpfn − tabpfn -0.0127 [-0.0212, -0.0042] better
  - pick − H -0.0129 [-0.0197, -0.0061] better
  - pick − table_tabpfn +0.0004 [-0.0017, +0.0025] equivalent
  - pick − table_missforest -0.0079 [-0.0149, -0.0010] better
  - crossfit − H -0.0107 [-0.0177, -0.0038] better
  - crossfit − table_tabpfn +0.0026 [-0.0016, +0.0068] none
  - crossfit − pick +0.0022 [-0.0016, +0.0059] none
- `kc2_20nan`: H 0.5651, tabpfn 0.5244, missforest 0.6054, table_tabpfn 0.5111, table_missforest 0.5377, pick 0.5202, picked_missforest 0.4000, crossfit 0.5117, crossfit_missforest 0.4000
  - table_missforest − H -0.0275 [-0.0566, +0.0017] none
  - table_missforest − missforest -0.0677 [-0.1214, -0.0141] better
  - table_tabpfn − H -0.0541 [-0.1097, +0.0016] none
  - table_tabpfn − tabpfn -0.0134 [-0.0687, +0.0419] none
  - pick − H -0.0450 [-0.0966, +0.0066] none
  - pick − table_tabpfn +0.0091 [-0.0100, +0.0282] none
  - pick − table_missforest -0.0175 [-0.0499, +0.0149] none
  - crossfit − H -0.0534 [-0.1084, +0.0016] none
  - crossfit − table_tabpfn +0.0007 [-0.0071, +0.0084] none
  - crossfit − pick -0.0084 [-0.0260, +0.0091] none

E169 PRIMARY `crossfit` on 26: worse than H on ['biodeg_probit2p0_60nan'], worse than `table_tabpfn` on ['credit-g_marf_60nan', 'biodeg_20nan'] -> mixed
E169 SECONDARY `crossfit` better than `table_tabpfn` on none; better than H on 19 of 26; `crossfit` − `pick` better on none, worse on ['credit-g_marf_60nan']
E169 SECONDARY `table_missforest`: worse than H on ['credit-g_80nan'], worse than missForest on none
E170 PRIMARY `crossfit` on E93's 6 off-MCAR variants here: worse than H on ['biodeg_probit2p0_60nan'] -> mixed
E170 SECONDARY on them: `crossfit` better than H on ['biodeg_marf_60nan', 'vehicle_marf_60nan', 'spambase_marf_60nan', 'credit-g_marf_60nan']; `table_tabpfn` worse than H on ['biodeg_probit2p0_60nan']; `crossfit` worse than `table_tabpfn` on ['credit-g_marf_60nan']
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```
