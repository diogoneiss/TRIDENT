# E163 on fresh seeds (E166, T55; pre-registered 2026-11-09 00:57 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** E163 ([10](10-hybrid-transplant.md)) was supported on one seed set (42/7/13), its stated threat. Before the user
  decides whether ADR 0031 decision 6 (the deployable `table` hybrid) extends to clinical tables, the same study runs on
  seeds 4545/5656/6767, never used on these variants.
- **The user's go:** "A: E163 em sementes novas" (2026-11-09 00:40 GMT-3), on the plan for the night.
- **Disclosure:** E163's printout is on the record. Nothing here re-reads it.

## Design

E163's design unchanged except the seeds, the tag and the state: E149's six transplant variants
`mimic-{cardio,sofa}-donor_{tmatch,trand,mcar}_20nan` × 4545/5656/6767, 18 runs of H (`--decode_checkpoint loss
--decode_calibration none`) under `experiment = mimic-hybrid-seeds-2026-11-09`, state
`/scratch/diogoneiss/trident-2026-11-09-mimic-hybrid-seeds`. Runner `mimic_hybrid.py --study e166`. TabPFN and
missForest refit offline per training fold, both asserts as in E163. All on medusa1 (ADR 0035), in the same four GPU
queues as E167 ([imputation-protocol/62](../imputation-protocol/62-hybrid-fresh-seeds.md)), after E167's cells. The tally
also prints E168's cross-fit pick ([imputation-protocol/63](../imputation-protocol/63-crossfit-side-pick.md)), descriptive here.

## Measures and rule (fixed now)

E163's, unchanged: fold-pair verdict on 15 (seed, fold) differences, 1% SESOI. On the two `tmatch` variants,
`table_tabpfn`:
- **supported** if better than H on at least one, worse than H on neither, and worse than TabPFN on neither;
- **refuted** if better than H on neither, or worse than H on either;
- **mixed** otherwise.

Everything else descriptive. **Read** once with all 18 in: `TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data uv run
--python 3.11 python -B scripts/experiments/mimic_hybrid.py --study e166 --tally` on medusa1. **What it leads to:**
with E163, the evidence for the user's call on ADR 0031 and clinical tables. Threats: E163's, less the one seed set.

## Amendment

- **Amendment (2026-11-09 12:00 GMT-3), stalled, nothing read:** all 54 runs trained by 04:07 GMT-3. During the offline pass TabPFN ran out of GPU memory on the spambase pairs at 04:21 GMT-3. Another user's job held 26.4 GB of each card, and TabPFN keeps one model per column. At about 04:48 GMT-3 medusa1's GPU 0 failed ("Unknown Error"), and CUDA has been unusable on both cards since. After that TabPFN fell back to the CPU: `kc2_marf_60nan` s7, `kc2_probit_60nan` s42/s13 and E166's nine cardio cells were computed there, and `electricity_20nan` s7 is uncertain. The pass refused the SOFA tables (over 1000 rows). Pairs without cells: E167's letter_20nan s7/s13 and all six spambase pairs; E166's nine SOFA cells. Recomputing on a GPU, and where, is the user's call.
- **Amendment (2026-11-09 12:23 GMT-3), the offline pass moves to gorgona8 (the user's decision, "rode o tabpfn na gorgona8 mesmo"):** H's runs stay medusa1's. Their selection files and ledgers were copied to the same paths on gorgona8, and the store copy there holds all 54 parents. The TabPFN and missForest refits are redone on gorgona8's GPU and CPU for all 54 pairs, not only the missing ones, so that every pair's other side comes from one machine and one device. The H-ledger and missForest asserts still apply. This departs from ADR 0035, which wants every arm of a pair on one machine: H is from medusa1 and TabPFN from gorgona8, the same split in every pair. medusa1's partial cells are kept under `medusa1-partial/`. Comparing their TabPFN fills with gorgona8's is a device check only, not a read.

## Outcome (read once 2026-11-09 19:35 GMT-3: supported; written 2026-11-09 19:35 GMT-3)

Read on gorgona8 at commit c211f20, after all 18 pairs' cells were in (gorgona8's offline pass, no FAILED line).

- **Primary, supported.** On `mimic-sofa-donor_tmatch_20nan` the TabPFN-side hybrid is better than H (−0.0114
  [−0.0178, −0.0049]) and has no call against TabPFN (+0.0061); on `mimic-cardio-donor_tmatch_20nan` no call against H
  (−0.0022) and better than TabPFN (−0.0561). E163 replicates on seeds 4545/5656/6767.
- **Others:** better than H on 5 of 6 variants; worse than TabPFN on `mimic-sofa-donor_trand_20nan` (+0.0149). The
  missForest side is better than H on 3 of 6, equivalent on the SOFA `tmatch`.
- **Descriptive, not part of the rule:** the cross-fit pick (E168) is worse than `table_tabpfn` on the SOFA `tmatch`
  (+0.0054 [+0.0007, +0.0102]) and equivalent or no call elsewhere.
- **What it leads to:** extending ADR 0031 decision 6 to clinical tables is the user's call.

The printout, unedited:

```
- `mimic-cardio-donor_tmatch_20nan` (primary): H 0.8691, tabpfn 0.9230, missforest 0.8966, table_tabpfn 0.8669, table_missforest 0.8581, pick 0.8654, picked_missforest 0.1333, crossfit 0.8657, crossfit_missforest 0.2000
  - table_missforest − H -0.0110 [-0.0169, -0.0051] better
  - table_missforest − missforest -0.0385 [-0.0507, -0.0262] better
  - table_tabpfn − H -0.0022 [-0.0105, +0.0062] none
  - table_tabpfn − tabpfn -0.0561 [-0.0681, -0.0440] better
  - pick − H -0.0037 [-0.0103, +0.0029] none
  - pick − table_tabpfn -0.0015 [-0.0059, +0.0028] equivalent
  - pick − table_missforest +0.0072 [-0.0018, +0.0163] none
  - crossfit − H -0.0034 [-0.0099, +0.0031] none
  - crossfit − table_tabpfn -0.0012 [-0.0057, +0.0032] equivalent
  - the pick took missForest on 2 of 15 folds, the cross-fit pick on 3
- `mimic-cardio-donor_trand_20nan`: H 0.8605, tabpfn 0.8998, missforest 0.8978, table_tabpfn 0.8468, table_missforest 0.8522, pick 0.8472, picked_missforest 0.4667, crossfit 0.8463, crossfit_missforest 0.3333
  - table_missforest − H -0.0084 [-0.0143, -0.0024] better
  - table_missforest − missforest -0.0457 [-0.0551, -0.0362] better
  - table_tabpfn − H -0.0137 [-0.0225, -0.0048] better
  - table_tabpfn − tabpfn -0.0529 [-0.0658, -0.0401] better
  - pick − H -0.0133 [-0.0207, -0.0060] better
  - pick − table_tabpfn +0.0004 [-0.0050, +0.0057] equivalent
  - pick − table_missforest -0.0050 [-0.0104, +0.0005] none
  - crossfit − H -0.0142 [-0.0214, -0.0069] better
  - crossfit − table_tabpfn -0.0005 [-0.0057, +0.0047] equivalent
  - the pick took missForest on 7 of 15 folds, the cross-fit pick on 5
- `mimic-cardio-donor_mcar_20nan`: H 0.8037, tabpfn 0.7703, missforest 0.8003, table_tabpfn 0.7736, table_missforest 0.7861, pick 0.7764, picked_missforest 0.2667, crossfit 0.7783, crossfit_missforest 0.3333
  - table_missforest − H -0.0177 [-0.0261, -0.0092] better
  - table_missforest − missforest -0.0142 [-0.0218, -0.0065] better
  - table_tabpfn − H -0.0301 [-0.0397, -0.0206] better
  - table_tabpfn − tabpfn +0.0034 [-0.0042, +0.0109] none
  - pick − H -0.0274 [-0.0372, -0.0176] better
  - pick − table_tabpfn +0.0027 [-0.0012, +0.0067] equivalent
  - pick − table_missforest -0.0097 [-0.0160, -0.0035] better
  - crossfit − H -0.0255 [-0.0354, -0.0156] better
  - crossfit − table_tabpfn +0.0046 [-0.0002, +0.0094] none
  - the pick took missForest on 4 of 15 folds, the cross-fit pick on 5
- `mimic-sofa-donor_tmatch_20nan` (primary): H 0.9136, tabpfn 0.8961, missforest 0.9795, table_tabpfn 0.9022, table_missforest 0.9121, pick 0.9057, picked_missforest 0.3333, crossfit 0.9076, crossfit_missforest 0.4000
  - table_missforest − H -0.0014 [-0.0052, +0.0024] equivalent
  - table_missforest − missforest -0.0674 [-0.0868, -0.0480] better
  - table_tabpfn − H -0.0114 [-0.0178, -0.0049] better
  - table_tabpfn − tabpfn +0.0061 [-0.0046, +0.0168] none
  - pick − H -0.0079 [-0.0159, +0.0001] none
  - pick − table_tabpfn +0.0035 [-0.0010, +0.0080] equivalent
  - pick − table_missforest -0.0065 [-0.0128, -0.0001] better
  - crossfit − H -0.0059 [-0.0136, +0.0017] none
  - crossfit − table_tabpfn +0.0054 [+0.0007, +0.0102] worse
  - the pick took missForest on 5 of 15 folds, the cross-fit pick on 6
- `mimic-sofa-donor_trand_20nan`: H 0.9330, tabpfn 0.9029, missforest 0.9731, table_tabpfn 0.9178, table_missforest 0.9361, pick 0.9202, picked_missforest 0.3333, crossfit 0.9202, crossfit_missforest 0.3333
  - table_missforest − H +0.0030 [-0.0061, +0.0122] none
  - table_missforest − missforest -0.0370 [-0.0502, -0.0238] better
  - table_tabpfn − H -0.0153 [-0.0200, -0.0105] better
  - table_tabpfn − tabpfn +0.0149 [+0.0012, +0.0285] worse
  - pick − H -0.0128 [-0.0203, -0.0053] better
  - pick − table_tabpfn +0.0024 [-0.0026, +0.0074] equivalent
  - pick − table_missforest -0.0159 [-0.0254, -0.0064] better
  - crossfit − H -0.0128 [-0.0203, -0.0053] better
  - crossfit − table_tabpfn +0.0024 [-0.0026, +0.0074] equivalent
  - the pick took missForest on 5 of 15 folds, the cross-fit pick on 5
- `mimic-sofa-donor_mcar_20nan`: H 0.9214, tabpfn 0.8952, missforest 0.9156, table_tabpfn 0.9031, table_missforest 0.9118, pick 0.9050, picked_missforest 0.4000, crossfit 0.9062, crossfit_missforest 0.6000
  - table_missforest − H -0.0096 [-0.0139, -0.0053] better
  - table_missforest − missforest -0.0038 [-0.0151, +0.0075] none
  - table_tabpfn − H -0.0183 [-0.0248, -0.0119] better
  - table_tabpfn − tabpfn +0.0078 [-0.0029, +0.0186] none
  - pick − H -0.0164 [-0.0230, -0.0098] better
  - pick − table_tabpfn +0.0019 [-0.0015, +0.0053] equivalent
  - pick − table_missforest -0.0068 [-0.0117, -0.0020] better
  - crossfit − H -0.0152 [-0.0222, -0.0082] better
  - crossfit − table_tabpfn +0.0031 [-0.0003, +0.0065] equivalent
  - the pick took missForest on 6 of 15 folds, the cross-fit pick on 9

PRIMARY `table` (TabPFN side) on mimic-cardio-donor_tmatch_20nan, mimic-sofa-donor_tmatch_20nan: better than H on ['mimic-sofa-donor_tmatch_20nan'], worse than H on none, worse than TabPFN on none -> supported
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```

## Decision

- **2026-11-09 19:51 GMT-3:** ADR 0031 decision 6's scope extended to the MIMIC-IV transplant variants with E163 and E166, on
  the user's decision ("vamos no 2b e 3").
