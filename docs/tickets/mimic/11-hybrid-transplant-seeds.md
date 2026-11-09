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
