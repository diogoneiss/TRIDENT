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
