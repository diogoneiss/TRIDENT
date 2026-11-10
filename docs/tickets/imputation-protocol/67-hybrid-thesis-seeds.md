# ADR 0031 decision 6's hybrid at the thesis seeds, on the 26 MCAR variants (E174, T54; pre-registered 2026-11-10 16:40 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** the thesis table's primary seed set is 101/202/303 (T21, ADR 0020). The deployable TabPFN-side hybrid
  (`table_tabpfn`, ADR 0031 decision 6) was accepted on 4545/5656/6767 (E90, E91, E95) and read again on 42/7/13 (E167,
  E169), never on the thesis seeds for MCAR variants. This study gives the thesis table a hybrid column on the same
  seeds as its other columns, and a third seed set for decision 6 on all 26 MCAR variants of E164's 38.
- **The user's go:** "Os dois em sequência" (2026-11-10, about 16:30 GMT-3), with E173 (the hybrid on the MIMIC-IV
  natural-gap tables). E174 runs first because it needs no code change; E173's runner needs the masked population.
- **Disclosure:** on 42/7/13 (E167, E169 printouts, already read) `table_tabpfn` is worse than H on none of these 26 and
  worse than TabPFN on none, better than H on 23. On 4545/5656/6767 decision 6 was accepted on a subset of them. Seeds
  101/202/303 have H runs of these variants from the thesis table's studies, but no hybrid cells (the hidden observed
  validation cells are written only by the hybrid's runner), so all runs are new.

## Design

- **Variants (26), `hybrid_fresh_seeds.MCAR`:** E164's 38 less E93's 12 off-MCAR ones: `biodeg_80nan`, `vehicle_80nan`,
  `credit-g_40nan`, `electricity_20nan`, `letter_20nan`, `spambase_20nan`, `electricity_40nan`, `electricity_80nan`,
  `kr-vs-kp_20/40/60/80nan`, `credit-g_20/60/80nan`, `biodeg_20/40nan`, `kc2_20/40/80nan`, `pendigits_20/40/60/80nan`,
  `vehicle_20/40nan`.
- **Cells:** 26 × 101/202/303 = 78 runs of H under `experiment = hybrid-thesis-seeds-2026-11-10`, state
  `/scratch/diogoneiss/trident-2026-11-10-hybrid-thesis-seeds` on gorgona8, runner
  `scripts/experiments/hybrid_fresh_seeds.py --study e174`. Offline per pair as E167: TabPFN refit (H's ledger asserted),
  then missForest refit (logged score asserted), into `cells-mf`.
- **Machine (ADR 0035):** everything on gorgona8 (medusa1's GPUs are down since 2026-11-09 22:40 GMT-3). Its trainings log
  to medusa1's store through the tunnel (ADR 0033); three GPU queues, longest first; two offline lanes, a pair 15 minutes
  after its run is done (the store copy's lag).
- **Cost:** about 11–12 h of training by the runner's per-table minutes, the offline pass overlapping it (`spambase_20nan`
  and `letter_20nan` dominate TabPFN); about 14–20 h in all, ending on 2026-11-11.

## Measures and rule (fixed now)

Fold-pair verdict on 15 (seed, fold) differences, 1% SESOI of the opponent's mean.
- **Primary, `table_tabpfn` (decision 6's form) on the 26:**
  - **supported** if worse than H on none and worse than TabPFN on at most 1;
  - **refuted** if worse than H on 2 or more, or worse than TabPFN on 3 or more;
  - **mixed** otherwise.
- **Secondary:** better than H on how many, better than TabPFN on how many; `table_missforest` worse than H on which;
  `crossfit` worse than `table_tabpfn` on which (descriptive after E169).
- Three seeds, fixed; no extension.
- **Read** once with all 78 in, on gorgona8: `uv run --python 3.11 python -B scripts/experiments/hybrid_fresh_seeds.py --study e174 --tally`.
- **What it leads to:** the hybrid column of the thesis table on its primary seeds. No ADR change is proposed by this
  study; a refutation would go to the user as a question on decision 6.
