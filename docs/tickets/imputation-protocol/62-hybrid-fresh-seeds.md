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
