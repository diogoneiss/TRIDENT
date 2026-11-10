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
