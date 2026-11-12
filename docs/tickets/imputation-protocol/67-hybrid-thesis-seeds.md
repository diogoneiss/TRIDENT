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

## Amendment, 2026-11-11 02:11 GMT-3: paused while the leader is down (nothing read)

- medusa1, which holds the store, stopped answering at about 00:42 GMT-3 (`ssh: No route to host`; gorgona8's
  `leadership.py tick` logged `peer_unreachable` at 01:12). The three `pendigits_20nan` runs (started 00:38) failed at
  02:04 on `set_terminated` ("Connection refused" on port 47731), the three `pendigits_40nan` runs at 02:10. Only
  `FINISHED` runs are ever read, so their orphans on the leader cannot enter a cell.
- The agent stopped the queues with `$S/STOP` at 02:07. Done: 12 of 78 runs and their 12 offline pairs
  (`letter_20nan`, `electricity_20nan`, `electricity_40nan`, `electricity_80nan`).
- `$S/resume.sh` (detached) polls the tunnel's `/health` every five minutes; after two answers in a row it removes STOP
  and relaunches `queue.sh 1–3` (cells with a selection file are skipped, so the six failed ones rerun) and two lanes of
  `$S/cells_resume.sh`, which is `cells.sh` reading only the training lines after the relaunch marker, so the old FAILED
  lines do not skip their reruns.
- Nothing else changes: machine, variants, seeds, rule and the single read. E173's queues still wait for E174's three
  "queue complete" lines, which the relaunched queues write. Its two offline lanes were released at 02:09 by the stopped
  lanes' "complete" lines; they only poll until E173's own runs exist, then may overlap E174's tail on the GPU.

## Amendment, 2026-11-11 23:25 GMT-3: relaunched at 14:06, leader down again (nothing read)

- The leader answered again at 13:57 GMT-3 and `resume.sh` relaunched at 14:06. By 22:23 the queues were complete with
  no failure; the store copy of 22:58 holds every run they finished.
- Correction to the amendment above: only the three `pendigits_40nan` runs reran. The three `pendigits_20nan` runs had
  written all five selection files before failing (the leader went silent during their folds and only their closing
  MLflow call raised), so the queues skipped them; their runs never reached `FINISHED` and cannot be read. Their
  selections were moved to `$S/selection.leader-down-2026-11-11/` at 23:25 so the queues rerun them.
- medusa1 went silent again at about 23:00. `$S/resume2.sh` (detached) waits for two answers five minutes apart, then
  relaunches `queue.sh 1–3` (only `pendigits_20nan` × 3 left) and E173's runners. The `cells_resume.sh` lanes keep
  running meanwhile on the store copy and take the reruns. 75 of 78 runs are done; nothing else changes.

## Amendment, 2026-11-12 16:46 GMT-3: `pendigits_20nan` rerun, its offline cells taken twice (nothing read)

- medusa1 answered again at about 15:10 GMT-3 on 2026-11-12 and `resume2.sh` relaunched at 15:20. The three
  `pendigits_20nan` runs trained from 15:21 to 16:26 and are `FINISHED` on the leader: 78 of 78.
- Their offline cells failed at 16:41–16:42 before computing anything (`FileNotFoundError: pendigits_20nan s101 fold 1:
  2 ledgers`): the lost runs of 2026-11-11 had left their result directories (`20261111_003846`) beside the reruns', and
  the ledger reader wants exactly one. Those directories were moved to `$S/results.leader-down-2026-11-11/`; the reruns'
  ledgers are untouched. `$S/cells_pendigits20.sh` took the three cells again from 16:43, same command and asserts
  (H's ledger against the `FINISHED` run, missForest's logged score).
- Variants, seeds, rule and the single read unchanged.
