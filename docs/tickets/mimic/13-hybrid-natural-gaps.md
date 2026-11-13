# ADR 0031 decision 6's hybrid on the MIMIC-IV tables with natural gaps (E173, T55; pre-registered 2026-11-10 16:45 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** ADR 0031 decision 6 (the deployable TabPFN-side hybrid, `table_tabpfn`) was extended on 2026-11-09 to the
  MIMIC-IV transplant variants (E163, E166), where real gap patterns are copied onto complete rows. The two tables whose
  gaps are natural (E148, [03](03-masked-cells-natural-gaps.md)) have never had a hybrid run. This study asks whether
  decision 6 also holds there, where TRIDENT already leads TabPFN on both tables.
- **The user's go:** "Os dois em sequência" (2026-11-10, about 16:30 GMT-3), with E174
  ([imputation-protocol/67](../imputation-protocol/67-hybrid-thesis-seeds.md)), which runs first.
- **Disclosure:** E148 (seeds 42/7/13, masked checkpoint, not H) read the decoder better than TabPFN on both tables
  (cardio −0.0387, SOFA −0.0108); no hybrid was computed. E171's printout (read) has `table_tabpfn` worse than TabPFN on
  the SOFA `trand` and `mcar` transplant variants. Seeds 101/202/303 and 4545/5656/6767 have no run on these tables.

## Design

- **Variants:** `mimic-cardio_natural` and `mimic-sofa_natural` (11,082 rows each), both primary.
- **Population:** these tables have no induced cells; every score, H's ledger assert and missForest's logged-score
  assert read the masked population (observed cells hidden from the test rows, E148's primary). The hybrid's validation
  cells are hidden observed cells as everywhere else. The helpers gained a `population` argument defaulting to
  `induced`; E171's printout is unchanged by it, and the masked path reproduces E148's logged scores exactly
  (`mimic-sofa_natural` s42 folds 1–2).
- **Cells:** 2 tables × two seed sets, pre-registered together: `--study e173` (101/202/303, tag
  `mimic-hybrid-natural-2026-11-10`, state `/scratch/diogoneiss/trident-2026-11-10-mimic-natural`) and `--study e173b`
  (4545/5656/6767, tag `mimic-hybrid-natural-b-2026-11-10`, state `…-mimic-natural-b`); 12 runs of H, runner
  `scripts/experiments/mimic_hybrid.py`. Offline per pair: TabPFN refit (H's ledger asserted), then missForest refit
  (logged score asserted), into `cells-mf`.
- **Machine (ADR 0035):** everything on gorgona8, its trainings logging to medusa1's store through the tunnel. Three GPU
  queues start once E174's three are complete; two offline lanes once E174's are, a pair 15 minutes after its run.
- **Cost:** H took 33–36 minutes a run in E148; with E174's offline pass beside it about 4–6 h of training, the offline
  pass about 3–5 h more; ending on 2026-11-11.

## Measures and rule (fixed now)

Fold-pair verdict on 15 (seed, fold) differences per seed set, 1% SESOI of the opponent's mean.
- **Per seed set, E163's rule on `table_tabpfn` over the two tables:** refuted if better than H on neither or worse than
  H on either; supported if, otherwise, worse than TabPFN on neither; mixed otherwise.
- **E173 primary:** **supported** if both seed sets read supported; **refuted** if either reads refuted; **mixed**
  otherwise.
- **Secondary:** `table_tabpfn` better than TabPFN on which; `table_missforest` worse than H on which; `crossfit` worse
  than `table_tabpfn` on which.
- Three seeds per set, fixed; no extension.
- **Read** once with all 12 in, on gorgona8, both commands in one sitting:
  `TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data uv run --python 3.11 python -B scripts/experiments/mimic_hybrid.py --study e173 --tally`,
  then the same with `--study e173b`.
- **What it leads to:** whether decision 6's scope may cover MIMIC-IV tables with natural gaps; the user's call.

## Amendment, 2026-11-11 23:25 GMT-3: first runs lost to the leader's outage (nothing read)

- E174 ([67](../imputation-protocol/67-hybrid-thesis-seeds.md)) was paused by medusa1's first outage (about 00:42 to
  13:57 GMT-3) and its queues completed at 22:23; E173's queues started at 22:26 with `mimic-cardio_natural` s101/s202
  and `mimic-sofa_natural` s101.
- medusa1 went silent again at about 23:00. `mimic-sofa_natural` s101 failed at 23:19 on its closing MLflow call; the
  agent set STOP in both states and stopped the other three trainers at 23:24, since none could finish on the leader.
  The three selections written by those runs were moved to `selection.leader-down-2026-11-11/`, so every E173 run reruns.
- E174's `resume2.sh` relaunches E173 once the leader answers twice: `queue_resume.sh 1–3` (same runner and cells, start
  after E174's three relaunched queues complete) and two lanes of `cells_resume.sh` (`cells.sh` reading only the lines
  after the relaunch marker, starting once E174's lanes are complete). Seeds, tables, rule and the single read unchanged.

## Amendment, 2026-11-12 16:03 GMT-3: seed set 4545/5656/6767 moves to medusa1 (nothing read, no E173 run exists)

- medusa1 answered again at about 15:10 GMT-3 on 2026-11-12 (rebooted; both GPUs pass a CUDA check) and `resume2.sh`
  relaunched at 15:20. The user: "a medusa voltou, continue. voce pode disparar filas dentro dela".
- **Split by seed set (ADR 0035):** `e173` (101/202/303) stays wholly on gorgona8, after E174's three `pendigits_20nan`
  reruns; `e173b` (4545/5656/6767) runs wholly on medusa1, training and offline cells, from now. Every comparison of the
  rule is inside one seed set, so every pair stays on one machine. `mimic_hybrid.HOSTS["e173b"]` is now `medusa`.
- medusa1: state `/scratch/diogoneiss/trident-2026-11-10-mimic-natural-b` there, three queues and two offline lanes on
  GPU 1 only (GPU 0 fell off the bus twice on 2026-11-09). On gorgona8 the `-b` state holds a STOP file, so its lanes
  drop that seed set and its queues end on the host check ("queue read FAILED", expected).
- **Read:** `--study e173 --tally` on gorgona8 and `--study e173b --tally` on medusa1, in one sitting once all 12 are in.
  Tables, seeds and rule unchanged.

## Amendment, 2026-11-12 16:46 GMT-3: the lost runs' result directories moved aside on gorgona8 (nothing read)

- E174's `pendigits_20nan` cells failed on "2 ledgers" because a lost run's result directory sat beside its rerun's
  (see [67](../imputation-protocol/67-hybrid-thesis-seeds.md)). The same would have hit `e173`'s cells, so the four
  directories written on 2026-11-11 (cardio s101/s202/s303, SOFA s101) were moved to
  `results.leader-down-2026-11-11/` in the state before any `e173` cell ran. `e173`'s trainings started at 16:32.

## Amendment, 2026-11-12 21:08 GMT-3: medusa1 rebooted at 17:38, one `e173b` cell taken again (nothing read)

- `e173b`'s six runs of H finished on medusa1 by 17:11 GMT-3 and are `FINISHED` in the leader's store; five of its six
  offline cells were done by 17:33. medusa1 rebooted at 17:38 (`who -b`; cause unknown), two minutes into the cell of
  `mimic-sofa_natural` s5656, which left only fold 1's files in `cells/` and no line in the log.
- The same `cells.sh 1` was launched again at 21:06; it skips the pairs already in `cells-mf` and recomputes that cell
  whole (every fold's files are rewritten). Another user's job now holds about 27 GB of each of medusa1's cards; the
  cell runs in the rest of GPU 1.
- `e173` on gorgona8 was not interrupted: its six runs finished by 19:29 (the leader's MLflow server was back before any
  of them closed), five of six cells done, the last running. Read and rule unchanged.

## Outcome (read once 2026-11-12 21:22 GMT-3: supported; written 2026-11-12 21:23 GMT-3)

Both tallies read in one sitting, after all 12 pairs' cells were in: `--study e173 --tally` on gorgona8 at commit
8c8bc81, then `--study e173b --tally` on medusa1 at 27ed35e (the same code; only documents changed between them). No
FAILED line in either state after the relaunch; the H-ledger and missForest asserts held on every pair.

- **Primary, supported.** On each seed set `table_tabpfn` (ADR 0031 decision 6's deployable hybrid, TabPFN side) is
  better than H on `mimic-cardio_natural` (−0.0025 [−0.0049, −0.0000] on 101/202/303; −0.0043 [−0.0069, −0.0017] on
  4545/5656/6767), equivalent to H on `mimic-sofa_natural` (−0.0017, −0.0007), and worse than TabPFN on neither.
- **Size:** the gain over H is 0.3% to 0.5% of H's score on the cardio table and none on SOFA. The hybrid is better than
  TabPFN on both tables in both sets (−0.0462 and −0.0183; −0.0432 and −0.0215), a lead H already had (E148).
- **Secondary.** `table_missforest` is worse than H on none (better on cardio, equivalent on SOFA, both sets).
  `crossfit` is worse than `table_tabpfn` on none (equivalent on both tables, both sets).
- **What it leads to:** whether ADR 0031 decision 6's scope covers the MIMIC-IV tables with natural gaps is the user's
  call. The evidence: never worse than either side on two tables and two seed sets, with a small gain over H on one.
- **The user's decision, 2026-11-12 22:24 GMT-3 ("sim, extenda!"):** decision 6's scope now covers these two tables; recorded in
  [ADR 0031](../../adr/0031-an-opt-in-offline-hybrid-output-with-tabpfn.md) (decision 6, "On clinical tables with
  natural gaps").

Printout of `--study e173 --tally` (gorgona8), unedited:

```
- `mimic-cardio_natural` (primary): H 0.7957, tabpfn 0.8394, missforest 0.8221, table_tabpfn 0.7932, table_missforest 0.7932, pick 0.7927, picked_missforest 0.5333, crossfit 0.7929, crossfit_missforest 0.4667
  - table_missforest − H -0.0025 [-0.0041, -0.0009] better
  - table_missforest − missforest -0.0290 [-0.0324, -0.0256] better
  - table_tabpfn − H -0.0025 [-0.0049, -0.0000] better
  - table_tabpfn − tabpfn -0.0462 [-0.0528, -0.0397] better
  - pick − H -0.0029 [-0.0046, -0.0012] better
  - pick − table_tabpfn -0.0005 [-0.0014, +0.0005] equivalent
  - pick − table_missforest -0.0004 [-0.0011, +0.0002] equivalent
  - crossfit − H -0.0028 [-0.0046, -0.0011] better
  - crossfit − table_tabpfn -0.0004 [-0.0013, +0.0006] equivalent
  - the pick took missForest on 8 of 15 folds, the cross-fit pick on 7
- `mimic-sofa_natural` (primary): H 0.8735, tabpfn 0.8902, missforest 0.9346, table_tabpfn 0.8718, table_missforest 0.8739, pick 0.8718, picked_missforest 0.0000, crossfit 0.8724, crossfit_missforest 0.1333
  - table_missforest − H +0.0004 [-0.0002, +0.0010] equivalent
  - table_missforest − missforest -0.0607 [-0.0674, -0.0540] better
  - table_tabpfn − H -0.0017 [-0.0039, +0.0005] equivalent
  - table_tabpfn − tabpfn -0.0183 [-0.0217, -0.0150] better
  - pick − H -0.0017 [-0.0039, +0.0005] equivalent
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0021 [-0.0043, +0.0001] equivalent
  - crossfit − H -0.0012 [-0.0031, +0.0008] equivalent
  - crossfit − table_tabpfn +0.0005 [-0.0006, +0.0017] equivalent
  - the pick took missForest on 0 of 15 folds, the cross-fit pick on 2

E173 PRIMARY, seed set 101/202/303: `table` (TabPFN side) on mimic-cardio_natural, mimic-sofa_natural: better than H on ['mimic-cardio_natural'], worse than H on none, worse than TabPFN on none -> supported
E173 SECONDARY: `table_tabpfn` better than TabPFN on ['mimic-cardio_natural', 'mimic-sofa_natural']; `table_missforest` worse than H on none; `crossfit` worse than `table_tabpfn` on none
E173 reads supported if both seed sets read supported, refuted if either reads refuted, mixed otherwise
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```

Printout of `--study e173b --tally` (medusa1), unedited:

```
- `mimic-cardio_natural` (primary): H 0.7935, tabpfn 0.8324, missforest 0.8206, table_tabpfn 0.7892, table_missforest 0.7906, pick 0.7898, picked_missforest 0.4667, crossfit 0.7894, crossfit_missforest 0.4000
  - table_missforest − H -0.0029 [-0.0049, -0.0010] better
  - table_missforest − missforest -0.0301 [-0.0344, -0.0257] better
  - table_tabpfn − H -0.0043 [-0.0069, -0.0017] better
  - table_tabpfn − tabpfn -0.0432 [-0.0493, -0.0371] better
  - pick − H -0.0037 [-0.0063, -0.0012] better
  - pick − table_tabpfn +0.0005 [-0.0013, +0.0024] equivalent
  - pick − table_missforest -0.0008 [-0.0020, +0.0004] equivalent
  - crossfit − H -0.0041 [-0.0067, -0.0016] better
  - crossfit − table_tabpfn +0.0002 [-0.0006, +0.0010] equivalent
  - the pick took missForest on 7 of 15 folds, the cross-fit pick on 6
- `mimic-sofa_natural` (primary): H 0.8779, tabpfn 0.8988, missforest 0.9344, table_tabpfn 0.8772, table_missforest 0.8787, pick 0.8772, picked_missforest 0.0000, crossfit 0.8776, crossfit_missforest 0.2667
  - table_missforest − H +0.0008 [-0.0005, +0.0020] equivalent
  - table_missforest − missforest -0.0558 [-0.0615, -0.0501] better
  - table_tabpfn − H -0.0007 [-0.0026, +0.0012] equivalent
  - table_tabpfn − tabpfn -0.0215 [-0.0270, -0.0161] better
  - pick − H -0.0007 [-0.0026, +0.0012] equivalent
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0014 [-0.0040, +0.0011] equivalent
  - crossfit − H -0.0003 [-0.0018, +0.0011] equivalent
  - crossfit − table_tabpfn +0.0004 [-0.0009, +0.0016] equivalent
  - the pick took missForest on 0 of 15 folds, the cross-fit pick on 4

E173 PRIMARY, seed set 4545/5656/6767: `table` (TabPFN side) on mimic-cardio_natural, mimic-sofa_natural: better than H on ['mimic-cardio_natural'], worse than H on none, worse than TabPFN on none -> supported
E173 SECONDARY: `table_tabpfn` better than TabPFN on ['mimic-cardio_natural', 'mimic-sofa_natural']; `table_missforest` worse than H on none; `crossfit` worse than `table_tabpfn` on none
E173 reads supported if both seed sets read supported, refuted if either reads refuted, mixed otherwise
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```
