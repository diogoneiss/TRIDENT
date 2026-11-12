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
