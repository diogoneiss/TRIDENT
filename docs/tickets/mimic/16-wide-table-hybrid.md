# The deployable hybrid on `mimic-wide`: safe and still a gain where TabPFN alone is weak? (E178, T57; pre-registered 2026-11-13 08:28 GMT-3)

This ticket was written and committed before the study's first cell.

- **Why:** ADR 0031 decision 6 (the opt-in offline hybrid, `table_tabpfn`: H's fills and TabPFN's blended with one
  numerical weight and one categorical pick per table, chosen on the fold's hidden validation cells) covers the two
  11,082-stay MIMIC-IV tables, under transplanted patterns (E163, E166) and natural gaps (E173). On `mimic-wide`, six times larger, E176 ([mimic/14](14-wide-table.md)) found
  TabPFN v2 alone behind the decoder and, on the real patterns, behind the bar. This study asks whether the hybrid
  is still never worse than H there, and whether it still gains anything.
- **The user's go:** "2 e 1, vá" (2026-11-13); option 1 was this hybrid.
- **Disclosure:**
  - E176's scores on these variants and seeds were read (decoder − TabPFN −0.0991 natural, −0.0577 `tmatch`,
    −0.0044 `mcar`), with readouts M and K. H, the deployable readout this study blends, was not run on this table.
  - With TabPFN that far behind, the hybrid may choose H's fill nearly everywhere and read equivalent to H. E163's
    rule calls that refuted (no gain). Said now: a reading refuted by "better than H on neither, worse on neither"
    means the hybrid is harmless and unneeded on this table, and decision 6's scope does not extend to it.
  - No cell of this study has run.

## Design

- **Cells:** H (`deployable_hybrid.train`: `--decode_checkpoint loss --decode_calibration none`, the hidden observed
  validation cells read in place) on `mimic-wide_natural`, `mimic-wide-donor_tmatch_20nan` and
  `mimic-wide-donor_mcar_20nan`; seeds 42, 7, 13, E176's; 9 runs, five folds each. Study `e178` of
  `scripts/experiments/mimic_hybrid.py`, tag `mimic-wide-hybrid-2026-11-13`, state
  `/scratch/diogoneiss/trident-2026-11-13-mimic-wide-hybrid`.
- **Offline pass:** `mimic_hybrid.py --study e178 cells <variant> <seed>` per run: TabPFN v2 refit on the fold (H's
  ledger asserted equal to the run's H), then missForest refit (asserted equal to the run's logged missForest). The
  natural table is scored on its masked population, the donor variants on their induced cells.
- **Machine (ADR 0035, the split of studies between servers):** every run and the offline pass on medusa1; two
  queues, one a GPU.
- **Cost:** E176's runs took about 80 minutes on the donor table and 105 on the natural one; H's are the same
  shape, less where the baseline cache already holds E176's folds: at most 14 GPU-hours, about 7 h in two queues.
  The offline pass is about 8 minutes of TabPFN a run plus missForest's refit, not timed.

## Measures and rule (fixed now)

`impute_score` per fold on each variant's scored cells, lower is better; fold-pair rule (95% t interval, every seed
agreeing), equivalence margin 1% of the opponent's mean (`missforest_hybrid.SESOI`). E163's rule, unchanged, on the
primaries `mimic-wide_natural` and `mimic-wide-donor_tmatch_20nan`:

- **Refuted** if `table_tabpfn` is better than H on neither primary, or worse than H on either.
- **Supported** if, otherwise, it is worse than TabPFN on neither.
- **Mixed** otherwise.
- **Secondary:** the same lines on `mcar`; the missForest side (`table_missforest`), the pick and the cross-fit pick
  as `--tally` prints them; how often each side is chosen.
- Three seeds, fixed; no extension.
- **Read** once with the 9 runs and their cells in, on medusa1:
  `TRIDENT_DATASETS_ROOT=/scratch/diogoneiss/trident-data uv run --python 3.11 python -B scripts/experiments/mimic_hybrid.py --study e178 --tally`
- **What it leads to:** whether ADR 0031 decision 6's scope reaches a clinical table of this size, the user's call.

## Amendment (2026-11-13 11:33 GMT-3): medusa1's GPU 0 fell off the bus, the rest runs on GPU 1

Written before any reading; no score of this study has been read.

- **What happened:** at 11:31:38 GMT-3 on 2026-11-13 medusa1's GPU 0 stopped answering (`nvidia-smi`: "Unable to
  determine the device handle for GPU0: 0000:41:00.0: Unknown Error"), its third failure since 2026-11-09. It had
  carried E177's queue and this study's queue 1 for three hours; the failure came two minutes after the first offline
  cell (TabPFN on `mimic-wide_natural` seed 42) started beside a training on it. Whether that load caused it is not known.
  GPU 1 kept working.
- **What it took down:** the H run of `mimic-wide_natural` seed 13, one minute in (FAILED in the store); the offline
  cells of `mimic-wide_natural` seed 42 and `mimic-wide-donor_mcar_20nan` seed 42 (no TabPFN fit finished). Queue 1
  then started H on `mimic-wide-donor_mcar_20nan` seed 13 with no CUDA device, on the CPU; it was killed after one
  minute, at 11:32:38, and its run stays RUNNING in the store, unfinished. The readers take FINISHED runs only, so
  neither run is read.
- **Seven H runs were finished or running on GPU 1 and stand:** six FINISHED before the failure, three of them on GPU 0
  (`mimic-wide_natural` 42, `mcar` 42, `tmatch` 7), and `tmatch` seed 13 on GPU 1.
- **Change:** the two lost H runs are rerun whole on GPU 1 after queue 2 ends, and every offline cell runs on GPU 1,
  one at a time (`finish_gpu1.sh` in the state directory, written before it ran). The design said one queue a GPU;
  both cards are RTX 5090s on the same machine, and H and the hybrid it is compared with come from the same run, so
  no contrast crosses cards. Seeds, cells, measures and rule unchanged. The wall clock grows by about two hours.

## Amendment (2026-11-13 19:09 GMT-3): GPU 1 was not usable either; the study is paused, nothing read

- **The first amendment was wrong about GPU 1.** Processes already on it kept running, but since GPU 0 fell no new
  process on medusa1 can start CUDA (`torch.cuda.is_available()` is False with `CUDA_VISIBLE_DEVICES=1`), as after
  the failure of 2026-11-09. This was not checked before `finish_gpu1.sh` was started.
- **What that did:** every offline cell failed in three seconds (TabPFN refuses more than 1,000 rows on the CPU), so
  no cell table exists. The rerun of H on `mimic-wide_natural` seed 13 started at 11:58 GMT-3 with no device and
  trained on the CPU for seven hours, until it was noticed and killed at 19:08 GMT-3 on 2026-11-13. Its run stays
  RUNNING in the store, as does the one-minute `mcar` seed 13 run of 11:31; neither is FINISHED, so neither is read.
  Its `selection/mimic-wide_natural_s13` files are from the CPU and must be deleted before that run is redone.
- **State:** 7 of 9 H runs FINISHED on a GPU (all but `mimic-wide_natural` and `mcar` at seed 13); 0 of 9 offline
  cells; `STOP` is in the state directory and nothing of this study is running. No score has been read.
- **To finish:** medusa1 needs a reboot for CUDA to come back; then the two H runs and the nine cells, by the
  ticket's design. Moving them to gorgona8 would split a study across machines by other than seed (ADR 0035) and is
  the user's call.

## Amendment (2026-11-13 20:17 GMT-3): the rest runs on gorgona8, on the user's call; a departure from ADR 0035

Written before any reading and before any cell on gorgona8; no score of this study has been read.

- **The user's go:** asked "Posso soltar a 1 no gorgona8?" (option 1: finish E178 there, said to leave ADR 0035, the
  rule that keeps a study on one machine or splits it by seed), the user answered "sim" (2026-11-13). medusa1 was
  not rebooted (boot time still 2026-11-12 17:38) and no new process there can start CUDA.
- **What moves:** the two missing H runs (`mimic-wide_natural` and `mimic-wide-donor_mcar_20nan` at seed 13) and all
  nine offline cells run on gorgona8 (RTX 3090 Ti), state at the same path there. The seven finished runs'
  `results/` and `selection/` are copied from medusa1 with rsync, leaving out `selection/mimic-wide_natural_s13` (the
  CPU run's files) and the two unfinished runs' `results/`; their MLflow artifacts were already on gorgona8 (the
  leadership copy). The new runs log to the leader's store through the tunnel and carry `host = gorgona8`.
- **The departure:** H is trained on medusa1 for seven (variant, seed) cells and on gorgona8 for two, a split that is
  not by seed (at seed 13 `tmatch` is medusa1's, the other two gorgona8's).
  - Every line of the rule is a fold pair inside one cell: `table_tabpfn`, H and TabPFN are scored on the same
    folds' cells, with H's fills from one run. No pair has one arm trained on each machine.
  - TabPFN and missForest are refit on gorgona8 for all nine cells, so in seven cells H is medusa1's and the other
    side gorgona8's, as in E169's offline pass (moved the same way on 2026-11-09). missForest's refit is asserted
    equal to the run's logged score; TabPFN's refit has no such check.
  - On `mimic-wide_natural` and `mcar` the seed-agreement line sets seeds 42 and 7 from medusa1 beside seed 13 from
    gorgona8, and levels across seeds mix the two machines (E86: they differ per fold).
- **Unchanged:** seeds, cells, measures, rule, three seeds with no extension. The single read is the same command,
  run on gorgona8, where the cell tables are.
- **Guards:** `torch.cuda.is_available()` was True in a new process on gorgona8 before the launch, and each run
  checks it again before it starts; the waiter fires on a FAILED line in `progress.log`.
- **The store:** the FAILED run and the two RUNNING orphans of 11:30 to 11:58 stay as they are. With the reruns
  FINISHED, each (variant, seed) has exactly one FINISHED run, which `cells` requires.
- **Code:** the new runs carry this amendment's commit; it differs from the seven runs' 1085c6e in documents and in
  `mimic_hybrid.py`'s docstring only.

## Outcome (2026-11-14 14:32 GMT-3; read once 2026-11-14 14:30 GMT-3, on gorgona8 at 48b6cbe, all 9 runs and their cells in)

Seven H runs trained on medusa1 on 2026-11-13 and two on gorgona8 (20:17 GMT-3 on 2026-11-13 to 01:09 on 2026-11-14);
the nine offline cells ran on gorgona8 from 20:19 GMT-3 on 2026-11-13 to 14:29 on 2026-11-14, with no FAILED line
after the move. The leader's store holds one FINISHED parent run for each (variant, seed), seven tagged
`host = medusa` and two `host = gorgona8`; the FAILED run and the two RUNNING orphans of the second amendment were
not read. `mimic_hybrid.py` is unchanged between the third amendment's commit and the read's. The printout, unedited:

```
- `mimic-wide_natural` (primary): H 0.6653, tabpfn 0.7644, missforest 0.6968, table_tabpfn 0.6653, table_missforest 0.6654, pick 0.6654, picked_missforest 0.0667, crossfit 0.6653, crossfit_missforest 0.0000
  - table_missforest − H +0.0000 [-0.0000, +0.0001] equivalent
  - table_missforest − missforest -0.0315 [-0.0326, -0.0303] better
  - table_tabpfn − H +0.0000 [+0.0000, +0.0000] equivalent
  - table_tabpfn − tabpfn -0.0990 [-0.0999, -0.0981] better
  - pick − H +0.0000 [-0.0000, +0.0001] equivalent
  - pick − table_tabpfn +0.0000 [-0.0000, +0.0001] equivalent
  - pick − table_missforest +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − H +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - the pick took missForest on 1 of 15 folds, the cross-fit pick on 0
- `mimic-wide-donor_tmatch_20nan` (primary): H 0.9087, tabpfn 0.9662, missforest 1.0177, table_tabpfn 0.9087, table_missforest 0.9087, pick 0.9087, picked_missforest 0.0000, crossfit 0.9087, crossfit_missforest 0.0000
  - table_missforest − H +0.0000 [+0.0000, +0.0000] equivalent
  - table_missforest − missforest -0.1089 [-0.1222, -0.0956] better
  - table_tabpfn − H +0.0000 [+0.0000, +0.0000] equivalent
  - table_tabpfn − tabpfn -0.0575 [-0.0591, -0.0559] better
  - pick − H +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − H +0.0000 [+0.0000, +0.0000] equivalent
  - crossfit − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - the pick took missForest on 0 of 15 folds, the cross-fit pick on 0
- `mimic-wide-donor_mcar_20nan`: H 0.7178, tabpfn 0.7216, missforest 0.8147, table_tabpfn 0.7154, table_missforest 0.7179, pick 0.7154, picked_missforest 0.0000, crossfit 0.7165, crossfit_missforest 0.4000
  - table_missforest − H +0.0001 [-0.0001, +0.0004] equivalent
  - table_missforest − missforest -0.0969 [-0.1017, -0.0921] better
  - table_tabpfn − H -0.0024 [-0.0034, -0.0014] better
  - table_tabpfn − tabpfn -0.0062 [-0.0069, -0.0054] better
  - pick − H -0.0024 [-0.0034, -0.0014] better
  - pick − table_tabpfn +0.0000 [+0.0000, +0.0000] equivalent
  - pick − table_missforest -0.0025 [-0.0035, -0.0014] better
  - crossfit − H -0.0012 [-0.0022, -0.0002] better
  - crossfit − table_tabpfn +0.0012 [+0.0002, +0.0021] worse
  - the pick took missForest on 0 of 15 folds, the cross-fit pick on 6

E178 PRIMARY `table` (TabPFN side) on mimic-wide_natural, mimic-wide-donor_tmatch_20nan: better than H on none, worse than H on none, worse than TabPFN on none -> refuted
(fold-pair t, seed agreement; SESOI 1% of the opponent's mean)
```

- **Reading, by the ticket's own rule: refuted**, by the clause "better than H on neither primary". On
  `mimic-wide_natural` and on `mimic-wide-donor_tmatch_20nan` the hybrid `table_tabpfn` is equivalent to H (+0.0000
  on both) and better than TabPFN (−0.0990, −0.0575): with TabPFN v2 that far behind, the rule chooses H's fill. The
  disclosure said this beforehand: the hybrid is harmless and unneeded on this table, and decision 6's scope does
  not extend to it.
- **Secondary:**
  - Under MCAR (`mimic-wide-donor_mcar_20nan`), where E176 found the two sides level, `table_tabpfn` is better than
    H (−0.0024, 0.3% of H's score) and than TabPFN (−0.0062).
  - The missForest side (`table_missforest`) is equivalent to H on all three variants and better than missForest on
    all three (−0.0315, −0.1089, −0.0969).
  - The pick between sides took missForest on 1 of 45 folds. The cross-fit pick took it on 6 of 15 `mcar` folds and
    is worse than `table_tabpfn` there (+0.0012), still better than H.
- **Limits:** the departure of the third amendment stands: H is medusa1's in seven cells and gorgona8's in two, and
  the other side was refit on gorgona8 in all nine. Every line of the rule is a fold pair inside one cell.
- **What it leads to:** ADR 0031 decision 6 (the opt-in offline hybrid) is not extended to `mimic-wide`; nothing in
  the ADR changes. On this table the hybrid costs a TabPFN refit a fold and returns H on the real gap patterns.
