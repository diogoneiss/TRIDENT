# Today's default on fresh seeds at 20nan (E51; pre-registered 2026-10-19)

This ticket was written and committed before the first GPU cell.

- **Why:** on 2026-10-19 the user gave the GPU's idle hours until 08:00 GMT-3 on 10-20 to rerun
  the variants that had not run under today's defaults for the longest time.
- **Which variants those are, read from the store** (finished parents, `is_mirror = false`):
  - **The nine `_20nan` variants.** Their latest runs of today's default are E34 (101/202/303,
    2026-10-03) and E35 (42/7/13, 10-04). For pendigits_20nan and letter_20nan, whose default is
    the bigger encoder (ADR 0018), the only such runs are E38's ALL arm on 101/202/303 (10-06).
  - **Every variant at 40nan and above** ran between 10-16 and 10-19 (E40, E44 to E47), so none of
    them is in this study.
- **Why fresh seeds:** a rerun of today's default on a seed set it has already run reproduces the
  same K fold by fold; E50's integrity check found a difference of 0. Only an unused seed set
  carries new information.
  - **The seeds:** 1001, 2002 and 3003. None of them is in the store. 707/808/909 was retired by
    E44.
- **What is new in these runs** besides the seeds:
  - the strong tier is live (ADR 0021), not backfilled;
  - ADR 0027's automatic calibration resolves to none at 20nan, logged by tag.
  - At 20nan the K readout is the same configuration as E34 and E35's, except on pendigits and
    letter, where it is E38 ALL's.
- **No code change.** New files are the runner, the report and the queue script. `src/` was not
  touched while E50 was live.

## Design (fixed before the first cell)

- **Cells:** the nine `_20nan` variants (electricity, letter, pendigits, spambase, kr-vs-kp, biodeg,
  credit-g, vehicle, kc2) × seeds 1001/2002/3003. 27 cells, tag
  `experiment=default-refresh-2026-10-20`.
- **The configuration** is today's default as a default run reads it (`load_hyperparameters`; the
  promoted file where one exists). It runs with `--score_checkpoints` and `--score_calibrated`, as
  E44 ran, so H, M and KC are logged beside K. Five folds. Runner:
  `scripts/experiments/default_refresh_run.py`.
- **The baseline cache was warmed on the CPU** (`warm_baseline_cache.py`, 3 lanes at 4 threads)
  while E50 held the GPU. Started 22:08 GMT-3 on 10-19.
- **Running it:**
  - Four queues (`scripts/experiments/run_default_refresh_queue.sh`), longest cells first, with
    E35's and E38's warm-cache cell times. About 262 minutes a queue.
  - Queue i starts once i of E50's four queues have ended, so the GPU keeps four trainers. A
    launcher in the state directory does it (`launch_after_e50.sh`).
- **The deadline:** every cell must end by 08:00 GMT-3 on 2026-10-20. A cell starts only if the
  current time plus 1.3 × its estimated minutes falls before 08:00. A skipped cell is logged as
  skipped.
- **Pre-launch check:** kc2_20nan on seed 42 with `--no-mlflow` must equal E35's run fold by fold
  on K, both populations (largest difference at most 1e-6).

## Measures

All on `impute_score`, induced population, over 15 fold pairs (3 seeds × 5 folds), with the
verdict rule and the 1% SESOI (`report_common`), read by
`scripts/experiments/default_refresh_report.py --tally`:

- **Primary:** K − bar (the weak tier's best).
- **Secondary:**
  - K − the strong tier's best, live;
  - K against missForest, the best of all and the per-type bar;
  - H − bar and M − bar (ADR 0024's deployable readouts);
  - K − H, K − M;
  - KC − K, a check of ADR 0027's threshold at 20nan;
  - minutes a cell.
- **The earlier calls** are K − bar and K − strong on 101/202/303 and on 42/7/13, from
  `thesis_table` with E41's per-fold CSVs. 42/7/13 is not compared on pendigits_20nan and
  letter_20nan (the old encoder).
- **Not read tonight:** E42's tuned bar has no pass on these folds.

## Reading, fixed now

- **Per variant:**
  - **a reversal:** the fresh call is better where an earlier set's is worse, or the other way;
  - **a drift:** any other change, such as better to no difference.
- **Primary:**
  - **holds:** K − bar shows no reversal on any complete variant;
  - **drifts:** at least one reversal, listed by name.
- **What each reading leads to:** no default changes from this study.
  - **holds:** the 20nan rows of the thesis table gain a third seed set.
  - **drifts:** each reversal gets a ticket.
- **Incomplete variants:** a variant without all three seeds by the deadline is reported as
  incomplete and not read.

## Launch

- **Checked before launch, with `--no-mlflow`, passed (2026-10-19, about 22:30 GMT-3):**
  - kc2_20nan on seed 42 equals E35's run fold by fold on K, both populations (largest difference
    0).
  - The run scored every baseline live, the strong tier included (condexp, ice, missforest and
    softimpute beside the weak tier). The configuration came from the task defaults.
- **The cache warm** started at 22:08 GMT-3 on 10-19, longest variants first, one lane a seed,
  logging to `warm/progress.log` in the state directory. electricity_20nan took 13 minutes a seed.
- **The launcher** (`launch_after_e50.sh` in `/scratch2/diogoneiss/trident-2026-10-20-default-refresh/`)
  starts refresh queue i once i of E50's queues have ended and the cache is warm (or at 02:15
  GMT-3), on the commit of this ticket.
