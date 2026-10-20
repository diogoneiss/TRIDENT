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

## Outcome (2026-10-20, 08:13 GMT-3)

- **All 27 cells finished, none failed, none skipped.**
  - The first queue started at 01:14 GMT-3 and the last ended at 06:35 GMT-3, before the 08:00
    deadline.
  - The queue with both electricity cells ran about 20 minutes past its estimate: electricity
    took 130 minutes a cell, against 101 estimated.
- **The waiter that should have triggered this reading fired at 08:10, not 06:35.** Its
  `pgrep -f run_default_refresh_queue` matched the waiter's own command line, so only its 08:10
  fallback ended it. The cells are unaffected.

**The printout** (`scripts/experiments/default_refresh_report.py --tally`, unedited):

```text
complete: 9 of 9; not read (incomplete by the deadline): none

K - bar, fresh seeds 1001/2002/3003: better 7 / no difference 1 / worse 1 (fold-pair t, seed agreement)
  electricity_20nan: fresh worse; earlier {'101/202/303': 'worse', '42/7/13': 'none'}
  letter_20nan: fresh better; earlier {'101/202/303': 'better'}
  pendigits_20nan: fresh better; earlier {'101/202/303': 'better'}
  spambase_20nan: fresh better; earlier {'101/202/303': 'none', '42/7/13': 'better'}
  kr-vs-kp_20nan: fresh better; earlier {'101/202/303': 'better', '42/7/13': 'better'}
  biodeg_20nan: fresh better; earlier {'101/202/303': 'better', '42/7/13': 'better'}
  credit-g_20nan: fresh better; earlier {'101/202/303': 'better', '42/7/13': 'better'}
  vehicle_20nan: fresh better; earlier {'101/202/303': 'better', '42/7/13': 'better'}
  kc2_20nan: fresh none; earlier {'101/202/303': 'none', '42/7/13': 'none'}
Primary: K - bar holds: reversals none (fold-pair t, seed agreement)
  changes short of a reversal: ['electricity_20nan (42/7/13: none -> worse)', 'spambase_20nan (101/202/303: none -> better)']

K - strong, fresh seeds 1001/2002/3003: better 6 / no difference 3 / worse 0 (fold-pair t, seed agreement)
  electricity_20nan: fresh none; earlier {'101/202/303': 'none', '42/7/13': 'none'}
  letter_20nan: fresh better; earlier {'101/202/303': 'better'}
  pendigits_20nan: fresh better; earlier {'101/202/303': 'better'}
  spambase_20nan: fresh none; earlier {'101/202/303': 'none', '42/7/13': 'none'}
  kr-vs-kp_20nan: fresh better; earlier {'101/202/303': 'better', '42/7/13': 'better'}
  biodeg_20nan: fresh better; earlier {'101/202/303': 'better', '42/7/13': 'better'}
  credit-g_20nan: fresh better; earlier {'101/202/303': 'none', '42/7/13': 'none'}
  vehicle_20nan: fresh better; earlier {'101/202/303': 'better', '42/7/13': 'better'}
  kc2_20nan: fresh none; earlier {'101/202/303': 'none', '42/7/13': 'none'}
Secondary: K - strong: reversals none
  changes short of a reversal: ['credit-g_20nan (101/202/303: none -> better)', 'credit-g_20nan (42/7/13: none -> better)']

H - bar: better 7 / no difference 1 / worse 1 (fold-pair t, seed agreement)
M - bar: better 7 / no difference 1 / worse 1 (fold-pair t, seed agreement)
K - H: better 2 / no difference 6 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
K - M: better 2 / no difference 6 / equivalent 1 / worse 0 (fold-pair t, seed agreement)
KC - K: better 3 / no difference 2 / equivalent 3 / worse 1 (fold-pair t, seed agreement)
K - per_type: better 7 / no difference 1 / worse 1 (fold-pair t, seed agreement)
KC better than K at 20nan (ADR 0027 keeps calibration off below 0.3): ['letter_20nan', 'pendigits_20nan', 'credit-g_20nan']
```

**Per variant** (built by script from the same runs). Each cell is a mean difference over 15 fold
pairs of the induced `impute_score`, with its percent of K and the call:

| variant | K | K − bar | K − strong | H − bar | KC − K | min/cell |
|---|---|---|---|---|---|---|
| `electricity_20nan` | 0.6130 | +0.0138 (+2.3%) worse | +0.0003 (+0.0%) none | +0.0182 (+3.0%) worse | +0.0001 (+0.0%) equivalent | 130 |
| `letter_20nan` | 0.4328 | -0.0398 (-9.2%) better | -0.0301 (-7.0%) better | -0.0396 (-9.2%) better | -0.0091 (-2.1%) better | 109 |
| `pendigits_20nan` | 0.3194 | -0.0310 (-9.7%) better | -0.0143 (-4.5%) better | -0.0292 (-9.1%) better | -0.0090 (-2.8%) better | 51 |
| `spambase_20nan` | 0.8493 | -0.0160 (-1.9%) better | -0.0082 (-1.0%) none | -0.0185 (-2.2%) better | +0.0050 (+0.6%) worse | 22 |
| `kr-vs-kp_20nan` | 0.5882 | -0.0205 (-3.5%) better | -0.0703 (-12.0%) better | -0.0124 (-2.1%) better | +0.0039 (+0.7%) none | 12 |
| `biodeg_20nan` | 0.5964 | -0.0491 (-8.2%) better | -0.0276 (-4.6%) better | -0.0518 (-8.7%) better | -0.0011 (-0.2%) equivalent | 5 |
| `credit-g_20nan` | 0.8740 | -0.0433 (-5.0%) better | -0.0135 (-1.5%) better | -0.0365 (-4.2%) better | -0.0016 (-0.2%) better | 6 |
| `vehicle_20nan` | 0.4168 | -0.0436 (-10.5%) better | -0.0269 (-6.5%) better | -0.0435 (-10.4%) better | -0.0001 (-0.0%) equivalent | 2 |
| `kc2_20nan` | 0.5085 | -0.0454 (-8.9%) none | -0.0252 (-5.0%) none | -0.0211 (-4.1%) none | -0.0036 (-0.7%) none | 3 |

**Reading, by the rule fixed in advance: holds.** No variant reverses its earlier K − bar call.

- **K − bar on the fresh seeds:** better 7, no difference 1 (kc2), worse 1 (electricity).
- **Two variants change short of a reversal,** and each now agrees with the other earlier set:
  - **electricity_20nan:** worse, as on 101/202/303; it was "no difference" on 42/7/13. The
    stated limit stands: the boosters set the bar there (T29).
  - **spambase_20nan:** better, as on 42/7/13; it was "no difference" on 101/202/303.
- **K − strong (live for the first time at 20nan):** better 6, no difference 3 (electricity,
  spambase and kc2), worse 0.
  - credit-g_20nan moves from "no difference" on both earlier sets to better (−1.5% of K).
- **H − bar** (the deployable readout, ADR 0024) gives the same calls as K − bar on all nine.

**Secondary, not in the rule: calibration at 20nan.** ADR 0027 turns calibration off below a
missing rate of 0.3.

- **KC − K is better on letter_20nan (−2.1% of K) and pendigits_20nan (−2.8%).** On E34 and E35,
  with the old encoder, the gain was −0.0015 to −0.0018 on these two variants. Here it is −0.009,
  under today's bigger encoder (ADR 0018).
- **credit-g_20nan:** better by −0.2% of K, inside the 1% margin.
- **spambase_20nan:** worse (+0.6%), as on E34 and E35.
- **The other five:** no difference or equivalent.
- **No default changes from this study.** The 20nan threshold for the two bigger-encoder variants
  is a question for the user (T20).

**What it leads to (fixed in advance):**

- **The thesis table:** its nine 20nan rows gain a third seed set, 1001/2002/3003.
- **The seeds:** 1001/2002/3003 are now spent.

