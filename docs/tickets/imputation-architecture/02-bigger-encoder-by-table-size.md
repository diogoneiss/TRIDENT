# Does a bigger encoder help the large tables? (pre-registered 2026-10-06)

This ticket was written and committed before the first run (`5292c4d`, 01:10 GMT-3 on 2026-10-06);
the queues started at 14:48:45 GMT-3, after the amendment in the Running notes (`50451c6`), so about
00:30 GMT-3 on 2026-10-07 is the expected end. It tests the hypothesis the user chose after E37 (ticket
[01](01-architecture-screening.md)), on seeds E37 did not use.

## Why

In E37 the bigger encoder (arm ALL: 4 layers, feed-forward 256, a final LayerNorm) was better on
3 of 5 variants and worse on none, but its mean change over the five was −0.27% and its
improvement was concentrated in the largest table: pendigits −4.0%, while credit-g_20nan, the
smallest, was +1.5% (a near miss). The hypothesis: a bigger encoder helps large tables and costs
little or nothing on the others. The user chose, before this run, what a confirmed hypothesis
leads to: a default that depends on the table's size, the bigger encoder only on the large
tables, the others unchanged. So the cost on the small and medium tables is measured and
reported, not ruled on.

## Design (fixed before launch)

- **Arm ALL:** today's imputation configuration (ADRs 0013 to 0015, the promoted file or the
  defaults) with `LAYERS` 4, `DIM_FEED` 256 and `ENCODER_FINAL_NORM layer`, through
  `scripts/experiments/architecture_run.py ... ALL ... --experiment architecture-large-2026-10-06`.
- **Reference R, read rather than rerun:** E34's cells on the same seeds, as their K readout
  (`impute/*/induced_checkpoint/*`), which equals a default run to the last digit (checked for this
  runner's path below).
- **Variants:** all 21 of ADR 0007, in three groups fixed now by the rows of their table:
  - **large, 10,000 rows or more:** pendigits_20nan (10,992), letter_20nan (20,000),
    electricity_20nan (45,312);
  - **medium:** kr-vs-kp (3,196) and spambase (4,601), each at 20/40/60/80nan;
  - **small, about 1,000 rows or fewer:** credit-g at 20/40/60/80nan, vehicle, biodeg and kc2 at
    20/60nan.
- **Seeds:** 101, 202, 303. **Size:** 63 cells, tagged `experiment=architecture-large-2026-10-06`,
  `arm=ALL`.
- **Pairing:** folds and masks are shared by each pair; the bigger encoder draws more at
  initialisation, so a pair is two draws, partly shared. The integrity check holds each pair's
  baseline scores equal.
- **Running it:** four concurrent queues (`scripts/experiments/run_architecture_size_queue.sh
  1..4`), one trainer each, CPU threads capped at 4; cells assigned longest first by E30's arm-M
  cell time times 1.82 (ALL's cost in E37), about 570 estimated minutes per queue, the three
  electricity cells first: about 9.5 h of wall time. No cell starts after 14:00 GMT-3 on 2026-10-06
  (17:00 UTC); a cell not started by then is reported missing. Nothing is rerun or extended on its
  result; a crash is fixed, its cell rerun from the start, and the rerun noted here.
- **Checks before launch:** the runner with no change (arm R0, `--no-mlflow`) on credit-g_20nan
  (defaults) and kr-vs-kp_40nan (promoted file) at seed 101 must equal E34's K readouts on every
  recorded metric, with equal baselines.

## Measures

- **Primary:** per large variant, ALL − R on the induced `impute_score`, paired by seed and fold
  (15 pairs), t-interval; a verdict only when the interval excludes zero and all three seeds' mean
  difference agree in sign.
- **Reading, fixed now:**
  1. **Supported:** "ALL better" on at least 2 of the 3 large variants, and "R better" on none of
     them. The recommendation is then the size-dependent default the user chose: the bigger
     encoder on the large tables, by an ADR and the user's decision. Its mechanism exists already:
     the large variants' promoted `*.imputation.json` files carry `LAYERS 4`, `DIM_FEED 256` and
     `ENCODER_FINAL_NORM layer` (ADR 0005's per-variant files), no new code. It covers the three
     variants studied (the 20nan ones); pendigits, letter and electricity also exist at 40/60/80nan,
     outside the 21, and would need their own check.
  2. **Refuted:** "R better" on any large variant.
  3. **Mixed:** anything else.
- **Secondary, descriptive (the cost the user asked to see):** per group (large, medium, small),
  the verdict counts and the mean relative change with a t-interval over its (variant, seed) units;
  minutes a cell against R's; the masked population. The 10,000-row threshold is fixed now; a gain
  in the medium group (kr-vs-kp_40nan was −1.6% in E37) is reported, and a lower threshold would be
  a new pre-registered study.
- **Analysis:** `scripts/experiments/architecture_size_report.py` (`--tally` prints the reading,
  the groups and the integrity check).

## Known limits, stated in advance

- **Three large tables, one of which generated the hypothesis:** pendigits is a replication;
  letter and electricity are the real tests.
- **Cost:** about 1.82x a cell in E37, so a large-table default pays it on every run of those
  tables (electricity, about 3 h a cell with four trainers sharing the GPU).
- **The prior for the small tables** from E37 is about +1% to +1.5% worse on credit-g; the
  size-dependent rule makes it irrelevant to the default, so it is only described.
- **Multiplicity:** 3 primary tests; 21 verdicts reported in all.

## Running notes

- **Check before launch, done:** the R0 runs of credit-g_20nan and kr-vs-kp_40nan at seed 101
  equal E34's K readouts on every recorded metric (14 and 10), with equal baselines (20 and 12).
  They ran with another session's uncommitted work in the tree (ADR 0016, missForest baselines),
  which adds baseline imputers but, as this check shows, leaves the model's scores and the four
  original baselines unchanged.
- **Amended 2026-10-06, 14:50 GMT-3, before any cell ran (no score seen):** the user sequenced the
  launch after another session's missForest work (ADR 0016) was committed (`8f77956`) and its store
  backfill applied (finished 10:28 GMT-3), so that E37 and E38 share the six-baseline bar. The
  launch therefore moved past the original cutoff, which becomes 08:00 GMT-3 on 2026-10-07
  (11:00 UTC). Nothing else changes; the integrity check compares the four ADR 0007 baselines,
  which the backfill did not alter. Every cell now also fits the two missForest baselines (cached
  per fold), which adds time to the estimate.
- **Stopped early, 2026-10-06, at the user's request** (relayed by another session and confirmed
  by the user directly at 20:25 GMT-3), so that ADR 0017 (an `lgbm` baseline in the bar, missForest
  outside it) could land: STOP files let each queue finish its running cell, the last at 21:12
  GMT-3. **15 of 63 cells ran, none failed:** all 9 large-table cells (pendigits, letter,
  electricity) and spambase_60nan and spambase_80nan, seeds 101, 202, 303. The primary reading
  rests only on the large tables, so it is complete; the other 48 cells (the rest of the medium
  group and the whole small group, the described cost) are paused and will resume after ADR 0017
  under the same design. The model's scores do not depend on the baseline code, and the integrity
  check reads only the four ADR 0007 baselines; the resumed cells' run records will carry ADR
  0017's baseline set, which is noted when they finish.
- **Resumption planned, 2026-10-07:** the user asked to resume the 48 paused cells once the other
  session's ADR 0017 backfill ends (about 03:30 GMT-3). The cutoff moves to 14:00 GMT-3 on
  2026-10-07 (17:00 UTC) so the resumed cells (about 4 h) fit; nothing else changes. Their run
  records carry ADR 0017's baseline set; the model's scores and the four ADR 0007 baselines the
  integrity check reads do not depend on it.
- **Resumed 2026-10-07 at 02:32 GMT-3**, after the other session's ADR 0017 backfill ended. A
  re-check with ADR 0017's code (arm R0, credit-g_20nan, seed 101) equals E34's K readout on all 14
  recorded metrics and the four ADR 0007 baselines.

## Outcome, primary (2026-10-06, 21:30 GMT-3)

The integrity check passed on all 15 completed pairs (baseline scores equal). ALL − R on the
induced `impute_score`, 15 fold pairs per variant; lower is better.

| variant | rows | R | ALL | change | ALL − R [95% interval] (seeds 101, 202, 303) | verdict | minutes a cell, R → ALL |
|---|---|---|---|---|---|---|---|
| `pendigits_20nan` | 10,992 | 0.3331 | 0.3203 | −3.8% | −0.0128 [−0.0151, −0.0105] (−0.0132, −0.0135, −0.0116) | ALL better | 38 → 73 |
| `letter_20nan` | 20,000 | 0.4590 | 0.4331 | −5.6% | −0.0259 [−0.0283, −0.0234] (−0.0246, −0.0270, −0.0261) | ALL better | 59 → 111 |
| `electricity_20nan` | 45,312 | 0.6183 | 0.6204 | +0.3% | +0.0021 [−0.0070, +0.0112] (+0.0049, −0.0000, +0.0015) | none | 101 → 179 |

**Reading, by the rule fixed in advance: supported.** ALL is better on 2 of the 3 large tables and
worse on none. The mean relative change over the nine (variant, seed) units is −3.0% [−5.1%,
−1.0%]. By the user's pre-stated choice, the recommendation is the size-dependent default: the
bigger encoder on the large tables, through their promoted files and an ADR; the user decides.

**Caveats, stated rather than ruled on.**
- **Electricity, the largest table, gains nothing** (+0.3%, no verdict, masked +0.013 without a
  verdict) and would pay the cost: 1.77x a cell, about 3 h with four trainers. Its features are
  few (8) and mostly categorical-time; table size alone does not decide where depth pays.
- **pendigits replicated** E37's gain on new seeds (−3.8% here, −4.0% there); letter, a real test,
  gained the most (−5.6%) and now beats its best baseline by a wider margin (0.4331 against
  0.4722).
- **Cost on the large tables:** 1.84x a cell summed (73, 111, 179 against 38, 59, 101 minutes).
- **Medium group, partial (described):** spambase_60nan −0.2% and spambase_80nan −0.2%, both no
  verdict, at 2.1x and 1.6x a cell. The rest waits for the resumed cells.
- **Time:** the 15 cells ran 14:48 to 21:12 GMT-3; at stop, the queues were on plan (0.98x of the
  estimate for the cells done).

## Outcome, complete (2026-10-07, 07:30 GMT-3)

All 63 cells ran, none failed: 15 from 14:48 to 21:12 GMT-3 on 2026-10-06, the other 48 from 02:32
to 07:15 GMT-3 on 2026-10-07 after the pause. The integrity check passed on all 63 pairs. The primary
reading above stands: **supported**. ALL − R on the induced `impute_score`, 15 fold pairs per
variant; lower is better.

| variant | group | R | ALL | change | ALL − R [95% interval] | verdict | minutes a cell, R → ALL |
|---|---|---|---|---|---|---|---|
| `pendigits_20nan` | large | 0.3331 | 0.3203 | −3.84% | −0.0128 [−0.0151, −0.0105] | ALL better | 38 → 73 |
| `letter_20nan` | large | 0.4590 | 0.4331 | −5.64% | −0.0259 [−0.0283, −0.0234] | ALL better | 58 → 111 |
| `electricity_20nan` | large | 0.6183 | 0.6204 | +0.34% | +0.0021 [−0.0070, +0.0112] | none | 101 → 179 |
| `kr-vs-kp_20nan` | medium | 0.5866 | 0.5951 | +1.45% | +0.0085 [−0.0095, +0.0266] | none | 17 → 32 |
| `kr-vs-kp_40nan` | medium | 0.7626 | 0.7532 | −1.23% | −0.0094 [−0.0171, −0.0016] | ALL better | 19 → 37 |
| `kr-vs-kp_60nan` | medium | 0.8711 | 0.8697 | −0.15% | −0.0013 [−0.0107, +0.0081] | none | 19 → 36 |
| `kr-vs-kp_80nan` | medium | 0.9815 | 0.9810 | −0.06% | −0.0006 [−0.0114, +0.0102] | none | 16 → 31 |
| `spambase_20nan` | medium | 0.8529 | 0.8473 | −0.66% | −0.0056 [−0.0197, +0.0084] | none | 34 → 70 |
| `spambase_40nan` | medium | 0.9068 | 0.9019 | −0.53% | −0.0048 [−0.0119, +0.0022] | none | 36 → 75 |
| `spambase_60nan` | medium | 0.9404 | 0.9386 | −0.20% | −0.0019 [−0.0046, +0.0009] | none | 33 → 70 |
| `spambase_80nan` | medium | 0.9846 | 0.9830 | −0.17% | −0.0016 [−0.0046, +0.0013] | none | 37 → 60 |
| `credit-g_20nan` | small | 0.8751 | 0.8932 | +2.07% | +0.0181 [+0.0056, +0.0306] | R better | 6 → 11 |
| `credit-g_40nan` | small | 0.9662 | 0.9549 | −1.17% | −0.0113 [−0.0242, +0.0016] | none | 4 → 9 |
| `credit-g_60nan` | small | 0.9932 | 0.9852 | −0.81% | −0.0080 [−0.0215, +0.0055] | none | 4 → 7 |
| `credit-g_80nan` | small | 1.0105 | 1.0078 | −0.26% | −0.0026 [−0.0097, +0.0044] | none | 5 → 8 |
| `vehicle_20nan` | small | 0.4113 | 0.4255 | +3.44% | +0.0142 [+0.0074, +0.0210] | R better | 4 → 6 |
| `vehicle_60nan` | small | 0.6398 | 0.6579 | +2.82% | +0.0180 [+0.0030, +0.0331] | R better | 3 → 6 |
| `biodeg_20nan` | small | 0.5898 | 0.5913 | +0.25% | +0.0015 [−0.0122, +0.0151] | none | 8 → 15 |
| `biodeg_60nan` | small | 0.7644 | 0.7702 | +0.76% | +0.0058 [−0.0081, +0.0197] | none | 7 → 12 |
| `kc2_20nan` | small | 0.5363 | 0.5797 | +8.09% | +0.0434 [−0.0189, +0.1057] | none | 2 → 4 |
| `kc2_60nan` | small | 0.5529 | 0.5739 | +3.80% | +0.0210 [+0.0001, +0.0419] | R better | 3 → 3 |

**By group (described, as the user asked):**

| group | variants | ALL better | R better | mean relative change [95% interval] | cost a cell |
|---|---|---|---|---|---|
| large (≥10,000 rows) | 3 | 2 | 0 | −3.0% [−5.1%, −1.0%] | 1.84x |
| medium (3,196–4,601 rows) | 8 | 1 (kr-vs-kp_40nan) | 0 | −0.2% [−0.6%, +0.2%] | 1.95x |
| small (~520–1,055 rows) | 10 | 0 | 4 | +1.9% [+0.5%, +3.2%] | 1.77x |

**Reading.** The size dependence holds across all three groups: the bigger encoder helps the large
tables, is neutral on the medium ones at almost twice the cost, and hurts the small ones (worse on
credit-g_20nan +2.1%, vehicle_20nan +3.4%, vehicle_60nan +2.8%, kc2_60nan +3.8%; better on none).
The size-dependent default the user chose fits that pattern: the large tables get it, the others
keep today's encoder. Electricity, the largest table, remains the exception (no gain at 1.77x).

**Time.** The 48 resumed cells took 4 h 43 min against about 4 h estimated (1.18x); the two
baselines added by ADRs 0016 and 0017, computed in every fold, are the likely share of the
difference.
