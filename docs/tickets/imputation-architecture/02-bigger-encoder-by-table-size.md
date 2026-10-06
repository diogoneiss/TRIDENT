# Does a bigger encoder help the large tables? (pre-registered 2026-10-06)

This ticket was written and committed before the first run (the commit and the queues' start
time are in the Running notes). It tests the hypothesis the user chose after E37 (ticket
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
