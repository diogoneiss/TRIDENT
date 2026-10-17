# The full MNAR family: spambase, the 20nan rung and biodeg's tilt dial (E46; pre-registered 2026-10-17)

This ticket was written and committed before the first cell.

- **Plan:** [2026-10-17-experiment-improvements](../../plans/2026-10-17-experiment-improvements.md)
  (§2f; review items 8–9; D18). E45's rule read "go for the full family" (ticket
  [05](05-mnar-smoke.md), Outcome). The user asked for this run on 2026-10-17, after E45's
  Outcome.
- **When:** on the four GPU queues, which are free since E45 ended (11:29 GMT-3). E42's CPU lanes
  and the E44 metric panel run beside it; they do not write the store.
- **Code:** `datasets/generate_mnar.py` gains `--slope` and `variant_name` (the dial's file names;
  the default slope keeps E45's names and draws, a unit test pins both). New scripts below. No
  source change: ADR 0021's merge waits for this study to end, so every cell imports the same
  `src/` as E45.

## Why

E45 showed that the machinery works and that the model's lead over the weak bar survives
value-dependent missingness on three small numerical tables at 60nan, shrinking only on biodeg.
This family asks the same of a larger table (spambase), of a lighter rung (20nan), and asks how
the cost on biodeg grows with the tilt's strength.

## Design (fixed before launch)

### Data (`datasets/generate_mnar.py`, seed 1060, as in E45)

**Seven new files.** Every feature column loses its MCAR sibling's count of cells, drawn without
replacement with weights Phi(slope · z) on the standardised column.

| file | slope | MCAR control |
|---|---|---|
| `spambase_probit_60nan` | 0.5 | `spambase_60nan` |
| `biodeg_probit_20nan` | 0.5 | `biodeg_20nan` |
| `kc2_probit_20nan` | 0.5 | `kc2_20nan` |
| `vehicle_probit_20nan` | 0.5 | `vehicle_20nan` |
| `biodeg_probit0p25_60nan` | 0.25 | `biodeg_60nan` (E45's cells) |
| `biodeg_probit1p0_60nan` | 1.0 | `biodeg_60nan` (E45's cells) |
| `biodeg_probit2p0_60nan` | 2.0 | `biodeg_60nan` (E45's cells) |

**Checked before launch (2026-10-17, about 14:30 GMT-3):**

- E45's `biodeg_probit_60nan` regenerates identically from the extended generator.
- On all seven files:
  - per-column counts equal the MCAR sibling's;
  - every observed cell equals the complete table's;
  - the label column is untouched;
  - no context-free row (none in the MCAR siblings either).
- **The tilt:** the hidden values' mean is above the observed values' mean on every column of six
  files. **kc2_probit_20nan is the exception:** 18 of 21 columns.
  - The three exceptions (`v(g)`, `e`, `b`) are heavy-tailed. A z-score puts most of their values
    at nearly the same weight, so at 20nan the draw is close to MCAR there (mean percentile rank
    of the hidden cells 0.51–0.52, against 0.48–0.52 under MCAR).
  - **So kc2_20nan is a weak tilt,** and its pair is read with that said. It stays in the design:
    the files were fixed before any cell ran.
- **Mean standardised shift** (hidden mean minus observed mean, in standard deviations, averaged
  over columns), for scale:
  - spambase_60nan +0.193, biodeg_20nan +0.285, kc2_20nan +0.141, vehicle_20nan +0.381;
  - biodeg_60nan at slopes 0.25, 0.5 (E45), 1.0 and 2.0: +0.222, +0.358, +0.568, +0.788
    (MCAR +0.008).

### Arms (runner `scripts/experiments/mnar_family_run.py`, tag `experiment=mnar-family-2026-10-17` plus `arm`)

E45's arms, unchanged; the runner reuses `mnar_smoke_run.py` with a different tag.

- **G:** today's default with `--score_checkpoints`. The headline read is H; K is read only for
  the oracle contrast.
- **Mx:** the gaps as `[NULL]` in the decode stage, the loss checkpoint, `--score_column_wise`
  and `--score_checkpoints`.

### Cells, seeds and runs

- **The new pairs:**
  - spambase at 60nan;
  - biodeg, kc2 and vehicle at 20nan.
  - Both arms, MNAR and MCAR, seeds 42, 7 and 13: 48 cells.
- **The dial:** the three new slopes on biodeg at 60nan, both arms, the same seeds: 18 cells. Slope
  0.5 and the MCAR control are E45's cells (same folds and seeds, same `src/`).
- **66 cells** in all. The config source of every variant is the task defaults (checked).
- **Integrity:** G on the four MCAR controls must reproduce E35's cells (seeds 42/7/13,
  `decode_checkpoint=loss`) fold by fold, K as E35's `induced_checkpoint` and H as its headline
  (`mnar_family_report.py --integrity`). E35 holds all twelve.
- **Running it:** four queues (`scripts/experiments/run_mnar_family_queue.sh`), CPU threads capped
  at 4.
  - The split balances estimated minutes: about 222 per queue, so about 3.7 h of wall time.
  - No cutoff; a `STOP` file stops between cells.

## Measures

All readings are on the induced `impute_score` over 15 fold pairs, with the verdict rule, through
`scripts/experiments/mnar_family_report.py`.

1. **Per mechanism, on each new pair (`--tally`):**
   - G's H against the weak bar, against missForest and against the strong tier;
   - the mechanism effect, (H − bar) under MNAR minus the same under MCAR;
   - the oracle's optimism, (K − H) under MNAR minus the same under MCAR;
   - t2's control, (Mx column-wise − G's H) per mechanism and its MNAR-minus-MCAR difference.
2. **The rung (`--rung`):** on biodeg, kc2 and vehicle, the mechanism effect at 20nan (E46) beside
   60nan (E45), and their difference (fold-paired: the variants share their folds).
3. **The dial (`--dial`):** on biodeg at 60nan, at each slope (0.25, 0.5, 1.0, 2.0), against E45's
   MCAR cells:
   - G's H against the bar and missForest;
   - the mechanism effect, the oracle's optimism and t2's MNAR-minus-MCAR difference.
4. **Signed bias per column,** offline from the ledgers with `scripts/experiments/metric_panel.py`,
   if time allows. It does not enter any rule.

**"strong" is missForest alone here, as in E45:** these runs are made before ADR 0021's merge, so
the linear arms are not logged live. ADR 0021's backfill would add them later.

## Reading, fixed now

**Primary: does the lead over the weak bar survive MNAR on the new pairs?** It is read on G's H
against the weak bar under MNAR, on the four new pairs (reading 1):

- **supported** if better on at least 3 and worse on none;
- **refuted** if worse on at least 2, or better on at most 1;
- **mixed** otherwise.

**The dial, fixed now:**

- **Dose-response: yes** if the mechanism effect's mean rises strictly with the slope over 0.25,
  0.5, 1.0 and 2.0, and at 2.0 its verdict is "worse".
- **Separately:** whether the lead over the bar survives the steepest tilt, i.e. G's H against the
  bar at slope 2.0 is "better".

**Everything else is descriptive:** missForest, the oracle's optimism, t2 and the rung. In
particular:

- kc2_20nan is a weak tilt;
- t2's Mx changes three things at once, so it does not isolate the `[NULL]`/`[MASK]` split.

**The known limit stands:** under MNAR the mean/mode denominator comes from biased observed
values, so `impute_score` is compared across mechanisms only as a difference of within-variant
gaps.

**If the integrity check fails** (a difference above 1e-6 on any MCAR control), nothing is read
until it is explained.

## Launch

To be filled in at launch: the pre-launch check and the launch time.
