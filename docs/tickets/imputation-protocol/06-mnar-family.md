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

- **Checked before launch, with `--no-mlflow`, passed (2026-10-17, about 14:36 GMT-3):**
  - **G on kc2_20nan at seed 42** equals E35's cell: K equals E35's `induced_checkpoint` and H its
    headline, on both populations, to the store's precision (induced K 0.5744053, H 0.5775417).
  - **Mx on kc2_probit_20nan at seed 99** (not a study seed) ran and logged its column-wise family.
    Its scores were not read.
- **Launched at 14:38 GMT-3** on 3c534bf: four queues in `/scratch2/diogoneiss/trident-2026-10-17-mnar-family/`
  (16, 16, 16 and 18 cells), progress in its `progress.log`.

## Outcome (2026-10-17, 18:15 GMT-3)

**Run.** All 66 cells ran from 14:38 to 18:11 GMT-3 (3 h 33 min against about 3.7 h planned).
None failed and none is missing.

**Integrity holds:** G on the four MCAR controls equals E35 on all 12 cells (largest difference 0).

```text
spambase_60nan s42: largest difference from E35 0
  spambase_60nan s7: largest difference from E35 0
  spambase_60nan s13: largest difference from E35 0
  biodeg_20nan s42: largest difference from E35 0
  biodeg_20nan s7: largest difference from E35 0
  biodeg_20nan s13: largest difference from E35 0
  kc2_20nan s42: largest difference from E35 0
  kc2_20nan s7: largest difference from E35 0
  kc2_20nan s13: largest difference from E35 0
  vehicle_20nan s42: largest difference from E35 0
  vehicle_20nan s7: largest difference from E35 0
  vehicle_20nan s13: largest difference from E35 0
```

**Primary, by the rule fixed in advance: supported.** Under MNAR, G's H is better than the weak bar
on all 4 new pairs:

- spambase_60nan −0.014;
- biodeg_20nan −0.035;
- kc2_20nan −0.094;
- vehicle_20nan −0.077.

Each line of the printouts below is a mean difference over the 15 fold pairs of the induced
`impute_score` (lower is better), its 95% t-interval and the verdict.

**The new pairs** (`mnar_family_report.py --tally`):

```text
=== spambase_probit_60nan against spambase_60nan
  MNAR G/H - bar: -0.0138 [-0.0214, -0.0062] -> better
  MNAR G/H - missforest: -0.0011 [-0.0074, +0.0051] -> none
  MNAR G/H - strong: -0.0011 [-0.0074, +0.0051] -> none
  MCAR G/H - bar: -0.0395 [-0.0439, -0.0352] -> better
  MCAR G/H - missforest: -0.0357 [-0.0416, -0.0298] -> better
  MCAR G/H - strong: -0.0357 [-0.0416, -0.0298] -> better
  (H - bar), MNAR minus MCAR: +0.0257 [+0.0166, +0.0348] -> worse
  (K - H), MNAR minus MCAR (the oracle's optimism): -0.0077 [-0.0156, +0.0002] -> none
  MNAR Mx column-wise - G/H: +0.0014 [-0.0044, +0.0072] -> none
  MCAR Mx column-wise - G/H: +0.0029 [+0.0006, +0.0052] -> worse
  t2, (Mx column-wise - G/H), MNAR minus MCAR: -0.0015 [-0.0079, +0.0050] -> none
=== biodeg_probit_20nan against biodeg_20nan
  MNAR G/H - bar: -0.0350 [-0.0489, -0.0211] -> better
  MNAR G/H - missforest: -0.0260 [-0.0384, -0.0136] -> better
  MNAR G/H - strong: -0.0260 [-0.0384, -0.0136] -> better
  MCAR G/H - bar: -0.0379 [-0.0578, -0.0180] -> better
  MCAR G/H - missforest: -0.0229 [-0.0433, -0.0025] -> better
  MCAR G/H - strong: -0.0229 [-0.0433, -0.0025] -> better
  (H - bar), MNAR minus MCAR: +0.0029 [-0.0167, +0.0224] -> none
  (K - H), MNAR minus MCAR (the oracle's optimism): +0.0064 [-0.0133, +0.0261] -> none
  MNAR Mx column-wise - G/H: +0.0326 [+0.0167, +0.0485] -> worse
  MCAR Mx column-wise - G/H: +0.0361 [+0.0150, +0.0571] -> worse
  t2, (Mx column-wise - G/H), MNAR minus MCAR: -0.0035 [-0.0286, +0.0216] -> none
=== kc2_probit_20nan against kc2_20nan
  MNAR G/H - bar: -0.0940 [-0.1453, -0.0428] -> better
  MNAR G/H - missforest: -0.0856 [-0.1387, -0.0324] -> better
  MNAR G/H - strong: -0.0856 [-0.1387, -0.0324] -> better
  MCAR G/H - bar: +0.0109 [-0.0416, +0.0635] -> none
  MCAR G/H - missforest: -0.0279 [-0.0986, +0.0427] -> none
  MCAR G/H - strong: -0.0279 [-0.0986, +0.0427] -> none
  (H - bar), MNAR minus MCAR: -0.1049 [-0.1606, -0.0493] -> better
  (K - H), MNAR minus MCAR (the oracle's optimism): +0.0355 [-0.0354, +0.1064] -> none
  MNAR Mx column-wise - G/H: +0.0333 [+0.0060, +0.0606] -> worse
  MCAR Mx column-wise - G/H: +0.0003 [-0.0200, +0.0206] -> none
  t2, (Mx column-wise - G/H), MNAR minus MCAR: +0.0330 [+0.0038, +0.0622] -> worse
=== vehicle_probit_20nan against vehicle_20nan
  MNAR G/H - bar: -0.0765 [-0.1135, -0.0395] -> better
  MNAR G/H - missforest: -0.0719 [-0.1236, -0.0201] -> better
  MNAR G/H - strong: -0.0719 [-0.1236, -0.0201] -> better
  MCAR G/H - bar: -0.0495 [-0.0644, -0.0346] -> better
  MCAR G/H - missforest: -0.0298 [-0.0516, -0.0081] -> better
  MCAR G/H - strong: -0.0298 [-0.0516, -0.0081] -> better
  (H - bar), MNAR minus MCAR: -0.0270 [-0.0669, +0.0129] -> none
  (K - H), MNAR minus MCAR (the oracle's optimism): +0.0084 [-0.0107, +0.0275] -> none
  MNAR Mx column-wise - G/H: +0.0491 [+0.0152, +0.0830] -> worse
  MCAR Mx column-wise - G/H: +0.0256 [+0.0143, +0.0370] -> worse
  t2, (Mx column-wise - G/H), MNAR minus MCAR: +0.0235 [-0.0080, +0.0550] -> none

Primary, G's H against the weak bar under MNAR on the 4 new pairs: better 4, worse 0, other 0 -> supported (fold-pair t, seed agreement)
```

**The rung** (`--rung`; 60nan from E45):

```text
=== biodeg
  (H - bar), MNAR minus MCAR, at 20nan: +0.0029 [-0.0167, +0.0224] -> none
  (H - bar), MNAR minus MCAR, at 60nan: +0.0809 [+0.0665, +0.0953] -> worse
  the same, 60nan minus 20nan: +0.0780 [+0.0549, +0.1012] -> worse
=== kc2
  (H - bar), MNAR minus MCAR, at 20nan: -0.1049 [-0.1606, -0.0493] -> better
  (H - bar), MNAR minus MCAR, at 60nan: -0.0198 [-0.1066, +0.0670] -> none
  the same, 60nan minus 20nan: +0.0852 [-0.0231, +0.1934] -> none
=== vehicle
  (H - bar), MNAR minus MCAR, at 20nan: -0.0270 [-0.0669, +0.0129] -> none
  (H - bar), MNAR minus MCAR, at 60nan: +0.0140 [-0.0024, +0.0303] -> none
  the same, 60nan minus 20nan: +0.0410 [-0.0050, +0.0870] -> none
```

**The dial** (`--dial`; slope 0.5 and the MCAR control from E45). The line "Mx column-wise −
G/H" per slope was added to the script after the first printout, and it is E45's per-mechanism
measure.

```text
=== slope 0.25: biodeg_probit0p25_60nan
  G/H - bar: -0.0146 [-0.0231, -0.0061] -> better
  G/H - missforest: +0.0237 [+0.0143, +0.0331] -> worse
  (H - bar), minus MCAR's: +0.0827 [+0.0684, +0.0970] -> worse
  (K - H), minus MCAR's: +0.0041 [-0.0056, +0.0138] -> none
  Mx column-wise - G/H: +0.0064 [+0.0007, +0.0121] -> worse
  t2, (Mx column-wise - G/H), minus MCAR's: -0.0142 [-0.0247, -0.0036] -> better
=== slope 0.5: biodeg_probit_60nan
  G/H - bar: -0.0165 [-0.0232, -0.0097] -> better
  G/H - missforest: +0.0204 [+0.0111, +0.0296] -> worse
  (H - bar), minus MCAR's: +0.0809 [+0.0665, +0.0953] -> worse
  (K - H), minus MCAR's: +0.0063 [-0.0007, +0.0132] -> none
  Mx column-wise - G/H: +0.0097 [+0.0034, +0.0161] -> worse
  t2, (Mx column-wise - G/H), minus MCAR's: -0.0108 [-0.0213, -0.0003] -> better
=== slope 1: biodeg_probit1p0_60nan
  G/H - bar: -0.0015 [-0.0051, +0.0021] -> none
  G/H - missforest: -0.0059 [-0.0101, -0.0018] -> better
  (H - bar), minus MCAR's: +0.0958 [+0.0832, +0.1085] -> worse
  (K - H), minus MCAR's: +0.0008 [-0.0067, +0.0083] -> none
  Mx column-wise - G/H: +0.0016 [-0.0026, +0.0058] -> none
  t2, (Mx column-wise - G/H), minus MCAR's: -0.0190 [-0.0271, -0.0108] -> better
=== slope 2: biodeg_probit2p0_60nan
  G/H - bar: +0.0054 [+0.0012, +0.0095] -> worse
  G/H - missforest: -0.0245 [-0.0320, -0.0170] -> better
  (H - bar), minus MCAR's: +0.1027 [+0.0882, +0.1172] -> worse
  (K - H), minus MCAR's: -0.0047 [-0.0124, +0.0030] -> none
  Mx column-wise - G/H: -0.0057 [-0.0101, -0.0014] -> better
  t2, (Mx column-wise - G/H), minus MCAR's: -0.0263 [-0.0348, -0.0178] -> better
  MCAR control (biodeg_60nan): Mx column-wise - G/H: +0.0205 [+0.0130, +0.0281] -> worse

Dose-response (the effect's mean rises with the slope, and at 2.0 it is 'worse'): no (means +0.0827, +0.0809, +0.0958, +0.1027; at 2.0 worse)
```

**The dial, by the rule fixed in advance: dose-response "no".**

- **The mechanism effect is flat from 0.25 to 0.5,** then rises: +0.083, +0.081, +0.096, +0.103.
  So it does not rise strictly, though it is "worse" at every slope.
- **The lead over the bar does not survive the steepest tilt.** At slope 2.0, G's H is worse than
  the bar (+0.005, against hgb/lgbm). At 1.0 there is no difference.

**Descriptively:**

- **MNAR costs the model on the larger table and at heavier missingness.**
  - **spambase:** the lead over the bar shrinks under MNAR (mechanism effect +0.026, worse).
    Against missForest it goes from better (−0.036) to no difference.
  - **biodeg:** no mechanism effect at 20nan, a large one at 60nan (the difference, +0.078, is
    worse). The cost grows with missingness there.
  - **vehicle:** no mechanism effect at either level.
- **kc2_20nan is the odd pair.** Under MCAR the model has no difference from the bar; under MNAR
  it is better (mechanism effect −0.105). It is the weak-tilt pair, and kc2's bar is knn, which
  depends on twin rows (E44's split), so this is noted, not explained.
- **missForest suffers more than the model as the tilt steepens on biodeg.**
  - The model against missForest goes from worse at 0.25 and 0.5 (+0.024, +0.020) to better at
    1.0 and 2.0 (−0.006, −0.025).
  - The boosters, which set the bar, degrade less than missForest.
- **The oracle's optimism does not grow under MNAR** on any pair or slope. So H stays the right
  readout.
- **t2's control:**
  - On the new pairs, Mx is worse than G or no different. Its MNAR-minus-MCAR difference has no
    verdict, except kc2_20nan, where Mx loses more under MNAR.
  - On the dial, its deficit shrinks at every slope against MCAR (−0.014 to −0.026). At slope
    2.0, Mx is better than G (−0.006), the only variant of E45 and E46 where the untied arm wins.
  - So the `[NULL]` arm gains as missingness becomes more informative, on biodeg only. Mx still
    changes three things at once.
- **"strong" is missForest alone,** as fixed in advance. ADR 0021's backfill would add the linear
  arms to these runs.

**The signed bias per column (measure 4)** was read afterwards, for E45's pairs too; see the
section at the end.

**For the user:**

- **For the thesis, the RQ1 MNAR arm:** the lead over the weak bar survives value-dependent
  missingness on 7 of 7 table-and-rung pairs (E45 and E46). It shrinks on spambase and on biodeg
  at 60nan, and it is lost on biodeg only at the steepest tilt (slope 2.0).
- **t2:** whether biodeg's dial result for the untied arm is worth a clean ablation (`[NULL]`
  gaps alone, without column-wise scoring and the loss checkpoint).

## Signed bias per column, E45's measure 5 and E46's measure 4 (added 2026-10-19, 00:21 GMT-3)

**How it was made:**

- **Source:** `scripts/experiments/metric_panel.py --baselines --cell_suffix _G`, on arm G's fold
  ledgers. Every baseline was refitted on the same folds, and nothing was written to the store.
- **Reading:** `scripts/experiments/mnar_bias_report.py`.
- **The model's fills are its ledger's,** the run's own checkpoint (K), not H.
- **The signed bias** is fill minus truth, averaged over the numerical columns, in scaled units;
  negative means under-predicting.
- **Δ** is MNAR minus MCAR, fold-paired.

```text
Signed bias (fill − truth, scaled units, mean over numerical columns), mean over 15 folds;
Δ = MNAR − MCAR, fold-paired, with the verdict rule (fold-pair t, seed agreement).

### biodeg_probit_60nan against biodeg_60nan

| method | MNAR | MCAR | Δ [95% interval] | verdict | columns under, MNAR |
|---|---|---|---|---|---|
| model | -0.3323 | -0.0396 | -0.2926 [-0.3040, -0.2813] | falls under MNAR | 98% |
| mean_mode | -0.4630 | -0.0105 | -0.4526 [-0.4619, -0.4432] | falls under MNAR | 99% |
| knn5 | -0.3814 | -0.0536 | -0.3278 [-0.3426, -0.3131] | falls under MNAR | 96% |
| knn10 | -0.3926 | -0.0609 | -0.3317 [-0.3447, -0.3187] | falls under MNAR | 98% |
| hgb | -0.2836 | +0.0272 | -0.3108 [-0.3213, -0.3003] | falls under MNAR | 95% |
| lgbm | -0.2826 | +0.0288 | -0.3114 [-0.3215, -0.3012] | falls under MNAR | 95% |
| missforest | -0.3088 | +0.0052 | -0.3140 [-0.3320, -0.2960] | falls under MNAR | 97% |
| condexp | -0.2892 | +0.0634 | -0.3526 [-0.3629, -0.3422] | falls under MNAR | 90% |
| softimpute | -0.1357 | +0.1821 | -0.3178 [-0.3383, -0.2973] | falls under MNAR | 73% |
| ice | -0.3012 | +0.0353 | -0.3365 [-0.3526, -0.3204] | falls under MNAR | 93% |

By |Δ|, the model ranks 1 of 10 (1 moves least; least: model, most: mean_mode).

### kc2_probit_60nan against kc2_60nan

| method | MNAR | MCAR | Δ [95% interval] | verdict | columns under, MNAR |
|---|---|---|---|---|---|
| model | -0.2658 | -0.0393 | -0.2265 [-0.3231, -0.1299] | falls under MNAR | 81% |
| mean_mode | -0.5194 | -0.0111 | -0.5082 [-0.6060, -0.4104] | falls under MNAR | 94% |
| knn5 | -0.4421 | -0.1416 | -0.3005 [-0.3916, -0.2095] | falls under MNAR | 96% |
| knn10 | -0.4844 | -0.1538 | -0.3305 [-0.4252, -0.2359] | falls under MNAR | 96% |
| hgb | -0.3402 | -0.0303 | -0.3100 [-0.4148, -0.2051] | falls under MNAR | 82% |
| lgbm | -0.3336 | -0.0263 | -0.3073 [-0.4143, -0.2002] | falls under MNAR | 81% |
| missforest | -0.2884 | -0.0194 | -0.2691 [-0.3704, -0.1677] | falls under MNAR | 81% |
| condexp | -0.3098 | +0.0180 | -0.3278 [-0.4144, -0.2412] | falls under MNAR | 82% |
| softimpute | -0.2648 | -0.0043 | -0.2605 [-0.3592, -0.1618] | falls under MNAR | 79% |
| ice | -0.3618 | -0.0344 | -0.3273 [-0.4311, -0.2236] | falls under MNAR | 92% |

By |Δ|, the model ranks 1 of 10 (1 moves least; least: model, most: mean_mode).

### vehicle_probit_60nan against vehicle_60nan

| method | MNAR | MCAR | Δ [95% interval] | verdict | columns under, MNAR |
|---|---|---|---|---|---|
| model | -0.1623 | -0.0297 | -0.1327 [-0.1504, -0.1149] | falls under MNAR | 87% |
| mean_mode | -0.5916 | -0.0020 | -0.5896 [-0.5977, -0.5816] | falls under MNAR | 100% |
| knn5 | -0.3569 | -0.0222 | -0.3347 [-0.3506, -0.3188] | falls under MNAR | 100% |
| knn10 | -0.3641 | -0.0239 | -0.3402 [-0.3552, -0.3253] | falls under MNAR | 100% |
| hgb | -0.2138 | -0.0118 | -0.2019 [-0.2160, -0.1879] | falls under MNAR | 98% |
| lgbm | -0.2138 | -0.0119 | -0.2020 [-0.2160, -0.1880] | falls under MNAR | 98% |
| missforest | -0.1867 | -0.0072 | -0.1795 [-0.1952, -0.1639] | falls under MNAR | 93% |
| condexp | -0.3864 | -0.0180 | -0.3684 [-0.3948, -0.3420] | falls under MNAR | 97% |
| softimpute | -0.2941 | -0.0093 | -0.2847 [-0.3042, -0.2653] | falls under MNAR | 96% |
| ice | -0.2986 | -0.0210 | -0.2776 [-0.2995, -0.2557] | falls under MNAR | 100% |

By |Δ|, the model ranks 1 of 10 (1 moves least; least: model, most: mean_mode).

### spambase_probit_60nan against spambase_60nan

| method | MNAR | MCAR | Δ [95% interval] | verdict | columns under, MNAR |
|---|---|---|---|---|---|
| model | -0.2754 | -0.0308 | -0.2446 [-0.2540, -0.2353] | falls under MNAR | 97% |
| mean_mode | -0.3052 | -0.0083 | -0.2969 [-0.3001, -0.2937] | falls under MNAR | 99% |
| knn5 | -0.3006 | -0.0230 | -0.2776 [-0.2842, -0.2710] | falls under MNAR | 96% |
| knn10 | -0.3054 | -0.0280 | -0.2775 [-0.2829, -0.2721] | falls under MNAR | 97% |
| hgb | -0.2343 | +0.0302 | -0.2645 [-0.2689, -0.2600] | falls under MNAR | 95% |
| lgbm | -0.2333 | +0.0311 | -0.2645 [-0.2687, -0.2603] | falls under MNAR | 95% |
| missforest | -0.2453 | +0.0341 | -0.2794 [-0.2977, -0.2611] | falls under MNAR | 96% |
| condexp | -0.2709 | +0.0161 | -0.2870 [-0.2924, -0.2816] | falls under MNAR | 96% |
| softimpute | -0.1628 | +0.1258 | -0.2886 [-0.3032, -0.2740] | falls under MNAR | 82% |
| ice | -0.2234 | +0.0475 | -0.2708 [-0.2793, -0.2624] | falls under MNAR | 91% |

By |Δ|, the model ranks 1 of 10 (1 moves least; least: model, most: mean_mode).

### biodeg_probit_20nan against biodeg_20nan

| method | MNAR | MCAR | Δ [95% interval] | verdict | columns under, MNAR |
|---|---|---|---|---|---|
| model | -0.0762 | -0.0315 | -0.0447 [-0.0553, -0.0341] | falls under MNAR | 68% |
| mean_mode | -0.3100 | -0.0177 | -0.2923 [-0.3073, -0.2773] | falls under MNAR | 91% |
| knn5 | -0.2134 | -0.0731 | -0.1403 [-0.1537, -0.1269] | falls under MNAR | 92% |
| knn10 | -0.2462 | -0.0828 | -0.1635 [-0.1759, -0.1510] | falls under MNAR | 93% |
| hgb | -0.0888 | -0.0094 | -0.0794 [-0.0913, -0.0675] | falls under MNAR | 75% |
| lgbm | -0.0903 | -0.0094 | -0.0809 [-0.0927, -0.0690] | falls under MNAR | 73% |
| missforest | -0.1043 | -0.0103 | -0.0940 [-0.1064, -0.0816] | falls under MNAR | 80% |
| condexp | -0.1012 | +0.0303 | -0.1316 [-0.1467, -0.1164] | falls under MNAR | 73% |
| softimpute | -0.0270 | +0.0555 | -0.0825 [-0.0932, -0.0719] | falls under MNAR | 55% |
| ice | -0.0152 | +0.0364 | -0.0516 [-0.0678, -0.0354] | falls under MNAR | 54% |

By |Δ|, the model ranks 1 of 10 (1 moves least; least: model, most: mean_mode).

### kc2_probit_20nan against kc2_20nan

| method | MNAR | MCAR | Δ [95% interval] | verdict | columns under, MNAR |
|---|---|---|---|---|---|
| model | -0.0497 | -0.0001 | -0.0495 [-0.0673, -0.0318] | falls under MNAR | 62% |
| mean_mode | -0.1768 | -0.0006 | -0.1762 [-0.2004, -0.1520] | falls under MNAR | 65% |
| knn5 | -0.1344 | -0.0592 | -0.0752 [-0.0940, -0.0565] | falls under MNAR | 90% |
| knn10 | -0.1525 | -0.0700 | -0.0826 [-0.1014, -0.0637] | falls under MNAR | 91% |
| hgb | -0.0434 | +0.0213 | -0.0647 [-0.0923, -0.0371] | falls under MNAR | 51% |
| lgbm | -0.0389 | +0.0242 | -0.0631 [-0.0911, -0.0351] | falls under MNAR | 50% |
| missforest | -0.0525 | +0.0026 | -0.0551 [-0.0746, -0.0356] | falls under MNAR | 45% |
| condexp | -0.0314 | +0.0535 | -0.0848 [-0.1179, -0.0518] | falls under MNAR | 48% |
| softimpute | -0.0597 | +0.0129 | -0.0727 [-0.0915, -0.0539] | falls under MNAR | 57% |
| ice | -0.0610 | -0.0103 | -0.0507 [-0.0648, -0.0365] | falls under MNAR | 55% |

By |Δ|, the model ranks 1 of 10 (1 moves least; least: model, most: mean_mode).

### vehicle_probit_20nan against vehicle_20nan

| method | MNAR | MCAR | Δ [95% interval] | verdict | columns under, MNAR |
|---|---|---|---|---|---|
| model | -0.0764 | -0.0214 | -0.0550 [-0.0712, -0.0388] | falls under MNAR | 69% |
| mean_mode | -0.3909 | -0.0318 | -0.3591 [-0.3982, -0.3200] | falls under MNAR | 96% |
| knn5 | -0.1144 | -0.0290 | -0.0855 [-0.1109, -0.0600] | falls under MNAR | 84% |
| knn10 | -0.1254 | -0.0288 | -0.0966 [-0.1217, -0.0715] | falls under MNAR | 89% |
| hgb | -0.0902 | -0.0186 | -0.0716 [-0.0926, -0.0505] | falls under MNAR | 76% |
| lgbm | -0.0893 | -0.0201 | -0.0692 [-0.0901, -0.0483] | falls under MNAR | 77% |
| missforest | -0.0990 | -0.0174 | -0.0816 [-0.1023, -0.0609] | falls under MNAR | 79% |
| condexp | -0.1339 | -0.0114 | -0.1225 [-0.1451, -0.0999] | falls under MNAR | 90% |
| softimpute | -0.0952 | +0.0045 | -0.0998 [-0.1250, -0.0746] | falls under MNAR | 86% |
| ice | -0.0776 | -0.0068 | -0.0708 [-0.0882, -0.0533] | falls under MNAR | 80% |

By |Δ|, the model ranks 1 of 10 (1 moves least; least: model, most: mean_mode).

### biodeg_probit0p25_60nan against biodeg_60nan

| method | MNAR | MCAR | Δ [95% interval] | verdict | columns under, MNAR |
|---|---|---|---|---|---|
| model | -0.2187 | -0.0396 | -0.1791 [-0.1988, -0.1594] | falls under MNAR | 94% |
| mean_mode | -0.2824 | -0.0105 | -0.2719 [-0.2839, -0.2599] | falls under MNAR | 98% |
| knn5 | -0.2308 | -0.0536 | -0.1771 [-0.1909, -0.1634] | falls under MNAR | 94% |
| knn10 | -0.2469 | -0.0609 | -0.1860 [-0.1990, -0.1730] | falls under MNAR | 96% |
| hgb | -0.1754 | +0.0272 | -0.2026 [-0.2171, -0.1882] | falls under MNAR | 91% |
| lgbm | -0.1754 | +0.0288 | -0.2042 [-0.2184, -0.1899] | falls under MNAR | 92% |
| missforest | -0.2006 | +0.0052 | -0.2059 [-0.2271, -0.1847] | falls under MNAR | 95% |
| condexp | -0.1486 | +0.0634 | -0.2120 [-0.2274, -0.1965] | falls under MNAR | 83% |
| softimpute | -0.0261 | +0.1821 | -0.2082 [-0.2270, -0.1895] | falls under MNAR | 53% |
| ice | -0.1724 | +0.0353 | -0.2077 [-0.2248, -0.1906] | falls under MNAR | 89% |

By |Δ|, the model ranks 2 of 10 (1 moves least; least: knn5, most: mean_mode).

### biodeg_probit1p0_60nan against biodeg_60nan

| method | MNAR | MCAR | Δ [95% interval] | verdict | columns under, MNAR |
|---|---|---|---|---|---|
| model | -0.5820 | -0.0396 | -0.5423 [-0.5660, -0.5187] | falls under MNAR | 100% |
| mean_mode | -0.8597 | -0.0105 | -0.8492 [-0.8672, -0.8313] | falls under MNAR | 99% |
| knn5 | -0.6892 | -0.0536 | -0.6355 [-0.6580, -0.6131] | falls under MNAR | 99% |
| knn10 | -0.7067 | -0.0609 | -0.6458 [-0.6664, -0.6253] | falls under MNAR | 99% |
| hgb | -0.5606 | +0.0272 | -0.5878 [-0.6044, -0.5712] | falls under MNAR | 99% |
| lgbm | -0.5598 | +0.0288 | -0.5886 [-0.6058, -0.5714] | falls under MNAR | 99% |
| missforest | -0.6274 | +0.0052 | -0.6326 [-0.6544, -0.6108] | falls under MNAR | 100% |
| condexp | -0.6571 | +0.0634 | -0.7205 [-0.7431, -0.6978] | falls under MNAR | 97% |
| softimpute | -0.5422 | +0.1821 | -0.7244 [-0.7483, -0.7005] | falls under MNAR | 95% |
| ice | -0.6307 | +0.0353 | -0.6660 [-0.6892, -0.6428] | falls under MNAR | 98% |

By |Δ|, the model ranks 1 of 10 (1 moves least; least: model, most: mean_mode).

### biodeg_probit2p0_60nan against biodeg_60nan

| method | MNAR | MCAR | Δ [95% interval] | verdict | columns under, MNAR |
|---|---|---|---|---|---|
| model | -1.4051 | -0.0396 | -1.3655 [-1.3939, -1.3371] | falls under MNAR | 99% |
| mean_mode | -1.8486 | -0.0105 | -1.8382 [-1.8606, -1.8158] | falls under MNAR | 99% |
| knn5 | -1.6666 | -0.0536 | -1.6130 [-1.6441, -1.5818] | falls under MNAR | 98% |
| knn10 | -1.6881 | -0.0609 | -1.6272 [-1.6570, -1.5975] | falls under MNAR | 99% |
| hgb | -1.3897 | +0.0272 | -1.4169 [-1.4420, -1.3918] | falls under MNAR | 99% |
| lgbm | -1.3864 | +0.0288 | -1.4152 [-1.4408, -1.3896] | falls under MNAR | 99% |
| missforest | -1.5649 | +0.0052 | -1.5701 [-1.6126, -1.5276] | falls under MNAR | 99% |
| condexp | -1.6611 | +0.0634 | -1.7244 [-1.7499, -1.6990] | falls under MNAR | 98% |
| softimpute | -1.5649 | +0.1821 | -1.7471 [-1.7751, -1.7190] | falls under MNAR | 94% |
| ice | -1.5701 | +0.0353 | -1.6054 [-1.6362, -1.5745] | falls under MNAR | 96% |

By |Δ|, the model ranks 1 of 10 (1 moves least; least: model, most: mean_mode).
```

**Reading (descriptive):**

- **Under MNAR every imputer falls below the hidden values,** on every pair and every method, as
  expected: the mechanism hides large values. Δ is negative with a verdict in all 100
  (pair, method) readings.
- **The model's bias moves least of the 10 methods on 9 of the 10 pairs.**
  - It ranks 2nd, behind knn5, at biodeg's slope 0.25.
  - mean/mode moves most everywhere, as it must: its fill is the observed mean.
  - Under MNAR the model under-predicts, for example −0.33 against −0.04 under MCAR on
    biodeg_60nan, but less than any baseline.
- **The cost grows with the tilt.** On biodeg at 60nan the model's Δ is −0.18, −0.29, −0.54 and
  −1.37 at slopes 0.25, 0.5, 1.0 and 2.0. That is still the smallest shift at every slope above
  0.25.
- **This is consistent with the scores.** The lead over the bar survives MNAR (E45, E46), because
  the model degrades least. It is lost at slope 2.0, where the shift is large for everyone.

