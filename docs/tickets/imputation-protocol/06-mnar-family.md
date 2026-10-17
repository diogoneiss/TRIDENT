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

**Not done:** the signed bias per column (measure 4), which enters no rule.

**For the user:**

- **For the thesis, the RQ1 MNAR arm:** the lead over the weak bar survives value-dependent
  missingness on 7 of 7 table-and-rung pairs (E45 and E46). It shrinks on spambase and on biodeg
  at 60nan, and it is lost on biodeg only at the steepest tilt (slope 2.0).
- **t2:** whether biodeg's dial result for the untied arm is worth a clean ablation (`[NULL]`
  gaps alone, without column-wise scoring and the loss checkpoint).

