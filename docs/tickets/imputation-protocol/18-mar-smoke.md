# A MAR smoke against matched MCAR controls (E63; pre-registered 2026-10-22)

This ticket was written and committed before the first cell.

- **Why:** this is T30.
  - Every variant so far is MCAR (`generate_splits.py`, E62's draws) or MNAR (E45, E46). Nothing
    tests missingness that depends on *other*, observed columns.
  - This is the third classical mechanism.
- **The user's go:** "Concordo com sua recomendação, vá" (2026-10-22, about 13:40 GMT-3), to "T30
  now on the GPU".
- **The code, with nothing in `src/`:**
  - `datasets/generate_mar.py`, the generator;
  - `scripts/experiments/mar_smoke_run.py`, the runner;
  - `scripts/experiments/mar_smoke_report.py`, the reader;
  - `scripts/experiments/run_mar_smoke_queue.sh`, the queues.

## Design (fixed before the first cell)

- **Tables:** biodeg, kc2, vehicle and kr-vs-kp at 60nan, as T30 recorded them. kr-vs-kp is
  all-categorical.
- **The generator** writes two variants per table from `_00nan`, which differ only in the
  mechanism.
  - **Drivers:** 30% of the feature columns, chosen at random, are fully observed in both.
    - A numerical driver enters standardised; a categorical one as its mode indicator,
      standardised.
  - **Every other column** loses exactly round(0.6 n) cells, the MCAR sibling's count:
    - **MAR** (`<base>_mar_60nan`): drawn without replacement with weights sigmoid(1.5 s), where
      s is the column's own random combination of the drivers, standardised. Weighted sampling at
      a fixed count is not exactly a logistic mechanism.
    - **Control** (`<base>_marctl_60nan`): the same count in the same columns, drawn uniformly.
  - **The counts the variants match:** 60% of each non-driver column, so about 42% of all
    feature cells, in both the MAR variant and its control. This is not the shipped `_60nan`'s 60%
    of every column. The control, not the shipped variant, is the comparison.
  - **Its checks passed, before this ticket:**
    - drivers fully observed, every other column at its count, observed cells unchanged;
    - **the manipulation,** the mean shift in s between hidden and observed rows:
      - MAR: biodeg +0.665, kc2 +0.421, vehicle +0.971, kr-vs-kp +0.777 (the floor is 0.3);
      - controls: +0.012, −0.042, −0.002, +0.001 (the ceiling is 0.1 in absolute value).
- **Cells:** 4 tables × 2 mechanisms × seeds 42/7/13 make 24 cells of today's default (arm G).
  - Each runs with `<base>_60nan`'s resolved configuration as an override, as E62 did, with
    `--score_checkpoints` and `--score_calibrated`.
  - ADR 0027's `auto` calibration resolves on the measured rate, about 0.42, to `gaps` in both.
  - TabPFN comes from an offline pass into the state directory's `tabpfn/`, one lane.
- **Running:**
  - four queues, about 1 h;
  - one TabPFN lane;
  - no cutoff;
  - state in `/scratch2/diogoneiss/trident-2026-10-22-mar-smoke/`.

## Measures and reading, fixed now (`mar_smoke_report.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule:

- **Per mechanism, with the 1% SESOI:** H − bar, H − strong (ADR 0028's tier), H − TabPFN, K − bar,
  K − strong and KC − strong.
  - H is the readout E45 and E46 read, the deployable one (ADR 0024).
- **The mechanism effect, fold-paired:** a seed builds the same folds on both variants.
  - (H − bar) MAR minus control;
  - (H − strong) MAR minus control;
  - the oracle's optimism, (K − H) MAR minus control.
- **Primary,** on the four tables:
  - **holds:** neither (H − bar) nor (H − strong) MAR minus control is worse on any table;
  - **loses ground:** worse on two or more table-contrasts;
  - **mixed:** otherwise.
- **The known limit, as in E45:** under MAR the mean/mode denominator comes from observed values
  the mechanism biases. So scores are compared across mechanisms only as differences of
  within-variant gaps.
- **What each reading leads to (no default changes; this is a smoke):**
  - **holds:** the thesis states that the model keeps its standing under MAR on these tables, as
    it did under MNAR (E45, E46, E58).
  - **loses ground** or **mixed:** the contrasts are named as a limit, and a fuller MAR family is
    proposed for the user to decide.

## Outcome (2026-10-22, 15:12 GMT-3)

**The run:** 24 of 24 cells and 24 of 24 TabPFN pairs, 13:44–15:10 GMT-3, with none failed.

**The printout** (`scripts/experiments/mar_smoke_report.py --tally`, unedited):

```text
## `biodeg` at 60nan
- MAR: H − bar (hgb/lgbm) -0.0358 better; H − strong (tabpfn) +0.0198 none; H − tabpfn (tabpfn) +0.0198 none; K − bar (hgb/lgbm) -0.0535 better; K − strong (tabpfn) +0.0021 none; KC − strong (tabpfn) +0.0010 none
- control: H − bar (lgbm) -0.0462 better; H − strong (tabpfn) -0.0212 none; H − tabpfn (tabpfn) -0.0212 none; K − bar (lgbm) -0.0596 better; K − strong (tabpfn) -0.0346 better; KC − strong (tabpfn) -0.0327 better
  - (H − bar) MAR minus control: +0.0104 [-0.0128, +0.0336] none
  - (H − strong) MAR minus control: +0.0410 [+0.0069, +0.0751] worse
  - (K − H) MAR minus control: -0.0043 [-0.0237, +0.0150] none

## `kc2` at 60nan
- MAR: H − bar (hgb) -0.1262 better; H − strong (tabpfn) -0.0149 none; H − tabpfn (tabpfn) -0.0149 none; K − bar (hgb) -0.1383 better; K − strong (tabpfn) -0.0270 none; KC − strong (tabpfn) -0.0276 none
- control: H − bar (knn5) -0.0537 none; H − strong (tabpfn) +0.0975 worse; H − tabpfn (tabpfn) +0.0975 worse; K − bar (knn5) -0.1023 better; K − strong (tabpfn) +0.0489 none; KC − strong (tabpfn) +0.0490 none
  - (H − bar) MAR minus control: -0.0725 [-0.1371, -0.0079] better
  - (H − strong) MAR minus control: -0.1124 [-0.1591, -0.0657] better
  - (K − H) MAR minus control: +0.0365 [-0.0030, +0.0760] none

## `vehicle` at 60nan
- MAR: H − bar (hgb/knn5/lgbm) -0.0408 better; H − strong (tabpfn) +0.0548 worse; H − tabpfn (tabpfn) +0.0548 worse; K − bar (hgb/knn5/lgbm) -0.0509 better; K − strong (tabpfn) +0.0448 worse; KC − strong (tabpfn) +0.0413 worse
- control: H − bar (hgb/knn10/lgbm) -0.0437 better; H − strong (tabpfn) +0.0491 worse; H − tabpfn (tabpfn) +0.0491 worse; K − bar (hgb/knn10/lgbm) -0.0562 better; K − strong (tabpfn) +0.0366 worse; KC − strong (tabpfn) +0.0326 worse
  - (H − bar) MAR minus control: +0.0029 [-0.0129, +0.0186] none
  - (H − strong) MAR minus control: +0.0057 [-0.0125, +0.0238] none
  - (K − H) MAR minus control: +0.0025 [-0.0117, +0.0167] none

## `kr-vs-kp` at 60nan
- MAR: H − bar (hgb/lgbm) -0.0282 better; H − strong (tabpfn) +0.0462 worse; H − tabpfn (tabpfn) +0.0462 worse; K − bar (hgb/lgbm) -0.0361 better; K − strong (tabpfn) +0.0382 worse; KC − strong (tabpfn) +0.0343 worse
- control: H − bar (lgbm) -0.0380 better; H − strong (tabpfn) +0.0358 worse; H − tabpfn (tabpfn) +0.0358 worse; K − bar (lgbm) -0.0441 better; K − strong (tabpfn) +0.0297 worse; KC − strong (tabpfn) +0.0277 worse
  - (H − bar) MAR minus control: +0.0098 [-0.0036, +0.0233] none
  - (H − strong) MAR minus control: +0.0103 [+0.0011, +0.0196] worse
  - (K − H) MAR minus control: -0.0018 [-0.0106, +0.0071] none

- (H − bar) MAR minus control: better 1 / no difference 3 / worse 0 (fold-pair t, seed agreement)
- (H − strong) MAR minus control: better 1 / no difference 1 / worse 2 (fold-pair t, seed agreement)
- (K − H) MAR minus control: better 0 / no difference 4 / worse 0 (fold-pair t, seed agreement)

Primary: worse MAR-minus-control contrasts: ['biodeg (H − strong) MAR minus control', 'kr-vs-kp (H − strong) MAR minus control'] -> loses ground (fold-pair t, seed agreement)
```

**Verdict: loses ground.** (H − strong), MAR minus control, is worse on two tables: biodeg
(+0.0410) and kr-vs-kp (+0.0103). The rule's threshold was two.

**Reading:**

- **Against the bar, the model keeps its standing under MAR.** (H − bar), MAR minus control, is
  never worse: better on kc2 and no difference on the other three. H beats the bar under both
  mechanisms on every table except kc2's control, where it shows no difference.
- **Against the strong tier, TabPFN each time, the direction depends on the table:**
  - **biodeg:** worse under MAR, with a wide interval [+0.0069, +0.0751]. H − TabPFN moves from
    −0.0212 (control) to +0.0198 (MAR), both no difference. K − strong moves from better to no
    difference.
  - **kr-vs-kp:** worse, and small (+0.0103, interval [+0.0011, +0.0196]). H − TabPFN moves from
    +0.0358 to +0.0462, worse under both.
  - **kc2:** better under MAR (−0.1124). The control is where H trails TabPFN most (+0.0975);
    under MAR, H − TabPFN is −0.0149, no difference. kc2 is the noisiest table at 522 rows.
  - **vehicle:** no difference (+0.0057).
- **The oracle's optimism does not change.** (K − H), MAR minus control, shows no difference on all
  four.
- **Unlike MNAR.** Under MNAR (E58), TabPFN gained no ground on the model, at 1 better, 9 no
  difference and 0 worse. Under MAR it gains on two of four tables and loses on one.
- **A hypothesis, not tested here.**
  - Under MAR, a column's test gaps sit in rows chosen by the drivers. TRIDENT's decode stage
    trains on masks drawn uniformly over observed cells, so its training gaps are spread in a
    different way from the test gaps.
  - TabPFN fits each column on its observed rows with the drivers as features. That fit is the
    one MAR leaves unbiased.
- **Limits:**
  - a smoke: four tables, one rate, one mechanism strength (slope 1.5), seeds 42/7/13;
  - the controls hide about 42% of the cells, not the shipped 60%;
  - scores are compared across mechanisms only as within-variant gaps.

**What it leads to (fixed in the ticket):**

- The contrasts are named as a limit. Under MAR, TRIDENT keeps its standing against the bar but
  can lose ground against TabPFN: biodeg and kr-vs-kp here, with kc2 the other way.
- **A fuller MAR family is proposed for the user to decide.**
  - It would add the 20nan rung and a slope dial, as E46 did for MNAR, and spambase and credit-g.
  - It would also test the hypothesis above with an arm whose decode masks follow the drivers.
- No default changes.
