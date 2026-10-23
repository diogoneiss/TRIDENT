# The MAR family: does MAR cost the model ground against the strong tier? (E64; pre-registered 2026-10-22)

This ticket was written and committed before the first cell.

- **Why:** E63's smoke read "loses ground" ([18](18-mar-smoke.md)). (H − strong), MAR minus
  control, was worse on biodeg and kr-vs-kp, better on kc2, and showed no difference on vehicle,
  at one rate and one draw on seeds 42/7/13. This family asks whether that holds on more tables,
  both rates, a new draw and fresh seeds.
- **The user's go:** "vamos fazer o 1 e 2 agora" (2026-10-22, about 18:55 GMT-3).
  - Item 1 is this family.
  - Item 2 is an arm whose decode masks follow the missingness propensity, E65. It is
    pre-registered in its own ticket before its first cell, once its flag is built, and runs after
    this study, so that `src/` is not edited while this study imports the checkout.
- **The code, with nothing in `src/`:**
  - `datasets/generate_mar_family.py`, the generator;
  - `scripts/experiments/mar_family_run.py`, the runner;
  - `scripts/experiments/mar_family_report.py`, the reader;
  - `scripts/experiments/run_mar_family_queue.sh`, the queues.

## Design (fixed before the first cell)

- **Tables and rates:** E63's four tables plus spambase and credit-g (mixed kinds), at 20nan and
  60nan. That makes 12 table-rates.
- **The generator** is E63's mechanism, with three differences:
  - a new seed (4060), so drivers and draws are independent of the smoke's;
  - separate random streams, so the slope dial shares one control and one set of driver scores;
  - the names `<base>_marf_<pp>nan` (MAR, slope 1.5) and `<base>_marfctl_<pp>nan` (control).
- **The dial:** biodeg_60nan at slopes 0.75 and 3 (`biodeg_marf0p75_60nan`, `biodeg_marf3_60nan`),
  against the same control.
- **The counts:** 30% of the feature columns are fully observed drivers. Every other column loses
  round(pct n) cells, so about 14% of feature cells at 20nan and about 42% at 60nan, in both MAR and
  control.
- **Its checks passed, before this ticket:**
  - counts and observed cells as specified;
  - the slope-1.5 shift in the driver score, hidden minus observed, between +0.363 (spambase 20nan)
    and +0.922 (credit-g 60nan), above the 0.3 floor;
  - every control within ±0.044, under the 0.1 ceiling;
  - the dial on biodeg_60nan: +0.454 at slope 0.75, +0.707 at 1.5 and +1.000 at 3.
- **Cells:** 26 variants × seeds 404/505/606 make 78 cells of today's default (arm G).
  - Each runs with `<base>_<pp>nan`'s resolved configuration as an override, with
    `--score_checkpoints` and `--score_calibrated`.
  - ADR 0027's `auto` resolves to `none` at 20nan (about 0.14 missing) and to `gaps` at 60nan
    (about 0.42).
  - TabPFN comes from an offline pass, two lanes.
- **Running:**
  - four queues, about 4 h;
  - two TabPFN lanes;
  - no cutoff;
  - state in `/scratch2/diogoneiss/trident-2026-10-22-mar-family/`.

## Measures and reading, fixed now (`mar_family_report.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule:

- **Per mechanism, with the 1% SESOI:** H − bar, H − strong, H − TabPFN, K − bar, K − strong and
  KC − strong.
- **The mechanism effect, fold-paired:** (H − bar), (H − strong) and (K − H), MAR minus control.
- **Primary,** over the 12 table-rates at slope 1.5, (H − strong) MAR minus control:
  - **supported** (MAR costs the model ground against the strong tier): worse on 4 or more, and
    better on at most 1;
  - **refuted:** worse on at most 1;
  - **mixed:** otherwise.
- **Secondary:**
  - (H − bar) MAR minus control, tallied the same way;
  - (K − H) MAR minus control;
  - biodeg_60nan and kr-vs-kp_60nan, the smoke's two, on the new draw and seeds;
  - the dial, descriptive: does (H − strong) MAR minus control grow with the slope?
- **What each reading leads to (no default changes):**
  - **supported:** MAR against the strong tier is a stated limit. E65's propensity-mask arm is the
    test of the hypothesis that the decode masks' mismatch causes it.
  - **refuted:** the smoke's two were specific to its draw and seeds. The thesis states that the
    standing holds under MAR, and E65 is read as exploratory.
  - **mixed:** reported, and the user chooses.

## Outcome (2026-10-23, 04:19 GMT-3)

**The run:** 78 of 78 cells and 78 of 78 TabPFN pairs.

- The queues ran from 18:57 GMT-3 on 2026-10-22 to 03:48 GMT-3 on 2026-10-23.
- **Two cells failed with CUDA out of memory** at 20:26 GMT-3: spambase_marfctl_20nan at s404 and
  s505.
  - The GPU then held E64's four queues, E65's two and both TabPFN lanes. One TabPFN process held
    8.6 GB.
  - That was my scheduling, not the design. Both were rerun unchanged by relaunching their
    queues, 03:49–04:18 GMT-3. The verdict was read only once all 78 were in.

**The printout** (`scripts/experiments/mar_family_report.py --tally`, unedited):

```text
## `biodeg_20nan`
- MAR: H − bar (hgb/lgbm) -0.0126 none; H − strong (missforest/tabpfn) -0.0221 none; H − tabpfn (tabpfn) -0.0384 none; K − bar (hgb/lgbm) -0.0032 none; K − strong (missforest/tabpfn) -0.0128 none; KC − strong (missforest/tabpfn) -0.0106 none
- control: H − bar (hgb/lgbm) -0.0217 none; H − strong (missforest/tabpfn) -0.0210 none; H − tabpfn (tabpfn) -0.0247 none; K − bar (hgb/lgbm) -0.0211 none; K − strong (missforest/tabpfn) -0.0204 none; KC − strong (missforest/tabpfn) -0.0208 none
  - (H − bar) MAR minus control: +0.0091 [-0.0220, +0.0402] none
  - (H − strong) MAR minus control: -0.0012 [-0.0575, +0.0552] none
  - (K − H) MAR minus control: +0.0088 [-0.0211, +0.0387] none

## `biodeg_60nan`
- MAR: H − bar (hgb/lgbm) -0.0097 none; H − strong (tabpfn) -0.0042 none; H − tabpfn (tabpfn) -0.0042 none; K − bar (hgb/lgbm) -0.0258 better; K − strong (tabpfn) -0.0203 none; KC − strong (tabpfn) -0.0273 better
- control: H − bar (hgb/lgbm) -0.0495 better; H − strong (tabpfn) -0.0333 better; H − tabpfn (tabpfn) -0.0333 better; K − bar (hgb/lgbm) -0.0576 better; K − strong (tabpfn) -0.0414 better; KC − strong (tabpfn) -0.0445 better
  - (H − bar) MAR minus control: +0.0398 [+0.0186, +0.0610] worse
  - (H − strong) MAR minus control: +0.0291 [+0.0081, +0.0500] worse
  - (K − H) MAR minus control: -0.0080 [-0.0242, +0.0081] none

## `kc2_20nan`
- MAR: H − bar (hgb/knn5) -0.1121 better; H − strong (tabpfn) +0.0889 worse; H − tabpfn (tabpfn) +0.0889 worse; K − bar (hgb/knn5) -0.1713 better; K − strong (tabpfn) +0.0296 none; KC − strong (tabpfn) +0.0262 none
- control: H − bar (knn5) +0.0118 none; H − strong (ice/tabpfn) +0.1216 worse; H − tabpfn (tabpfn) +0.1175 worse; K − bar (knn5) -0.0003 none; K − strong (ice/tabpfn) +0.1095 worse; KC − strong (ice/tabpfn) +0.1025 worse
  - (H − bar) MAR minus control: -0.1239 [-0.2481, +0.0003] none
  - (H − strong) MAR minus control: -0.0327 [-0.1303, +0.0648] none
  - (K − H) MAR minus control: -0.0471 [-0.1223, +0.0281] none

## `kc2_60nan`
- MAR: H − bar (hgb/lgbm) -0.0888 better; H − strong (ice) -0.0500 none; H − tabpfn (tabpfn) -0.0682 better; K − bar (hgb/lgbm) -0.1101 better; K − strong (ice) -0.0713 better; KC − strong (ice) -0.0730 better
- control: H − bar (knn5) -0.1348 better; H − strong (tabpfn) -0.0421 none; H − tabpfn (tabpfn) -0.0421 none; K − bar (knn5) -0.1556 better; K − strong (tabpfn) -0.0629 none; KC − strong (tabpfn) -0.0646 better
  - (H − bar) MAR minus control: +0.0460 [+0.0058, +0.0862] worse
  - (H − strong) MAR minus control: -0.0079 [-0.0732, +0.0573] none
  - (K − H) MAR minus control: -0.0005 [-0.0532, +0.0521] none

## `vehicle_20nan`
- MAR: H − bar (hgb/lgbm) -0.0436 better; H − strong (ice) +0.0080 none; H − tabpfn (tabpfn) -0.0061 none; K − bar (hgb/lgbm) -0.0450 better; K − strong (ice) +0.0066 none; KC − strong (ice) +0.0069 none
- control: H − bar (lgbm) -0.0517 better; H − strong (tabpfn) +0.0105 none; H − tabpfn (tabpfn) +0.0105 none; K − bar (lgbm) -0.0567 better; K − strong (tabpfn) +0.0055 none; KC − strong (tabpfn) +0.0062 none
  - (H − bar) MAR minus control: +0.0080 [-0.0271, +0.0431] none
  - (H − strong) MAR minus control: -0.0025 [-0.0433, +0.0382] none
  - (K − H) MAR minus control: +0.0037 [-0.0065, +0.0138] none

## `vehicle_60nan`
- MAR: H − bar (hgb/lgbm) -0.0295 better; H − strong (tabpfn) +0.0208 none; H − tabpfn (tabpfn) +0.0208 none; K − bar (hgb/lgbm) -0.0349 better; K − strong (tabpfn) +0.0154 none; KC − strong (tabpfn) +0.0147 none
- control: H − bar (hgb/lgbm) -0.0553 better; H − strong (tabpfn) +0.0234 worse; H − tabpfn (tabpfn) +0.0234 worse; K − bar (hgb/lgbm) -0.0591 better; K − strong (tabpfn) +0.0196 worse; KC − strong (tabpfn) +0.0181 worse
  - (H − bar) MAR minus control: +0.0258 [+0.0084, +0.0432] worse
  - (H − strong) MAR minus control: -0.0026 [-0.0329, +0.0277] none
  - (K − H) MAR minus control: -0.0016 [-0.0177, +0.0145] none

## `kr-vs-kp_20nan`
- MAR: H − bar (lgbm) -0.0191 better; H − strong (tabpfn) +0.0037 none; H − tabpfn (tabpfn) +0.0037 none; K − bar (lgbm) -0.0180 better; K − strong (tabpfn) +0.0048 none; KC − strong (tabpfn) +0.0151 worse
- control: H − bar (hgb/lgbm) -0.0112 none; H − strong (tabpfn) +0.0107 none; H − tabpfn (tabpfn) +0.0107 none; K − bar (hgb/lgbm) +0.0051 none; K − strong (tabpfn) +0.0269 worse; KC − strong (tabpfn) +0.0344 worse
  - (H − bar) MAR minus control: -0.0079 [-0.0335, +0.0176] none
  - (H − strong) MAR minus control: -0.0070 [-0.0265, +0.0125] none
  - (K − H) MAR minus control: -0.0152 [-0.0340, +0.0037] none

## `kr-vs-kp_60nan`
- MAR: H − bar (hgb/lgbm) -0.0360 better; H − strong (tabpfn) +0.0336 worse; H − tabpfn (tabpfn) +0.0336 worse; K − bar (hgb/lgbm) -0.0370 better; K − strong (tabpfn) +0.0326 worse; KC − strong (tabpfn) +0.0236 worse
- control: H − bar (lgbm) -0.0396 better; H − strong (tabpfn) +0.0316 worse; H − tabpfn (tabpfn) +0.0316 worse; K − bar (lgbm) -0.0385 better; K − strong (tabpfn) +0.0327 worse; KC − strong (tabpfn) +0.0269 worse
  - (H − bar) MAR minus control: +0.0036 [-0.0097, +0.0170] none
  - (H − strong) MAR minus control: +0.0020 [-0.0091, +0.0131] none
  - (K − H) MAR minus control: -0.0021 [-0.0136, +0.0095] none

## `spambase_20nan`
- MAR: H − bar (hgb/lgbm) -0.0064 none; H − strong (missforest) -0.0011 equivalent; H − tabpfn (tabpfn) -0.0860 better; K − bar (hgb/lgbm) -0.0072 none; K − strong (missforest) -0.0019 none; KC − strong (missforest) +0.0006 equivalent
- control: H − bar (hgb/lgbm) -0.0017 none; H − strong (missforest) -0.0015 none; H − tabpfn (tabpfn) -0.0855 better; K − bar (hgb/lgbm) -0.0086 better; K − strong (missforest) -0.0085 better; KC − strong (missforest) +0.0005 equivalent
  - (H − bar) MAR minus control: -0.0048 [-0.0185, +0.0089] none
  - (H − strong) MAR minus control: +0.0004 [-0.0099, +0.0107] none
  - (K − H) MAR minus control: +0.0062 [-0.0046, +0.0170] none

## `spambase_60nan`
- MAR: H − bar (hgb/lgbm) -0.0211 better; H − strong (tabpfn) -0.0373 better; H − tabpfn (tabpfn) -0.0373 better; K − bar (hgb/lgbm) -0.0239 better; K − strong (tabpfn) -0.0402 better; KC − strong (tabpfn) -0.0429 better
- control: H − bar (hgb/lgbm) -0.0147 better; H − strong (tabpfn) -0.0355 better; H − tabpfn (tabpfn) -0.0355 better; K − bar (hgb/lgbm) -0.0190 better; K − strong (tabpfn) -0.0398 better; KC − strong (tabpfn) -0.0419 better
  - (H − bar) MAR minus control: -0.0063 [-0.0166, +0.0039] none
  - (H − strong) MAR minus control: -0.0018 [-0.0127, +0.0092] none
  - (K − H) MAR minus control: +0.0014 [-0.0069, +0.0098] none

## `credit-g_20nan`
- MAR: H − bar (hgb/lgbm) -0.0409 better; H − strong (tabpfn) +0.0224 worse; H − tabpfn (tabpfn) +0.0224 worse; K − bar (hgb/lgbm) -0.0419 better; K − strong (tabpfn) +0.0213 worse; KC − strong (tabpfn) +0.0201 worse
- control: H − bar (hgb/lgbm) -0.0268 better; H − strong (tabpfn) +0.0064 none; H − tabpfn (tabpfn) +0.0064 none; K − bar (hgb/lgbm) -0.0148 none; K − strong (tabpfn) +0.0183 worse; KC − strong (tabpfn) +0.0179 worse
  - (H − bar) MAR minus control: -0.0142 [-0.0364, +0.0081] none
  - (H − strong) MAR minus control: +0.0160 [-0.0005, +0.0325] none
  - (K − H) MAR minus control: -0.0130 [-0.0375, +0.0114] none

## `credit-g_60nan`
- MAR: H − bar (knn10) -0.0190 better; H − strong (tabpfn) +0.0386 worse; H − tabpfn (tabpfn) +0.0386 worse; K − bar (knn10) -0.0295 better; K − strong (tabpfn) +0.0282 worse; KC − strong (tabpfn) +0.0245 worse
- control: H − bar (knn10) -0.0132 better; H − strong (tabpfn) +0.0349 worse; H − tabpfn (tabpfn) +0.0349 worse; K − bar (knn10) -0.0195 better; K − strong (tabpfn) +0.0286 worse; KC − strong (tabpfn) +0.0283 worse
  - (H − bar) MAR minus control: -0.0059 [-0.0275, +0.0157] none
  - (H − strong) MAR minus control: +0.0036 [-0.0123, +0.0196] none
  - (K − H) MAR minus control: -0.0041 [-0.0245, +0.0164] none

## The dial: `biodeg_60nan`, (H − strong) MAR minus control by slope
- slope 0p75: +0.0285 [-0.0033, +0.0604] none
- slope 1p5: +0.0291 [+0.0081, +0.0500] worse
- slope 3: +0.0181 [-0.0033, +0.0395] none

- (H − bar) MAR minus control: better 0 / no difference 9 / worse 3 (fold-pair t, seed agreement)
- (H − strong) MAR minus control: better 0 / no difference 11 / worse 1 (fold-pair t, seed agreement)
- (K − H) MAR minus control: better 0 / no difference 12 / worse 0 (fold-pair t, seed agreement)

Primary, (H − strong) MAR minus control: worse on ['biodeg_60nan'], better on none -> refuted (fold-pair t, seed agreement)
```

**Verdict: refuted.** (H − strong), MAR minus control, is worse on 1 of the 12 table-rates
(biodeg_60nan) and better on none. The rule's floor for "supported" was 4.

**Reading:**

- **Against the strong tier, MAR does not broadly cost the model ground.**
  - biodeg_60nan replicates the smoke on a new draw and new seeds: +0.0291, against +0.0410 in E63.
  - kr-vs-kp_60nan does not replicate (+0.0020, no difference).
  - The other ten table-rates show no difference.
- **Against the bar, MAR costs some ground at 60nan.** (H − bar), MAR minus control, is worse on
  biodeg_60nan, kc2_60nan and vehicle_60nan: the three small numerical tables at the higher rate.
  - It is no difference on the other nine, and better nowhere.
  - In E63's smoke it was never worse, so this is new at 60nan on these tables.
- **The oracle's optimism does not move.** (K − H), MAR minus control, shows no difference on all
  12.
- **The dial on biodeg_60nan is not monotone** (descriptive): +0.0285 at slope 0.75 (no
  difference), +0.0291 at 1.5 (worse), +0.0181 at 3 (no difference).
- **The smoke's verdict, "loses ground", does not survive the family.** It rested on biodeg and
  kr-vs-kp at one draw and seed set, and only biodeg comes back.

**What it leads to (fixed in the ticket):**

- The thesis states that, under MAR, TRIDENT's standing against the strong tier holds on these six
  tables at 20 and 60nan.
  - biodeg_60nan is the exception, on two draws.
  - The bar gains on three small numerical tables at 60nan.
- E65 is read as exploratory.
- No default changes.
