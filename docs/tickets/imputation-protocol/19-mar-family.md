# The MAR family: does MAR cost the model ground against the strong tier? (E64; pre-registered 2026-10-22)

This ticket was written and committed before the first cell.

- **Why:** E63's smoke read "loses ground" ([18](18-mar-smoke.md)). (H − strong), MAR minus
  control, was worse on biodeg and kr-vs-kp, better on kc2, and showed no difference on vehicle,
  at one rate and one draw on seeds 42/7/13. This family asks whether that holds on more tables,
  both rates, a new draw and fresh seeds.
- **The user's go:** "vamos fazer o 1 e 2 agora" (2026-10-22, about 15:20 GMT-3).
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
