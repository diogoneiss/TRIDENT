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
