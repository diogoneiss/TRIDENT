# Fresh MCAR draws: is a result tied to one missingness pattern? (E62; pre-registered 2026-10-22)

This ticket was written and committed before the study's first cell. The gate cell runs without
MLflow and is reported under "Launch".

- **Why:** this is T31. Each MCAR rate is a single draw from `_00nan`: `inject_nans` with seed
  42 + the rate.
  - The seeds vary the folds and the model, never which cells are missing.
  - The critique asked about it (`docs/reviews/imputation-critique/08-experimentation-design.md`),
    and the review of 2026-10-16 named it ("one MCAR draw each").
- **The user's go:** "Faça a t31 agora então: gere os geradores, runners e execute logo depois"
  (2026-10-22, about 06:40 GMT-3).
- **The code, with nothing in `src/`:**
  - `datasets/generate_mcar_draws.py`, the generator;
  - `scripts/experiments/mcar_draws_run.py`, the runner;
  - `scripts/experiments/mcar_draws_report.py`, the reader;
  - `scripts/experiments/run_mcar_draws_queue.sh`, the queues.

## Design (fixed before the first cell)

- **Variants:** pendigits_20nan, kr-vs-kp_40nan, credit-g_60nan and biodeg_80nan. These span the
  sizes, both column kinds and every rate.
- **Draws:**
  - **Draw 1 is the shipped file.** The generator regenerates it with seed 42 + pp and refuses to
    go on unless the two are equal cell for cell; all four are.
  - **Draws 2 and 3** use seeds 42 + pp + 1000 and 42 + pp + 2000. They are written as
    `<base>_d2_<pp>nan` and `<base>_d3_<pp>nan`, with the rate last, as the MNAR variants have it.
  - Every draw has the shipped per-column count, and its gaps overlap the shipped draw's at about
    the rate, as independent draws do. The data directory is gitignored; the generator is the
    record.
- **Cells:** 4 variants × 3 draws × seeds 42/7/13 make 36 cells of today's default.
  - A seed builds the same folds on all three draws, since folds depend on the labels only.
  - **Each draw trains with the configuration the shipped variant resolves,** passed as an override.
    So ADR 0018's file for pendigits_20nan reaches its draws too.
  - `--score_checkpoints` and `--score_calibrated` are on, as in E44 and E51.
  - ADR 0027's `auto` calibration resolves on each draw's measured rate, which equals the
    original's.
  - The strong tier is scored live in the run (ADR 0021). TabPFN comes from an offline pass
    (`tabpfn_run.py`) into the state directory's `tabpfn/`, one lane.
- **Draw 1 is rerun rather than reused.**
  - E35's runs at 42/7/13 predate ADR 0018 on pendigits_20nan and ADR 0019's removal of
    kr-vs-kp_40nan's file.
  - biodeg_80nan never ran at 42/7/13.
  - So the three draws come from one study on one commit.
- **The gate,** before launch, with `--no-mlflow`: credit-g_60nan, draw 1, seed 42, against E35's
  run, fold by fold on K. Any difference stops the launch, since it would mean the override path
  does not reproduce a default run.
- **Running:**
  - four queues, longest cell first, about 4 h;
  - one TabPFN lane beside them;
  - no cutoff;
  - state in `/scratch2/diogoneiss/trident-2026-10-22-mcar-draws/`.

## Measures and reading, fixed now (`mcar_draws_report.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI:

- **Per draw:**
  - K − bar;
  - K − strong (ADR 0028's tier, TabPFN included);
  - K − TabPFN;
  - KC − strong (KC is the default output at 30% and above, ADR 0027).
- **Across a variant's three draws, per contrast:**
  - **same:** one call;
  - **a reversal:** better in one draw and worse in another;
  - **a drift:** any other change.
  - **Descriptive:** the span of the three draws' means against the mean interval half-width.
- **Primary:** K − bar and K − strong on all four variants.
  - **holds:** no reversal;
  - **tied to the draw:** any reversal, each named.
- **Secondary:** the patterns on K − TabPFN and KC − strong, and every drift, listed.
- **What each reading leads to (no default changes from this study):**
  - **holds:** the thesis states that the calls do not depend on the MCAR draw on these four
    tables.
  - **tied to the draw:** each reversal is named as a limit in the thesis, its table's claims are
    qualified, and it gets a ticket.
- **An incomplete variant** (any draw without all three seeds) is reported as incomplete and not
  read.

## Launch

- **The gate passed (2026-10-22, 06:44 GMT-3).** credit-g_60nan, draw 1, seed 42, with
  `--no-mlflow`, equals E35's run fold by fold on K, with a largest difference of 0.
  - The configuration came from the task defaults through the override.
  - The measured rate, 0.6000, resolved `auto` to `gaps`.
  - The run scored the strong tier live.
- **Launched** on the commit of this ticket: four queues (`run_mcar_draws_queue.sh`) and
  `tabpfn_lane.sh`, both in the state directory.
