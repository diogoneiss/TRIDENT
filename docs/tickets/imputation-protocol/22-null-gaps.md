# The gaps as `[NULL]` in the decode stage, alone: does the token help under MNAR? (E67; pre-registered 2026-10-23)

This ticket was written and committed before the study's first cell. The gate cell runs without
MLflow and is reported under "Launch".

- **Why:** this is T33, accepted by the user on 2026-10-22.
  - E45 and E46's t2 readout, Mx, changed three things at once: the gaps shown as `[NULL]` in the
    decode stage, the loss checkpoint, and column-wise scoring. It does not isolate the
    `[NULL]`/`[MASK]` split, which the user takes as TRIDENT's main claim.
  - On biodeg's dial, Mx's deficit to G's H shrank against MCAR at every slope, and at slope 2.0
    Mx beat G by 0.006: the only variant of E45 and E46 where the untied arm won.
- **The user's go:** "vamos na parte de imputacao agora entao" (2026-10-23, about 17:50 GMT-3).
  That was the answer to the proposal to run T33 on the GPU and T34 on the CPU.
- **The code, with nothing in `src/`:**
  - `scripts/experiments/null_gaps_run.py`, the runner;
  - `scripts/experiments/null_gaps_report.py`, the reader;
  - `scripts/experiments/run_null_gaps_queue.sh`, the queues.

## Design (fixed before the first cell)

- **Arm N:** today's default with `DECODE_GAP_TOKEN null`, and nothing else.
  - A real gap is shown as the stored `[NULL]` instead of `[MASK]` in the decode stage, in
    training and in scoring alike.
  - N keeps today's checkpoint (K) and all-at-once scoring, with `--score_checkpoints` as E45 and
    E46 ran it.
  - Under MCAR this is E32's comparison in reverse; ADR 0014 adopted `mask` there.
- **The comparison is G:** E45's and E46's runs of today's default (arm G), same variant, seed and
  folds. Every G cell exists.
- **Eight MNAR pairs, 13 variants, seeds 42/7/13, 39 cells of N:**
  - biodeg's tilt dial at 60nan: `biodeg_probit0p25_60nan`, `biodeg_probit_60nan` (slope 0.5,
    E45), `biodeg_probit1p0_60nan` and `biodeg_probit2p0_60nan`, all against E45's
    `biodeg_60nan`;
  - E46's four new pairs: `spambase_probit_60nan` against `spambase_60nan`, and
    `<base>_probit_20nan` against `<base>_20nan` for biodeg, kc2 and vehicle.
- **The gate,** before launch, with `--no-mlflow`: biodeg_probit_60nan, seed 42, arm G, against E45's
  run, fold by fold on K and H. Any difference stops the launch, since the pairing with E45 and E46's
  G would then not hold.
- **Running:** four queues, about 2 h, with no cutoff. State in
  `/scratch2/diogoneiss/trident-2026-10-23-null-gaps/`.

## Measures and reading, fixed now (`null_gaps_report.py --tally`)

On the induced `impute_score` over 15 fold pairs, with the verdict rule:

- **Per pair and mechanism, with the 1% SESOI of G's value:** N − G on H (the readout E45 and E46
  read) and on K.
- **The interaction, fold-paired:** (N − G), MNAR minus MCAR, on H and on K.
- **Primary,** the interaction on H over the 8 pairs:
  - **supported** (the `[NULL]` token helps more under value-dependent missingness): better on 3 or
    more, and worse on none;
  - **refuted:** better on at most 1;
  - **mixed:** otherwise.
- **The dial,** descriptive: the interaction on H by slope (0.25, 0.5, 1.0, 2.0). Does it grow with
  the tilt?
- **What each reading leads to (no default changes; ADR 0014 stands):**
  - **supported:** the thesis has an imputation-side reading of the `[NULL]`/`[MASK]` split. Under
    value-dependent missingness, showing a gap as a gap helps relative to MCAR. The classification
    side (wiki t2) is next.
  - **refuted:** the split gives no MNAR-specific gain on imputation, and E45 and E46's Mx signal
    came from its other two changes.
  - **mixed:** reported, and the user chooses.

## Launch

- **The gate passed (2026-10-23, 17:55 GMT-3).** biodeg_probit_60nan, seed 42, arm G, with
  `--no-mlflow`, equals E45's G run fold by fold on K and on H, with a largest difference of 0
  on both. So the pairing with E45's and E46's G holds at today's commit.
- **Launched** on the commit of this ticket: four queues (`run_null_gaps_queue.sh`), beside
  E68's CPU lanes.
