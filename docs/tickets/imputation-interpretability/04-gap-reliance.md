# Does a decoder that can tell a real gap from its own masks read the gap pattern under MNAR? (E81; pre-registered 2026-10-27)

This ticket was written and committed before the study's first cell.

- **Why:** T38's study F asks how much a fill relies on the gap pattern, which an informative
  mechanism (MNAR) makes worth reading.
- **The question cannot be asked of today's default model, and this is a finding in itself.**
  - Under `DECODE_GAP_TOKEN mask` (ADR 0014), a real gap and a cell the decode stage hides on
    purpose are the same input token. No change to the input can vary "truly missing" while holding
    "value unknown" fixed.
  - A signed shift from hiding a column under self-masking would also be confounded. The observed
    sample is skewed, since the large values are the missing ones.
- **The flag that makes it askable already exists:** `--decode_gap_token null`, the behaviour before
  ADR 0014.
  - It shows real gaps as `[NULL]` and the stage's own masks as `[MASK]`.
  - So it can learn that `[NULL]` at d means "d was likely large".
- **The user's go:** "otimo, quero fazer E e F agora, vá" (2026-10-27, before 08:29 GMT-3).
- **Code:** no change to `src/`.
  - `scripts/experiments/gap_reliance.py` reads the null arm's model (K) in place after each fold,
    random states restored, as E78 does.
  - `run_gap_reliance_queue.sh` runs four queues.
  - The views are built at the pandas level and encoded, the way the column-wise scoring builds
    them. The unhidden all-`[MASK]` view is asserted to equal the fold ledger's fill.
- **Smoke, outside the study:** biodeg_probit_20nan at seed 1, arm null, `--no-mlflow`.
  - It ran end to end (7 min) and wrote five reliance files. The fill assert passed on every fold.
  - Only the files' shapes were printed: 152 pairs over 36 targets in fold 1, every pair with at
    least 10 cells. No value was read.

## The measure

For a numerical target j, the test rows are shown with every gap as `[NULL]` and j's gaps as
`[MASK]`. This is the column-wise view, the null model's own shape.
- Then one other numerical column d has its gaps switched to `[MASK]` too. The pair qualifies when
  |Spearman(j, d)| ≥ 0.2 on the training split.
- d's value is absent either way; only the "truly missing" label changes.
- **The pair's shift:** over the rows where both j and d are gaps (at least 10), the mean of
  sign(ρ) × (fill with d as `[NULL]` − fill with d as `[MASK]`), in scaled units.
  - Under self-masking it is positive where the model reads `[NULL]` at d as "d was large".
  - Under MCAR, where the label carries nothing, it should be about 0.
- **One fold's S:** the mean over the qualifying pairs.
- **The mask arm:** S is not computed. A `[NULL]` in its decode inputs would be outside its training
  distribution.

## Design (fixed before the first cell)

- **Variants, all at 60nan, from `datasets/generate_mnar.py` (E45/E46):**
  - **the dial:** biodeg at MCAR (`biodeg_60nan`) and at slopes 0.25, 0.5, 1.0 and 2.0;
  - **breadth:** the probit pairs (slope 0.5) for vehicle, kc2 and spambase, each with its MCAR
    `_60nan` twin.
- **Arms:**
  - **null:** `--decode_gap_token null`;
  - **mask:** today's default.
  - Everything else is today's default configuration.
- **Seeds and size:** 101, 202 and 303, five folds; 11 variants × 2 arms × 3 seeds = 66 cells.
- **Running:** four GPU queues, MLflow tag `experiment=gap-reliance-2026-10-27` and `arm=null|mask`,
  state in `/scratch2/diogoneiss/trident-2026-10-27-gap-reliance/`. About 3.5–4 h; the 12 spambase
  cells set it.

## Rule, fixed now (`gap_reliance.py --tally`)

- **Primary, on the dial (arm null), SESOI 0.01 scaled units:**
  - **supported:** at slope 2.0, S's 95% t-interval over 15 folds lies above +0.01 with every
    seed's mean above 0, *and* at MCAR S's interval does not lie above +0.01;
  - **refuted:** at slope 2.0, S's interval does not lie above 0;
  - **mixed:** otherwise. For example, S above the SESOI at MCAR as well, which would mean the
    model treats the two tokens differently for reasons other than the mechanism.
- **Descriptive:**
  - S along the dial and whether it rises strictly. E46's strict dose-response was "no" on another
    measure, so this is not primary.
  - S on the three probit pairs beside their MCAR twins.
  - Accuracy, null − mask, under K and KC, fold-paired with the 1% SESOI, per variant. Does the arm
    that can read the pattern gain under MNAR and lose under MCAR, as E32 found under MCAR?
  - The mask arm's K against E79's run of the same cell, on the four MCAR twins at 101/202/303.

**What each reading leads to:**
- **Supported:** the thesis can say that the default cannot read the gap pattern, by construction.
  A decoder that can do so uses it under MNAR and not under MCAR.
  - The accuracy comparison says whether it pays. Any change to a default goes through an ADR the
    user decides.
- **Refuted:** even a decoder able to read the pattern does not, at the strongest tilt. The MNAR
  cost (E46) is then not a reading the default is missing.

## Limits fixed now

- K only, as in E78.
- Self-masking (probit) MNAR only, at 60nan.
- The tilt dial is on one table (biodeg); the pairs add three tables at one slope.
