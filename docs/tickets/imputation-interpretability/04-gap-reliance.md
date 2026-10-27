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

## Outcome (2026-10-27, 12:29 GMT-3): mixed

- **Ran:** 66 of 66 cells, 2026-10-27, from 08:30 to 12:24 GMT-3; none failed. The mask arm's K
  reproduced E79's runs to 0.0 on the 12 MCAR-twin cells.
- **Primary: mixed, as the rule reads it.**
  - At slope 2.0, S is +0.0049 [+0.0015, +0.0083]: above 0, with every seed above 0, but not above
    the 0.01 SESOI.
  - At MCAR, S is −0.0022 [−0.0031, −0.0012], not above, as the rule wants.
- **The dial (descriptive) carries the signal the primary point misses.**
  - S is −0.002, +0.020, +0.035, +0.070 and +0.005 at slopes 0, 0.25, 0.5, 1.0 and 2.0.
  - Slopes 0.25 to 1.0 are well above the SESOI, rising, with every seed agreeing.
  - Then it collapses at 2.0. "Strictly rising" is false.
- **The pairs (descriptive):**
  - S is above the SESOI under MNAR on vehicle (+0.043) and kc2 (+0.020), and not on their MCAR
    twins (−0.005, −0.004).
  - spambase sits below the SESOI on both (+0.008 under MNAR, +0.001 under MCAR).
- **Accuracy (descriptive): the null arm is worse than the mask arm on all 11 variants,** under K
  (+0.005 to +0.097) and under KC.
  - That includes every MNAR variant, where it reads the pattern.
  - On biodeg the gap is largest at slope 2.0, but it does not rise steadily: K +0.056, +0.033,
    +0.051, +0.069 and +0.084 at slopes 0 to 2.0.
- **Why slope 2.0 collapses (post hoc, not tested):**
  - At slope 2.0 the observed 40% of each column is nearly all its lower part. The decode stage's
    targets, which are observed cells, then hardly ever show a large value, so a fill has little
    room to move up when `[NULL]` says "large".
  - Also, fewer pairs reach |ρ| ≥ 0.2 on the truncated training split: 295 a fold, against 564 at
    MCAR, with mean |ρ| unchanged at 0.42.
  - Neither is a verdict.
- **What it leads to:**
  - **The default cannot read the gap pattern, by construction.** This is a finding about the
    method, from the design, not the data.
  - **A decoder that can read it does so under moderate self-masking** (slopes 0.25–1.0, and the
    vehicle and kc2 pairs), and not under MCAR. It does not do so measurably at the strongest tilt,
    the point the rule named.
  - **Reading it does not pay:** the null arm is less accurate everywhere, MNAR included. ADR
    0014's mask default stands under MNAR too.
  - No default changes, and no ADR is proposed.

The reader's printout (`gap_reliance.py --tally`), unedited:

```text
- `biodeg_60nan`: S -0.0022 [-0.0031, -0.0012] seeds -0.0017, -0.0028, -0.0020 -> not above; accuracy (descriptive): K null − mask +0.0560 [+0.0461, +0.0659] worse; KC null − mask +0.0320 [+0.0240, +0.0400] worse
- `biodeg_probit0p25_60nan`: S +0.0201 [+0.0188, +0.0215] seeds +0.0210, +0.0194, +0.0200 -> above the SESOI; accuracy (descriptive): K null − mask +0.0329 [+0.0240, +0.0419] worse; KC null − mask +0.0256 [+0.0179, +0.0334] worse
- `biodeg_probit_60nan`: S +0.0351 [+0.0332, +0.0370] seeds +0.0360, +0.0356, +0.0336 -> above the SESOI; accuracy (descriptive): K null − mask +0.0509 [+0.0425, +0.0593] worse; KC null − mask +0.0432 [+0.0372, +0.0491] worse
- `biodeg_probit1p0_60nan`: S +0.0704 [+0.0600, +0.0808] seeds +0.0605, +0.0742, +0.0765 -> above the SESOI; accuracy (descriptive): K null − mask +0.0689 [+0.0609, +0.0768] worse; KC null − mask +0.0600 [+0.0532, +0.0668] worse
- `biodeg_probit2p0_60nan`: S +0.0049 [+0.0015, +0.0083] seeds +0.0022, +0.0073, +0.0051 -> not above; accuracy (descriptive): K null − mask +0.0843 [+0.0775, +0.0911] worse; KC null − mask +0.0796 [+0.0734, +0.0858] worse
- `vehicle_probit_60nan`: S +0.0430 [+0.0393, +0.0467] seeds +0.0437, +0.0444, +0.0409 -> above the SESOI; accuracy (descriptive): K null − mask +0.0712 [+0.0581, +0.0842] worse; KC null − mask +0.0658 [+0.0526, +0.0790] worse
- `vehicle_60nan`: S -0.0047 [-0.0069, -0.0025] seeds -0.0052, -0.0046, -0.0043 -> not above; accuracy (descriptive): K null − mask +0.0231 [+0.0114, +0.0349] worse; KC null − mask +0.0236 [+0.0115, +0.0356] worse
- `kc2_probit_60nan`: S +0.0204 [+0.0163, +0.0245] seeds +0.0199, +0.0215, +0.0198 -> above the SESOI; accuracy (descriptive): K null − mask +0.0965 [+0.0722, +0.1208] worse; KC null − mask +0.0943 [+0.0706, +0.1180] worse
- `kc2_60nan`: S -0.0040 [-0.0054, -0.0025] seeds -0.0051, -0.0039, -0.0028 -> not above; accuracy (descriptive): K null − mask +0.0626 [+0.0240, +0.1013] worse; KC null − mask +0.0474 [+0.0183, +0.0765] worse
- `spambase_probit_60nan`: S +0.0081 [+0.0071, +0.0091] seeds +0.0082, +0.0084, +0.0076 -> not above; accuracy (descriptive): K null − mask +0.0089 [+0.0059, +0.0119] worse; KC null − mask +0.0070 [+0.0042, +0.0098] worse
- `spambase_60nan`: S +0.0007 [+0.0002, +0.0011] seeds +0.0007, +0.0008, +0.0005 -> not above; accuracy (descriptive): K null − mask +0.0052 [+0.0018, +0.0086] worse; KC null − mask +0.0034 [+0.0004, +0.0063] worse

- mask arm's K against E79's run of the same cell, largest fold difference: 0.00e+00 over 12 cells
- dial (descriptive): S at slopes 0/0.25/0.5/1/2: -0.0022, +0.0201, +0.0351, +0.0704, +0.0049; strictly rising: False
- pairs (descriptive): vehicle_probit_60nan S above: True, vehicle_60nan S above: False; kc2_probit_60nan S above: True, kc2_60nan S above: False; spambase_probit_60nan S above: False, spambase_60nan S above: False

Primary: S at slope 2.0 +0.0049 [+0.0015, +0.0083], at MCAR -0.0022 [-0.0031, -0.0012] -> mixed (SESOI 0.01)
```
