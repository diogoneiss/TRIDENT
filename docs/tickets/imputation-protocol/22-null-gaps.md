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


## Outcome (2026-10-23, 21:53 GMT-3)

**The run:** 39 of 39 cells of N, 17:55–20:01 GMT-3 on four queues, with none failed. Every pair
found its three G seeds in E45 or E46.

**The printout** (`scripts/experiments/null_gaps_report.py --tally`, unedited):

```text
## `biodeg_probit0p25_60nan` against `biodeg_60nan`
- H: MNAR N − G +0.0297 [+0.0222, +0.0372] worse; MCAR N − G +0.0583 [+0.0422, +0.0744] worse; (N − G), MNAR minus MCAR -0.0286 [-0.0442, -0.0130] better
- K: MNAR N − G +0.0282 [+0.0194, +0.0371] worse; MCAR N − G +0.0489 [+0.0387, +0.0591] worse; (N − G), MNAR minus MCAR -0.0206 [-0.0354, -0.0059] better

## `biodeg_probit_60nan` against `biodeg_60nan`
- H: MNAR N − G +0.0659 [+0.0485, +0.0833] worse; MCAR N − G +0.0583 [+0.0422, +0.0744] worse; (N − G), MNAR minus MCAR +0.0076 [-0.0131, +0.0283] none
- K: MNAR N − G +0.0507 [+0.0403, +0.0611] worse; MCAR N − G +0.0489 [+0.0387, +0.0591] worse; (N − G), MNAR minus MCAR +0.0018 [-0.0127, +0.0164] none

## `biodeg_probit1p0_60nan` against `biodeg_60nan`
- H: MNAR N − G +0.0757 [+0.0670, +0.0844] worse; MCAR N − G +0.0583 [+0.0422, +0.0744] worse; (N − G), MNAR minus MCAR +0.0174 [+0.0029, +0.0319] worse
- K: MNAR N − G +0.0734 [+0.0643, +0.0824] worse; MCAR N − G +0.0489 [+0.0387, +0.0591] worse; (N − G), MNAR minus MCAR +0.0245 [+0.0106, +0.0384] worse

## `biodeg_probit2p0_60nan` against `biodeg_60nan`
- H: MNAR N − G +0.0987 [+0.0912, +0.1061] worse; MCAR N − G +0.0583 [+0.0422, +0.0744] worse; (N − G), MNAR minus MCAR +0.0404 [+0.0238, +0.0570] worse
- K: MNAR N − G +0.0875 [+0.0803, +0.0946] worse; MCAR N − G +0.0489 [+0.0387, +0.0591] worse; (N − G), MNAR minus MCAR +0.0386 [+0.0259, +0.0513] worse

## `spambase_probit_60nan` against `spambase_60nan`
- H: MNAR N − G +0.0188 [+0.0101, +0.0275] worse; MCAR N − G +0.0301 [+0.0012, +0.0590] worse; (N − G), MNAR minus MCAR -0.0113 [-0.0420, +0.0193] none
- K: MNAR N − G +0.0139 [+0.0105, +0.0173] worse; MCAR N − G +0.0048 [+0.0028, +0.0068] worse; (N − G), MNAR minus MCAR +0.0091 [+0.0048, +0.0133] worse

## `biodeg_probit_20nan` against `biodeg_20nan`
- H: MNAR N − G +0.0305 [+0.0133, +0.0478] worse; MCAR N − G +0.0256 [+0.0090, +0.0422] worse; (N − G), MNAR minus MCAR +0.0049 [-0.0191, +0.0290] none
- K: MNAR N − G +0.0247 [+0.0104, +0.0390] worse; MCAR N − G +0.0322 [+0.0004, +0.0639] worse; (N − G), MNAR minus MCAR -0.0074 [-0.0436, +0.0287] none

## `kc2_probit_20nan` against `kc2_20nan`
- H: MNAR N − G +0.0156 [-0.0100, +0.0412] none; MCAR N − G +0.0069 [-0.0169, +0.0306] none; (N − G), MNAR minus MCAR +0.0088 [-0.0245, +0.0420] none
- K: MNAR N − G +0.0079 [-0.0100, +0.0258] none; MCAR N − G -0.0000 [-0.0241, +0.0240] none; (N − G), MNAR minus MCAR +0.0079 [-0.0135, +0.0293] none

## `vehicle_probit_20nan` against `vehicle_20nan`
- H: MNAR N − G +0.0425 [+0.0104, +0.0746] worse; MCAR N − G +0.0167 [+0.0025, +0.0309] worse; (N − G), MNAR minus MCAR +0.0258 [-0.0076, +0.0593] none
- K: MNAR N − G +0.0235 [-0.0022, +0.0492] none; MCAR N − G +0.0068 [+0.0007, +0.0129] worse; (N − G), MNAR minus MCAR +0.0167 [-0.0078, +0.0413] none

## The dial (biodeg at 60nan), (N − G on H), MNAR minus MCAR by slope
- slope 0p25: better
- slope 0p5: none
- slope 1p0: worse
- slope 2p0: worse

- (N − G on H), MNAR minus MCAR: better 1 / no difference 5 / worse 2 (fold-pair t, seed agreement)
- (N − G on K), MNAR minus MCAR: better 1 / no difference 4 / worse 3 (fold-pair t, seed agreement)

Primary, (N − G on H), MNAR minus MCAR: better on ['biodeg_probit0p25_60nan'], worse on ['biodeg_probit1p0_60nan', 'biodeg_probit2p0_60nan'] -> refuted (fold-pair t, seed agreement)
```

**Verdict: refuted.** The interaction on H is better on 1 of 8 pairs (biodeg's slope 0.25) and
worse on 2 (slopes 1.0 and 2.0); the rule's refuted is better on at most 1.

**Reading:**

- **N is worse than G almost everywhere, under both mechanisms.** N − G on H is "worse" on 7 of 8
  MNAR variants and on 4 of 5 MCAR controls (kc2_20nan shows no difference either way). Under MCAR
  this replicates E32's choice in reverse: ADR 0014's `mask` beats `null` in the decode stage.
- **The dial runs against the hypothesis.** On biodeg at 60nan, N's deficit on H grows with the
  tilt: +0.030 at slope 0.25, +0.066 at 0.5, +0.076 at 1.0 and +0.099 at 2.0, against +0.058 under
  MCAR. So the interaction is better at the mildest tilt and worse at the two steepest. The
  steeper the value-dependent missingness, the more showing a gap as `[NULL]` costs, not the less.
- **K tells the same story** (better 1, worse 3): the slope-1.0 and slope-2.0 pairs and
  spambase_probit_60nan are worse.
- **E45/E46's Mx signal came from its other two changes.** Mx beat G at slope 2.0, but the
  `[NULL]` token alone is at its largest deficit there. The loss checkpoint and column-wise scoring
  carried Mx's gain, as the ticket's refuted branch says.
- **What it leaves:** the `[NULL]`/`[MASK]` split has no MNAR-specific gain on the imputation side
  under today's default. The thesis claim's test moves to the classification side (wiki t2). No
  default changes; ADR 0014 stands.
