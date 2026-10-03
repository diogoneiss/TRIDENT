# Does choosing the checkpoint, or calibrating the guesses, on the validation gaps help? (pre-registered 2026-10-03)

This ticket was written and committed before the first run, at 12:52 GMT-3 on 2026-10-03 (commit
`4cfde31`; the queues started 12:52:04 GMT-3, so about 18:30–19:00 GMT-3 is the expected end). The user chose these two ideas on 2026-10-03 after E33, to be
studied on top of ADR 0014's default.

## Why

Two weaknesses of the decode stage were found in E30's ledgers (diagnosis of 2026-10-02) and
neither is about the token shape that T02 fixed:

- **The checkpoint is chosen on a few noisy cells of the wrong population.** The decode stage
  keeps the epoch of the lowest loss on the fixed validation mask. At high missingness that mask
  hides few cells (about 4% of an 80nan row: some 80 cells on credit-g's validation split), while
  the same split holds an order of magnitude more of its own gaps, the population the headline
  scores. E06 found the two criteria disagree (ρ 0.605), and E32's P arm blew up on three folds
  of spambase_60nan at the epoch the loss chose (T07).
- **The guesses are overconfident.** Regressing the true value on the guess, per column and
  fold, gave a median slope of 0.85; in 46% of column-folds it was under 0.8. A categorical guess
  also answers where the training mode would have been right more often. Estimated on four test
  folds of E30's arm M, shrinking the numbers and falling back to the mode at low confidence
  moved the mean score over the 21 variants from 0.7745 to about 0.767, and hurt kc2.

Both can be measured on one decode trajectory, without changing training: the checkpoint is
only which saved weights are scored, and the calibration only post-processes their guesses.

## Design (fixed before launch)

- **One run per cell, four readouts of the same trajectory.** A plain imputation run under
  today's defaults (ADR 0014), reading the variant's promoted file or the defaults as any run
  does, five folds, with `--score_induced_checkpoint` and `--score_calibrated`:
  - **H**, the headline: the epoch of the lowest validation-mask loss, guesses as they are;
  - **C**, H's guesses calibrated on the validation rows' own gaps;
  - **K**, the epoch whose validation rows' own gaps score best (the induced `impute_score`
    against the complete sibling, ADR 0008's population), guesses as they are;
  - **KC**, K's guesses calibrated on K's own validation gaps.
- **The calibration, fixed now** (`src/training/calibration.py`):
  - a number `p` becomes `b + α (p − b)`, with `b` the column's training mean; `α` is the
    least-squares factor of the truth on the guess over the column's validation gap cells,
    clipped to [0, 1], then pulled toward 1 as `(n α + 30) / (n + 30)` for a column with `n`
    such cells;
  - a category whose confidence is below a threshold is answered with the training mode; the
    threshold is chosen among none and the 0.1, 0.2, 0.3 and 0.5 quantiles of the column's
    validation confidences as the one with the lowest validation error, only in a column with at
    least 30 validation gap cells; a tie keeps the guesses.
- **Variants:** all 21 of ADR 0007. **Seeds:** 101, 202, 303, so that H must equal E33's arm-G
  cells to the last digit (the integrity check: a single differing cell means the diagnostics
  leaked into training). **Size:** 63 cells, tagged `experiment=selection-2026-10-03`.
- **Running it:** four concurrent queues (`scripts/experiments/run_selection_queue.sh 1..4`),
  one trainer each, CPU threads capped at 4; cells assigned longest first by E30's arm-M cell
  times, about 313 minutes of those per queue. E33's cells ran at those times; the per-epoch
  validation scoring adds an estimated 10 to 15%, so about 5.5 to 6 h of wall time is expected.
  No cell starts after 02:00 GMT-3 on 2026-10-04 (05:00 UTC); a cell not started by then is
  reported missing. Nothing is rerun or extended on its result; a crash is fixed, its cell rerun
  from the start, and the rerun noted here.
- **Checks before launch:** smoke runs of credit-g_20nan and kr-vs-kp_40nan at seed 101 with
  both flags must equal E33's G cells on every `impute/*` headline metric and carry the four
  diagnostic families; their diagnostic values were not read. The unit suite (mypy included)
  and both regression fixtures must pass.

## Measures

- **Primary:** per variant, C − H and K − H on the induced `impute_score`, paired by seed and
  fold (15 pairs within runs), t-interval; a verdict only when the interval excludes zero and all
  three seeds' mean difference agree in sign. A negative difference favours the candidate.
- **Reading, fixed now, on all 21 variants** (neither mechanism is specific to the missing
  level):
  1. A candidate (C or K) **qualifies** if no variant shows "H better".
  2. C and K are independent mechanisms, so each qualifying one is recommended on its own.
  3. **KC** is recommended over them only if it qualifies against H and shows no "K better" or
     "C better" verdict against either.
  4. If neither C nor K qualifies, nothing changes.
  A default switch for either needs its own ADR and the user's decision: K changes model
  selection (it would come with a flag and a tag), C redefines what the model's guess is.
- **Secondary, descriptive:**
  - KC − H, KC − K, KC − C, the same way;
  - K's masked population against H's;
  - the two criteria's chosen epochs (`decode/loss_checkpoint_epoch` against
    `decode/induced_checkpoint_epoch`);
  - the calibration's reach: mean `α` and how many categorical columns got a threshold, the
    honest companion of any gain, since the 30-cell guards leave thin columns untouched;
  - differences by missing level; means over the 21 variants; variants below the best
    baseline under each readout; time.
- **Analysis:** `scripts/experiments/selection_report.py` (`--tally` for the counts and the
  integrity check).

## Known limits, stated in advance

- **The validation gaps are spent twice in KC** (to choose the epoch and to fit the
  calibration), once in C and K. The test split is never touched by either.
- **The validation split has about a quarter of the cells the ledger estimate fitted on**, so
  the calibration's gain should be smaller than estimated, and thin columns stay as they were.
- **Multiplicity:** 42 primary tests (two candidates on 21 variants) with the seed-agreement
  rule, plus the secondaries.
- **Within-run pairs:** like E31, each pair is one trajectory read twice, so the intervals are
  narrow and a small effect can earn a verdict.

## Running notes

- **Checks before launch, done:** the smoke runs of credit-g_20nan and kr-vs-kp_40nan at seed
  101 with both flags equal E33's G cells on every `impute/*` headline metric (55 and 35) and
  carry the four diagnostic families; their diagnostic values were not read. The unit suite
  (mypy included, 292 tests) and both regression fixtures pass.

## Outcome (2026-10-03, 19:00 GMT-3)

All 63 cells ran (12:52 to 18:43 GMT-3), none failed, none missing. The integrity check passed:
every cell's headline (H), induced and masked, equals E33's arm-G cell to the last digit, so the
diagnostics did not touch training. Numbers from `scripts/experiments/selection_report.py` (kept
with `--tally` in the state directory). Lower is better; a negative difference favours the
candidate. Induced cells, 15 within-run fold pairs per variant; the epochs are fold means.

| variant | H | C | K | KC | best baseline | C − H | verdict | K − H | verdict | epoch, loss → gaps |
|---|---|---|---|---|---|---|---|---|---|---|
| `credit-g_20nan` | 0.8817 | 0.8801 | 0.8751 | 0.8738 | 0.9138 | −0.0016 [−0.0030, −0.0002] | C better | −0.0066 [−0.0177, +0.0044] | none | 91 → 113 |
| `credit-g_40nan` | 0.9734 | 0.9598 | 0.9662 | 0.9579 | 0.9645 | −0.0136 [−0.0207, −0.0066] | C better | −0.0072 [−0.0222, +0.0079] | none | 45 → 83 |
| `credit-g_60nan` | 1.0003 | 0.9836 | 0.9932 | 0.9849 | 1.0000 | −0.0167 [−0.0251, −0.0083] | C better | −0.0071 [−0.0273, +0.0130] | none | 58 → 48 |
| `credit-g_80nan` | 1.0305 | 1.0120 | 1.0105 | 1.0048 | 1.0000 | −0.0185 [−0.0298, −0.0072] | C better | −0.0200 [−0.0361, −0.0040] | K better | 25 → 12 |
| `kr-vs-kp_20nan` | 0.5947 | 0.6024 | 0.5866 | 0.5941 | 0.6118 | +0.0077 [−0.0018, +0.0172] | none | −0.0081 [−0.0231, +0.0068] | none | 253 → 266 |
| `kr-vs-kp_40nan` | 0.7600 | 0.7509 | 0.7626 | 0.7521 | 0.7721 | −0.0090 [−0.0148, −0.0032] | C better | +0.0027 [−0.0097, +0.0151] | none | 110 → 132 |
| `kr-vs-kp_60nan` | 0.8716 | 0.8590 | 0.8711 | 0.8609 | 0.9090 | −0.0126 [−0.0190, −0.0062] | C better | −0.0005 [−0.0092, +0.0081] | none | 71 → 92 |
| `kr-vs-kp_80nan` | 1.0104 | 0.9771 | 0.9815 | 0.9733 | 1.0000 | −0.0333 [−0.0437, −0.0229] | C better | −0.0289 [−0.0453, −0.0125] | K better | 32 → 21 |
| `spambase_20nan` | 0.8495 | 0.8548 | 0.8529 | 0.8579 | 0.8597 | +0.0053 [+0.0018, +0.0088] | H better | +0.0034 [−0.0048, +0.0116] | none | 281 → 287 |
| `spambase_40nan` | 0.9064 | 0.9034 | 0.9068 | 0.9024 | 0.9345 | −0.0029 [−0.0046, −0.0012] | C better | +0.0004 [−0.0053, +0.0061] | none | 209 → 235 |
| `spambase_60nan` | 0.9438 | 0.9410 | 0.9404 | 0.9380 | 0.9814 | −0.0028 [−0.0053, −0.0004] | C better | −0.0033 [−0.0077, +0.0010] | none | 131 → 133 |
| `spambase_80nan` | 0.9946 | 0.9818 | 0.9846 | 0.9786 | 1.0000 | −0.0128 [−0.0178, −0.0078] | C better | −0.0100 [−0.0159, −0.0040] | K better | 92 → 61 |
| `vehicle_20nan` | 0.4148 | 0.4147 | 0.4113 | 0.4110 | 0.4643 | −0.0000 [−0.0008, +0.0007] | none | −0.0035 [−0.0096, +0.0027] | none | 283 → 316 |
| `vehicle_60nan` | 0.6586 | 0.6489 | 0.6398 | 0.6315 | 0.7034 | −0.0097 [−0.0126, −0.0068] | C better | −0.0188 [−0.0390, +0.0015] | none | 184 → 231 |
| `biodeg_20nan` | 0.5991 | 0.5980 | 0.5898 | 0.5918 | 0.6508 | −0.0011 [−0.0073, +0.0052] | none | −0.0093 [−0.0291, +0.0106] | none | 237 → 279 |
| `biodeg_60nan` | 0.7795 | 0.7758 | 0.7644 | 0.7610 | 0.8622 | −0.0038 [−0.0096, +0.0020] | none | −0.0151 [−0.0368, +0.0066] | none | 206 → 263 |
| `kc2_20nan` | 0.5402 | 0.5376 | 0.5363 | 0.5350 | 0.5216 | −0.0025 [−0.0124, +0.0074] | none | −0.0038 [−0.0204, +0.0128] | none | 251 → 269 |
| `kc2_60nan` | 0.5699 | 0.5639 | 0.5529 | 0.5495 | 0.7242 | −0.0060 [−0.0149, +0.0029] | none | −0.0170 [−0.0385, +0.0045] | none | 179 → 235 |
| `pendigits_20nan` | 0.3335 | 0.3319 | 0.3331 | 0.3315 | 0.3510 | −0.0017 [−0.0020, −0.0013] | C better | −0.0005 [−0.0013, +0.0004] | none | 385 → 393 |
| `letter_20nan` | 0.4595 | 0.4578 | 0.4590 | 0.4575 | 0.4722 | −0.0017 [−0.0020, −0.0014] | C better | −0.0005 [−0.0016, +0.0007] | none | 388 → 394 |
| `electricity_20nan` | 0.6177 | 0.6169 | 0.6183 | 0.6186 | 0.6022 | −0.0008 [−0.0038, +0.0023] | none | +0.0006 [−0.0055, +0.0067] | none | 367 → 329 |

**Tally.**
- **C − H:** C better on 13 variants, H better on 1 (spambase_20nan, +0.0053 [+0.0018, +0.0088],
  all three seeds), no detectable difference on 7.
- **K − H:** K better on 3 (credit-g_80nan −0.020, kr-vs-kp_80nan −0.029, spambase_80nan −0.010),
  H better on none, no detectable difference on 18.
- **KC − H (secondary):** KC better on 9, H better on none. **KC − K:** KC better on 12, K better on
  1 (spambase_20nan, +0.0050). **KC − C:** no detectable difference on all 21.

**Reading, by the rule fixed in advance.**
1. **K qualifies:** no variant shows "H better". It is the recommendation this result carries:
   choosing the decode checkpoint by the validation rows' own gaps.
2. **C does not qualify:** spambase_20nan shows "H better", by 0.005 on every seed.
3. **KC is not recommended over K:** it qualifies against H, but spambase_20nan shows "K better"
   than it, by the same mechanism.
A default switch to K needs its own ADR, flag and tag, and the user's decision.

**Secondary measures.**
- **Means over the 21 variants:** H 0.7519, C 0.7453, K 0.7446, KC 0.7412. Below the best baseline:
  H on 15 variants, C on 18, K on 17, KC on 18.
- **By missing level (mean difference):** K − H −0.003 (20nan), −0.001 (40), −0.010 (60), −0.020
  (80); C − H +0.000, −0.009, −0.009, −0.022; KC − H −0.002, −0.009, −0.016, −0.026. Both
  mechanisms help most where gaps are many.
- **The two criteria choose different epochs where gaps are many:** at 80nan the gap criterion
  stops earlier (credit-g 25 → 12, kr-vs-kp 32 → 21, spambase 92 → 61), the loss on the few
  masked cells having kept training into overfitting; at 20nan it usually picks a later epoch.
- **The calibration's reach:** mean α from 0.51 (spambase_80nan) to 0.98 (vehicle_20nan), lowest
  where gaps are many; a categorical threshold in 4 to 7 columns per fold on credit-g and 13 to 18
  on kr-vs-kp, none on the numerical tables' few categorical columns.
- **The only loss, spambase_20nan,** is a table where the model is already below the best
  baseline (0.8495 against 0.8597) and α (0.83) shrinks guesses that were not overconfident.
- **K's masked population** is better or equal on every variant but three tiny ones
  (credit-g_20nan +0.004, spambase_20nan +0.001, vehicle_20nan +0.001).
- **Time.** Planned about 5.5 to 6 h of wall time, ending 18:30 to 19:00 GMT-3; it took 5 h 51 min
  and ended at 18:43, within the plan (about 1.02x its midpoint). The per-epoch scoring cost no
  measurable time: electricity's cells took 101 min against 100 for E30's arm M.

**What follows.** The user decides whether K becomes the imputation default (an ADR with a flag
and a tag, as ADR 0003's scheduler was). C and KC failed the rule on one variant each, by a small
margin, while gaining elsewhere; a calibration that leaves well-calibrated columns alone (for
example a larger prior, or shrinking only where the fitted α is clearly below 1) would be a new,
pre-registered study, not a reading of this one.
