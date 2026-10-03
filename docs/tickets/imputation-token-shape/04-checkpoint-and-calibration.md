# Does choosing the checkpoint, or calibrating the guesses, on the validation gaps help? (pre-registered 2026-10-03)

This ticket was written and committed before the first run (the commit and the queues' start
time are in the Running notes). The user chose these two ideas on 2026-10-03 after E33, to be
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
