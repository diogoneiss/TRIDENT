# A deployable decode checkpoint (E43; pre-registered 2026-10-16)

This ticket was written and committed before the first cell, on 2026-10-16 at 22:58 GMT-3 (`9a7e249`). The queues started at 22:58:11
GMT-3.

- **Plan:** [2026-10-17-experiment-improvements](../../plans/2026-10-17-experiment-improvements.md)
  (§2c, §3; D9, D10, D19), accepted by the user on 2026-10-16. Path B runs this study before the
  confirmation (E44), so that E44 carries the M readout.
- **Code:** `DECODE_CHECKPOINT masked` and `--score_checkpoints`, authorized by D9 and landed in
  the same commit as this ticket.

## Why

Threat 3 of the 2026-10-16 review: today's default checkpoint (K, ADR 0015) uses the true values of
the validation rows' own gaps.

- **The oracle problem.** Only the validation rows are used, so it is not test leakage. But no
  baseline gets an equivalent selection, and natural data cannot support it: a deployed model has
  no truth for its gaps.
- **The 80nan claims rest on it.** At 80nan, E34 and E35 found the model beating the mean/mode
  fill on 0 of 3 variants under the loss checkpoint (H), and on 2 of 3 under K.
- **Part of K's gain may be a better stopping rule.** K also improves the masked test cells, which
  it never chose on.

The question is whether a criterion that needs no gap truth recovers that gain.

**The candidate, M.** It keeps the epoch whose fixed validation mask scores best by
`impute_score`, the λ-free score on the same cells the validation loss is computed on. Those
cells are observed values hidden on purpose, so M works on any table, complete sibling or not.

## The code (landed with this ticket)

- **`Hyperparameters.decode_checkpoint` gains `masked`** (`--decode_checkpoint masked`).
  - `_ValidationMaskScore` in `src/training/decoding.py` scores the fixed validation mask every
    epoch, in eval mode. A unit test holds it equal to the end-of-run pandas path.
  - The imputation default stays `induced` (K). Every earlier run's `decode_checkpoint` tag stays
    true, so nothing is backfilled; a run that asks for `masked` is tagged `masked`.
- **`--score_checkpoints`, a diagnostic.** The test split is also scored at the epoch every other
  criterion would keep, as `impute/<population>/{loss,induced,masked}_checkpoint/*`, without
  baselines.
  - The epochs are logged as `decode/{loss,induced,masked}_checkpoint_epoch`.
  - The watchers are eval-mode forward passes, so the run draws nothing more. Each readout equals
    what a run under that criterion keeps.
  - **Unit tests show:**
    - the headline, losses and weights are unchanged with the flag;
    - a `masked` run's headline equals a K run's `masked_checkpoint` family;
    - a K run's `loss_checkpoint` family equals a `loss` run's headline;
    - `masked` keeps the same epoch on a table with no complete sibling.
  - **Suite:** the unit suite passes (358 tests, mypy included).
- **No calibration at M** (D10). KC stays an oracle readout at K.

## Design (fixed before launch)

- **Cells:** one run per (variant, seed) of today's default as a default run reads it, plus
  `--score_checkpoints --score_calibrated`.
  - **Runner:** `scripts/experiments/checkpoint_run.py`; tag `experiment=deployable-checkpoint-2026-10-17`.
  - **Arms, within each run, on one trajectory:**
    - K, the headline and an oracle;
    - H, `loss_checkpoint`;
    - M, `masked_checkpoint`;
    - KC, `calibrated`, an oracle.
- **Variants:** the 9 at ≥60nan, where K's verdicts fell in E34 and E35:
  - kr-vs-kp, spambase and credit-g at 60nan and 80nan;
  - vehicle_60nan, biodeg_60nan and kc2_60nan.

  None reads a promoted file.
- **Seeds:** 111, 222 and 333. The store holds no imputation run on any of them.
- **Size:** 27 cells.
- **Running it:** four queues (`scripts/experiments/run_checkpoint_queue.sh 1..4`), one trainer
  each, with the CPU threads capped at 4.
  - Cells are assigned longest first by E40's cell times: about 111 minutes a queue.
  - No cell starts after 08:00 GMT-3 on 2026-10-17; a cell not started by then is reported missing.
  - Nothing is extended on its result. A crash is fixed and its cell rerun from the start, and the
    rerun is noted here.

### Checks before launch, on credit-g_60nan at seed 404 (an E40 cell), with `--no-mlflow`

1. The runner must equal E40's cell, fold by fold, on every headline model metric, every baseline
   column and every calibrated column. This shows that `--score_checkpoints` leaves the run
   untouched and that the runner is a default run.
2. A `main.py --decode_checkpoint masked` run must have a headline equal to the first check's
   `masked_checkpoint` family. This shows the readout is what the criterion keeps.

**Results: passed** (2026-10-16, 22:54–22:57 GMT-3). Both runs ran on the GPU at once and read
the baseline cache.

1. **Check 1.** Fold by fold, the runner equals E40's cell on all 15 headline model metrics, 60
   baseline columns and 9 calibrated columns. It adds 28 columns, the `loss_checkpoint` and
   `masked_checkpoint` families of both populations.
2. **Check 2.** The `masked` run's headline equals the first run's `masked_checkpoint` family on
   all 14 metrics, and its kept epochs equal the M epochs.

**The check cell's numbers were seen before launch.** They are on seed 404, not a study seed, and
they came after this ticket's reading was written:

| arm | induced `impute_score` |
|---|---|
| K | 0.9927 |
| H | 1.0017 |
| M | 0.9906 |
| KC | 0.9813 |

The reading was not changed after seeing them.

**The warm-up driver** for E44 (`scripts/experiments/warm_baseline_cache.py`) was checked on the
same cell: it found every entry E40's live run wrote (7,218 entries before and after, 0 s).

## Measures

**Primary,** per variant, on the induced `impute_score`, over 15 within-run fold pairs: M − H and
M − K. A verdict needs the 95% t-interval to exclude zero and all three seeds' mean differences to
agree in sign.

**Reading, fixed now,** applied in order, first match wins:

1. **M suffices:** K is better than M on none of the 9. Claims that must transfer report M.
2. **M helps:** M is better than H on at least 2 of the 3 80nan variants and worse than H on none
   of the 9. M becomes the deployable headline, by an ADR amending ADR 0015, and K is reported as
   the oracle bound.
3. **Otherwise:** H is the deployable headline, and "beats the mean/mode fill at 80nan" is
   labelled oracle-only.

The imputation default stays K in every case. Changing it would need its own decision.

**Secondary, descriptive:**

- K − H, a third seed set for ADR 0015;
- KC − K, a fourth set for E40's sign at 60nan and 80nan;
- each of K, H and M against the bar at 80nan. The bar is the best baseline outside
  `summary.OUTSIDE_THE_BAR` in each run's fold CSV, so lgbm is included;
- the epochs each criterion keeps;
- the masked population;
- minutes a cell.

**Analysis:** `scripts/experiments/checkpoint_report.py`; `--tally` prints the reading.

## Known limits, stated in advance

- **These are development variants** (threat 2). The confirmation (E44) carries the same readouts
  on unused variants.
- **The validation mask's draw is MCAR,** so M selects on cells that look like the masked
  population, not on gaps. Under MNAR that is the point: the mask's truths are observed values,
  never the hidden ones.
- **Within-run pairs give narrow intervals,** so a small effect can earn a verdict.
- **Multiplicity:** 2 primary contrasts on 9 variants, with the seed-agreement rule.

## Outcome (2026-10-17, 02:00 GMT-3)

**Run.** All 27 cells ran from 22:58 GMT-3 on 2026-10-16 to 01:50 GMT-3 on 2026-10-17 (2 h 52 min
against about 1.9 h planned). The CPU was shared with E41, E42 and the warm-up. None failed and
none is missing.

**Table.** From `scripts/experiments/checkpoint_report.py`. Induced `impute_score`, 15
within-run fold pairs, lower is better; the bar is the best of the bar's baselines in the run's
fold CSV.

| variant | K | H | M | KC | bar | M − H | M − K | K − H | KC − K |
|---|---|---|---|---|---|---|---|---|---|
| kr-vs-kp_60nan | 0.8646 | 0.8701 | 0.8868 | 0.8630 | 0.9049 | +0.0168, H better | +0.0222, K better | none | none |
| kr-vs-kp_80nan | 0.9806 | 0.9992 | 1.0139 | 0.9696 | 1.0000 | none | +0.0332, K better | −0.0185, K better | −0.0110, KC better |
| spambase_60nan | 0.9405 | 0.9447 | 0.9447 | 0.9379 | 0.9829 | 0 (identical) | none | none | −0.0026, KC better |
| spambase_80nan | 0.9852 | 0.9941 | 0.9941 | 0.9791 | 1.0000 | 0 (identical) | none | none | −0.0061, KC better |
| credit-g_60nan | 0.9905 | 1.0109 | 1.0081 | 0.9828 | 1.0000 | none | +0.0175, K better | −0.0204, K better | −0.0077, KC better |
| credit-g_80nan | 1.0121 | 1.0161 | 1.0200 | 1.0072 | 1.0000 | none | none | none | −0.0049, KC better |
| vehicle_60nan | 0.6434 | 0.6489 | 0.6489 | 0.6352 | 0.7020 | 0 (identical) | none | none | −0.0082, KC better |
| biodeg_60nan | 0.7619 | 0.7722 | 0.7722 | 0.7600 | 0.8612 | 0 (identical) | +0.0104, K better | −0.0104, K better | none |
| kc2_60nan | 0.5754 | 0.5835 | 0.5835 | 0.5725 | 0.7181 | 0 (identical) | none | none | none |

**Reading, by the rule fixed in advance: 3, otherwise.**

- K is better than M on 4 of the 9 variants (kr-vs-kp_60nan, kr-vs-kp_80nan, credit-g_60nan,
  biodeg_60nan).
- M is better than H on none of the 80nan variants, and worse on kr-vs-kp_60nan.

So **H is the deployable headline,** and "beats the mean/mode fill at 80nan" is labelled
oracle-only: at 80nan, K is better than the bar on kr-vs-kp and spambase, while H and M are better
on none. The imputation default stays K.

**Why M adds nothing over H on numerical tables.**

- **The mechanism:** on a table with only numerical columns, the decode loss is the mean squared
  error on the fixed validation mask, and M's score is that error's RMSE over a constant. Both
  rank the epochs the same way, and both keep the first best epoch, so M keeps H's epoch exactly
  (spambase, vehicle, biodeg, kc2: identical scores and epochs).
- **Where M can differ:** only where categorical columns enter, since there the loss is
  cross-entropy and M counts errors.
- **Where it differed, it chose later epochs and did worse:** kr-vs-kp_60nan, epoch 133 against
  88.

This explains the result rather than refuting the criterion's design.

**Secondary:**

- **ADR 0015 on a third seed set:** K − H is "K better" on 3 of 9 (kr-vs-kp_80nan,
  credit-g_60nan, biodeg_60nan) and worse on none.
- **The calibration (T20) on a fourth seed set:** KC − K is "KC better" on 6 of 9, exactly E40's
  six named variants, and worse on none.
- **Context for KC:** KC is oracle-fitted too, so it does not help the deployability question
  (D10).

**Consequences:**

- **ADR 0015's Consequences gain a line,** without changing the default: transfer claims use H; K
  is the oracle bound.
- **E44** reports H as its deployable readout, as its ticket says, with M kept as descriptive.
- **A deployable criterion that improves on H** would need a signal other than the observed
  validation cells: a held-out share of the gaps cannot exist without truth, so for instance an
  ensemble or a smoothed stopping rule. It is not planned.
