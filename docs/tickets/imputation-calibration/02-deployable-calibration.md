# A deployable calibration: fitted on the validation mask, read with H (E47; pre-registered 2026-10-18)

This ticket was written and committed before the first cell.

- **Why now:** ADR 0027 made the calibrated output (KC) the imputation default at 40% missing and
  above. KC inherits K's dependence on the validation gaps' truth: both are fitted on cells a
  deployed model cannot see.
  - E43 named H, the loss's checkpoint, the deployable headline.
  - The deployable analogue of KC is H after a calibration fitted on the fixed validation mask's
    cells. Those cells are observed, so a deployed model can fit it.
  - It is untested (T20's follow-up in `docs/TASKS.md`). The user agreed to test it on 2026-10-18.
- **Code:** a new diagnostic flag, `--score_masked_calibration`, off by default. It logs
  `impute/induced/masked_calibrated/*` and `impute/masked/masked_calibrated/*` and touches nothing
  else; a unit test holds every other key, the model and the losses equal to a run without it.
  No default changes.

## Design (fixed before launch)

### The readouts, all on one trajectory per run

Runner: `scripts/experiments/deployable_calibration_run.py`; tag
`experiment=deployable-calibration-2026-10-18`.

| readout | what it is | where it is logged |
|---|---|---|
| H | the loss's checkpoint, uncalibrated (the headline) | `impute/<pop>/*` |
| HC_m | H after a calibration fitted on the validation mask's cells | `impute/<pop>/masked_calibrated/*` |
| HC_g | H after the calibration fitted on the validation gaps (an oracle) | `impute/induced/calibrated/*` |
| K | the validation gaps' checkpoint | `impute/<pop>/induced_checkpoint/*` |
| KC | K after the gaps' calibration (ADR 0027's output) | `impute/induced/induced_checkpoint_calibrated/*` |

The runs pin `--decode_checkpoint loss` and `--decode_calibration none`; neither changes training.

- **The calibration** is ADR 0027's, unchanged (`src/training/calibration.py`):
  - each numerical column shrunk toward the training mean by a fitted α, with a prior of 30 cells;
  - the categorical confidence thresholds, on columns with at least 30 fit cells.
- **Only the cells it is fitted on differ.** The validation mask hides about 20% of a row's
  observed cells, so it holds far fewer cells than the validation gaps at heavy missingness (about
  4% of an 80nan row). The prior then keeps α nearer 1, and few categorical columns reach 30 cells.
  This is stated in advance as the likely limit.

### Cells

- **Variants:** E40's 12 at 40% missing and above.
  - kr-vs-kp, spambase and credit-g at 40, 60 and 80nan;
  - vehicle, biodeg and kc2 at 60nan.
- **Seeds:** E40's, 404, 505 and 606: 36 cells.
  - **Why reuse them:** K and KC on these cells must equal E40's fold by fold. That is the
    integrity check (`--integrity`).
  - HC_m and H were never read on these seeds; E40 read only KC − K.
- **Running it:** four queues (`scripts/experiments/run_deployable_calibration_queue.sh`), CPU threads
  capped at 4.
  - The queues balance about 145 minutes each.
  - No cell starts after 08:00 GMT-3 on 2026-10-19; unstarted cells are reported missing.

## Measures

All readings are on `impute_score` over 15 fold pairs, with the verdict rule and the 1% SESOI
(ADR 0020), through `scripts/experiments/deployable_calibration_report.py --tally`.

**Primary:** HC_m − H on the induced population, per variant.

**Secondary (descriptive):**

- HC_g − H, and the share of HC_g's gain that HC_m keeps;
- HC_m − H on the masked population, where the fit is in-distribution;
- KC − K, which replicates E40 by construction;
- HC_m − K, the deployable candidate against today's oracle checkpoint;
- H and HC_m against the weak bar and the strong tier;
- the mean α of each fit, and the validation mask's cells per fold.

## Reading, fixed now

**Primary, on E40's six named variants** (kr-vs-kp_80nan, spambase_60nan, spambase_80nan,
credit-g_60nan, credit-g_80nan, vehicle_60nan):

- **supported** if HC_m is better than H on at least 4 of the 6 named and worse on none of the 12;
- **refuted** if HC_m is worse on at least 2 of the 12, or better on at most 1 of the named;
- **mixed** otherwise.

**What each reading leads to (a proposal, never a default change without the user):**

- **supported:** propose a deployable calibration mode (for example `DECODE_CALIBRATION masked`)
  by an ADR amending 0027, for runs whose claims must transfer;
- **refuted or mixed:** ADR 0027 stays as it is. KC is reported as an oracle readout beside H, as K
  already is.

**If the integrity check fails** (a difference above 1e-6 on K or KC), nothing is read until it is
explained.

## Launch

To be filled in at launch: the pre-launch check and the launch time.
