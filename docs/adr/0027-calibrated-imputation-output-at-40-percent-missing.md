# Calibrate the imputation output on the validation gaps at 40% missing and above

## Status

Accepted (2026-10-17, about 22:45 GMT-3). The user decided this after E44's confirmation, choosing
"ADR: padrão a partir de 40nan" among the options for reviving T20 (the
[proposal](../proposals/calibration-as-imputation-output.md), parked on 2026-10-16). Like ADRs
0013 to 0015, it is a deliberate exception to the rule that old behaviour stays the default. It
amends ADR [0015](0015-decode-checkpoint-chosen-on-the-validation-gaps.md) (what the kept
checkpoint outputs) and ADR [0013](0013-imputation-defaults-from-the-confirmatory-study.md)
(imputation's task defaults), for the imputation task only.

**Implemented on 2026-10-17** on the branch `feat/adr-0027-calibration`. **Landed on 2026-10-17 at
about 23:55 GMT-3** (fast-forward to `17121a5`), at the user's choice, while E42's missForest
lanes ran. Those lanes import only `types.py` and `mlflow_utils.py` from the touched files, and
both changes are additive. The merged tree passes 403 unit tests and both integration fixtures in
the main checkout. Every imputation run from `17121a5` on applies this ADR.

The numbers 0022 to 0026 are reserved by the
[plan of 2026-10-17](../plans/2026-10-17-experiment-improvements.md) for its proposed ADRs.

## Context

The decode stage keeps the checkpoint chosen on the validation rows' own gaps (K, ADR 0015). The
calibration (`src/training/calibration.py`) is fitted on those same gaps at that checkpoint:

- each numerical column's guess is shrunk toward the training mean by a fitted α, with a prior of
  30 cells;
- a categorical guess whose confidence falls below a fitted threshold is answered with the mode.

Until now it was only a diagnostic readout (`--score_calibrated`, `impute/induced/calibrated/*`).
KC is K after the calibration.

**The evidence:**

| study | seeds | variants | KC against K |
|---|---|---|---|
| E34 | 101/202/303 | the 21 | never worse at 40nan and above; worse on spambase_20nan |
| E35 | 42/7/13 | the 21 | the same; worse on spambase_20nan and kr-vs-kp_20nan |
| E40 | 404/505/606 | 12 at 40nan and above | better on 7 of 12, including all 6 named in advance; worse on none |
| E43 | 111/222/333 | 9 at 60nan and above | better on E40's six again |
| E44 | 707/808/909 | the 15 unused | better on 8 of 15, equivalent on 4, worse on none; better on pendigits_40nan and letter_40nan |

- **The gain is small and grows with missingness.** In E40 the mean of KC − K is −0.0009 at
  40nan, −0.0037 at 60nan and −0.0059 at 80nan. Where it earns a verdict, it is between −0.2% and
  −1.3%.
- **At 20nan the calibration fails.** That is why the rule leaves 20nan alone.
- **E44 answered the proposal's open question** (40 rather than 60): KC is better at 40nan there,
  by the secondary rule fixed before the confirmation was read.

## Decision

1. **The calibration is a named choice.** `Hyperparameters.decode_calibration`
   (`DECODE_CALIBRATION`, `--decode_calibration`) takes one of three values:
   - `none`: every run before this ADR, and the dataclass default;
   - `gaps`: the induced output is always calibrated;
   - `auto`: calibrate where the table misses at least `CALIBRATION_MISSING_RATE` = 0.3 of its
     feature cells.
2. **Imputation defaults to `auto`.** `task_defaults("imputation")` gains `DECODE_CALIBRATION auto`,
   beneath every source and flag. Classification has no decode stage.
3. **The runner resolves `auto` once per run, and records what it applied.**
   - It measures the share of the variant's feature cells that are missing, on the whole table.
     The label never counts.
   - It sets the value to `gaps` or `none` before anything is recorded. So the parameters, the
     hyperparameter file and the `decode_calibration` tag name what the run applied.
   - The tag `measured_missing_rate` keeps the measured share.
   - The cut at 0.3 separates the benchmark's 20nan variants (about 0.2) from its 40nan and
     heavier ones (about 0.4 and up). Being measured, it also works on data that is not named
     `_XXnan`.
4. **Without a complete table the run keeps `none`, and says so.** The calibration needs the
   validation gaps' truth, as the induced checkpoint does (ADR 0015, decision 3).
5. **Under `gaps`, the induced headline is the calibrated output.**
   - `impute/induced/*` scores the calibrated cells. The fold ledger's induced rows hold the
     calibrated guesses, and so do the per-column errors.
   - The uncalibrated scores are kept under `impute/induced/uncalibrated/*`. The calibration's
     summary is under `impute/induced/calibration/mean_alpha` and `n_thresholded_columns`.
   - The `--score_calibrated` diagnostic would repeat the headline, so it is scored only for an
     uncalibrated run.
6. **These stay uncalibrated:**
   - checkpoint selection;
   - the masked population, which was never checked;
   - the other checkpoints' readouts (`--score_checkpoints`);
   - the search objective (ADR 0005, ADR 0008), because the calibration is fitted on the
     validation gaps.
   - The baselines are scored on the same cells as before; nothing of theirs changes.
7. **Reading runs:** `scripts/experiments/report_common.readout_key` reads K, H and M as
   uncalibrated on every run. KC is the headline of a calibrated run, and the `calibrated`
   diagnostic of an older one. A test pins it.
8. **The `decode_calibration` tag** is on every imputation parent and trial.
   - Earlier runs are stamped `none` by `scripts/backfill_decode_calibration_tag.py`, marked
     `decode_calibration_backfilled=true`, with their mirrors stamped in place.
   - **Applied on the user's go, 2026-10-18, 02:43 to 02:55 GMT-3,** right after ADR 0021's
     backfill and behind the same backup.
     - **Stamped:** 2,680 runs in 1,276 trees. A second dry run finds 0.
     - **Mirrors:** 2,670 carry the tag. The 10 without are 9 `FAILED` sources (not mirrored,
       ADR 0006) and one trial under a deleted study, as in ADR 0015.
9. **`--decode_calibration none` recovers the old behaviour.** The README's command for the
   imputation defaults before ADRs 0013 to 0015 gains it.

## Verification

- **The default equals E40's KC on a mixed table:** credit-g_60nan at seed 404, from the
  worktree, without MLflow. Both α and the threshold paths run there. On all five folds:
  - the headline `impute/induced/impute_score` and `rmse_num_z` equal E40's
    `impute/induced/calibrated/*`;
  - `impute/induced/uncalibrated/*` equals E40's headline;
  - `calibration/mean_alpha` (0.6798) and `n_thresholded_columns` equal E40's;
  - the masked headline and the baselines are unchanged.
  - The largest difference is 0 everywhere. The run logged "Measured missing rate 0.6000:
    decode calibration 'auto' resolves to 'gaps'."
- **A 20nan default run is unchanged:** kc2_20nan at seed 42 equals E46's G cell fold by fold
  (largest difference 0; induced 0.5744053, masked 0.5271286), with no calibration key. Its
  measured rate is 0.1992.
- **Regression fixtures:** both pass unmodified. credit-g_20nan reads its mapping without a task,
  so it stays on `none`; vehicle_00nan is a classification run.
- **The unit suite** (403 tests, mypy included) covers:
  - the flag, the guard and the default;
  - the runner's resolution by rate and its fallback without a complete table;
  - the decode stage: under `gaps` the headline equals the older diagnostic on the same
    trajectory, and the uncalibrated family equals the old headline. Training, the masked
    population, the baselines and the search objective are untouched.
  - `readout_key`;
  - the tag backfill.

## Consequences

- **KC inherits K's dependence on the validation gaps' truth.**
  - E43 named H, the loss's checkpoint, the deployable headline: it needs no complete table.
  - This ADR does not touch H, and claims that must transfer are still read on H.
  - The deployable analogue, a calibration fitted on the masked validation cells and read with H,
    is untested. It is a follow-up (T20's note in `docs/TASKS.md`), not built here.
- **Numbers recorded before this ADR keep their meaning.**
  - The thesis's confirmation statement and the development and confirmation tables are on K,
    from runs before this ADR.
  - From this ADR on, a default imputation run at 40% missing or more reports KC as its headline,
    and its K under `uncalibrated`.
  - Comparisons across the boundary read through `report_common`, which keeps K as K.
- **Reproducing an earlier cell** at 40% missing or more through the task defaults needs
  `--decode_calibration none`. That includes the study runners of E43 to E46, which read the
  task defaults, and any check cell on such a variant.
- **The gaps to the baselines** (`gap_to_best_baseline`, `gap_to_best_strong_baseline`) are taken
  from the headline, so they move with it on calibrated runs.
- **A promoted configuration** written by `--promote_best` names `DECODE_CALIBRATION`, like every
  other key, so the file cannot move if the default changes (ADR 0005, decision 5).
- **The cost** is one extra forward pass over the validation gaps per fold, on calibrated runs
  only.
