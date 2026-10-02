# Make the learning-rate schedule selectable, keep the legacy one as default, and tag it in MLflow

## Status

Accepted (2026-09-09). Resolves backlog item B1 in `docs/BACKLOG.md`.

**Amended 2026-10-01** by [ADR 0013](0013-imputation-defaults-from-the-confirmatory-study.md): `cosine_legacy` stays the default for classification; the
imputation task defaults to `cosine`, the schedule of every imputation run in the store.

## Context

Both training stages built `CosineAnnealingLR(T_max=<epochs>)` and then called
`scheduler.step()` after every **mini-batch**. `T_max` counts scheduler steps, so
the cosine period was `epochs` batches, not `epochs` epochs. With the vehicle
configuration (300 epochs, roughly 2 batches per epoch) the learning rate reached
zero half way through training and climbed back to its starting value by the end.
The number of completed cycles equals the batches-per-epoch ratio, so a 45k-row
dataset such as electricity ran about 176 cycles per stage while vehicle ran 2.
Cross-dataset comparisons were therefore confounded by an accidental, size-dependent
sawtooth.

This behaviour dates from the first commit, has never changed, and produced every
MLflow run stored before this decision. `AGENTS.md` protects scheduler cadence
precisely because changing it moves every published number.

## Decision

1. **The schedule is a named hyperparameter.** `Hyperparameters.lr_scheduler`
   (JSON key `LR_SCHEDULER`) selects one of:

   | Name | Torch scheduler | Advances | Shape |
   |---|---|---|---|
   | `cosine_legacy` | `CosineAnnealingLR(T_max=epochs)` | per batch | The published behaviour: one cosine period every `epochs` batches |
   | `cosine` | `CosineAnnealingLR(T_max=epochs)` | per epoch | Single half-cosine from the base rate to zero across training; the obvious intent of the original code |
   | `warmup_cosine` | `LambdaLR` | per batch | Linear warmup over the first 10% of optimizer steps, then a half-cosine to zero over the rest |
   | `constant` | none | never | Fixed base rate; the FT-Transformer style baseline |
   | `plateau` | `ReduceLROnPlateau(factor=0.5, patience=10)` | per epoch, on validation loss | Halves the rate after 10 epochs without a new best validation loss |

   The `--lr_scheduler` flag overrides the JSON value for every dataset in the
   invocation, including `--all` batches and every Optuna trial.

2. **`cosine_legacy` stays the default.** Existing JSON configs, the reviewed
   `vehicle_00nan` regression fixture, and any run launched without the flag
   keep producing bit-identical results to before. Choosing a better schedule is a
   measured decision, not a silent one: the point of this change is to make the
   comparison possible, not to pre-empt it.

3. **The schedule is an MLflow tag, `lr_scheduler`,** on parent runs, diagnostic
   children, Optuna study parents and Optuna trials, in addition to the
   `LR_SCHEDULER` param on parents. Tags are what the MLflow comparison table
   filters and groups on, so the schedule becomes an explicit axis next to
   `run_role`, `dataset_variant` and `missingness_percent`.

4. **Existing runs are backfilled.** `scripts/backfill_lr_scheduler_tag.py`
   stamps `lr_scheduler=cosine_legacy` on every run that lacks the tag and adds
   `lr_scheduler_backfilled=true` so an inferred value is distinguishable from a
   recorded one. It is a dry run unless `--apply` is passed, is idempotent, and
   reports deleted runs without touching them because MLflow refuses tag writes on
   them. It was applied to the 191 runs recorded between 2026-07-03 and 2026-09-09.

5. **Each stage logs its learning rate per epoch** as `pretrain/learning_rate`
   and `finetune/learning_rate`, sampled after the epoch trains and before the
   epoch-level schedules advance, so the schedule that actually ran is auditable
   from the diagnostic children. For `cosine_legacy` this samples the sawtooth at
   epoch boundaries, which can look flat when the period divides the batch count.

## How to compare

Filter `tags.run_role = 'parent'` as before, then add
`tags.lr_scheduler = '<name>'` to compare like with like, or group by that tag to
compare schedules on one dataset. Runs whose tag was inferred also carry
`tags.lr_scheduler_backfilled = 'true'`.

## Considered options

- **Fix the cadence in place** (step once per epoch, no flag). Rejected: it
  changes every stored result without a way to reproduce them and leaves the old
  runs indistinguishable from new ones.
- **Set `T_max = epochs * batches_per_epoch` and keep per-batch stepping.**
  Rejected as the only fix because it still leaves no warmup and no baseline
  without a schedule; it is available as the decay half of `warmup_cosine`.
- **Make the new schedule the default.** Rejected for now: the regression fixture,
  the JSON configs and the August batch run all assume the legacy schedule.
  Revisit once the comparison across the nine datasets has been run.
- **Tune warmup fraction, plateau factor and patience.** Deliberately fixed
  constants in `src/training/schedulers.py` rather than hyperparameters, to keep
  the search space and the JSON schema small until a schedule is chosen.

## Consequences

- `MlflowTracker.parent_run` gains a required `lr_scheduler` keyword.
- Scheduler names are validated at three points: the argparse `choices`, the
  `Hyperparameters` constructor, and `StageScheduler`. An unknown name fails
  before any training starts.
- `AGENTS.md`'s protection of scheduler cadence now means: do not change what a
  named schedule does. Adding a new named schedule is not a behaviour change.
- The `hyperparameters.json` artifact and `LR_SCHEDULER` param make every new
  run self-describing even without the tag.
