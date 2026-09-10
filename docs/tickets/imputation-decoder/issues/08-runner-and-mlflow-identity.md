# 08. Runner, MLflow identity, batch summary

Status: ready-for-agent
Blocked by: 02, 06, 07
Plan task: 8. ADR 0004 decisions 7, 8, 12. Wayfinder tickets 06, 08, 09.

## Goal

Wire the task into the runner, stamp the two new tags on every run kind, name imputation
runs with the `impute_` prefix, and make the batch summary print the task's ranking
metric.

## Seams under test

- `run_training(request)`: dispatches on `task_spec(request.task)`; times the decode
  stage as `time/decode_seconds`; logs the preview and ledger on the fold tracker and the
  per-column table on the parent.
- `MlflowTracker.parent_run(..., task, is_optuna=False)` and the diagnostic children
  (temporary MLflow backend, as `tests/unit/test_training_tracking.py` already does):
  every run kind carries `task` and `is_optuna`; imputation parents are named
  `impute_<dataset>_<timestamp>`; the parent holds
  `cv/test/impute/masked/impute_score/mean` and the per-column artifact.
- `main.run_all`: the summary column is the task's ranking metric with a matching header.

## Acceptance criteria

- [ ] Parent, `best_fold` and `worst_fold` carry both tags; a classification run gains
      `task=classification`, `is_optuna=false` and nothing else changes.
- [ ] Imputation run name prefix `impute_`.
- [ ] `_execution_tags`, `_log_diagnostic_children` and the study parent in `opt.py` all
      set both tags (the third is finished in ticket 10, but the constants land here).
- [ ] Tag constants in `src/mlflow_utils.py`.
- [ ] Unit suite and vehicle regression green, regression unedited.

## Constraints

- A `--retrain_best` parent stays `is_optuna=false`.
- Same `TRIDENT/<base>` experiment; no new experiment.

## Comments

- 2026-09-10 (from ticket 02): also log `task` as a run parameter here, via
  `_log_execution_params`. It was left out of ticket 02 so that every change to the
  tracking seam lands in one place. `logged_hyperparameters(request)` in `config.py` is
  the public replacement for the runner's old private `_mlflow_hyperparameters`.

- 2026-09-10 (from ticket 06): the **classification guard** moved here. A test that
  `run_training` with the default task never constructs a `TridentDecoder` is only
  meaningful once this ticket adds the task dispatch; before that it passes vacuously.

- 2026-09-10 (from ticket 07): `ArtifactWriter.write_hyperparameters` writes a fixed
  classification key set, so an imputation run's `hyperparameters.json` would claim
  `EPOCH_FINE` and `LABELS`, the same trap `logged_hyperparameters` avoids for MLflow
  params. Give it the task (or the request) here.
