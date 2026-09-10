# 08. Runner, MLflow identity, batch summary

Status: done (this commit) on `feat/imputation-task`
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

- [x] Parent, `best_fold` and `worst_fold` carry both tags; a classification run gains
      `task=classification`, `is_optuna=false` and nothing else changes.
- [x] Imputation run name prefix `impute_`.
- [x] `_execution_tags`, `_log_diagnostic_children` and the study parent in `opt.py` all
      set both tags (the third is finished in ticket 10, but the constants land here).
- [x] Tag constants in `src/mlflow_utils.py`.
- [x] Unit suite and vehicle regression green, regression unedited.

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

- 2026-09-10, five red-green slices across `test_training_runtime.py` and
  `test_training_tracking.py`:

  | Slice | Behaviour pinned |
  |---|---|
  | 1 | The runner runs the stage its task asks for, and only that one |
  | 2 | Every run says which task it trained and whether a search made it |
  | 3 | An imputation run calls its best fold the one with the lowest score |
  | 4 | An imputation run leaves a preview and a ledger for every fold, and a classification run leaves none |
  | 5 | A loss plot is named after the stage that produced it |

  Slice 1 is the classification guard deferred here from ticket 06: it is only meaningful
  once the dispatch exists.

  **Slice 5 came from the end-to-end run.** An imputation run was writing its decode
  curves to `finetune_losses.png`, a file claiming to be a stage it never ran. Nothing
  failed; the plot was simply mislabelled, and only looking at the output caught it.

  Three consequences of adding a run parameter, each a real behaviour change rather than
  a broken test: two existing assertions pin the exact parameter set and gained `task`,
  and the runner's parent-run kwargs assertion gained it too.

  `write_hyperparameters` now takes the task, so an imputation run's
  `hyperparameters.json` no longer claims `EPOCH_FINE` and `LABELS` (the finding passed
  here from ticket 07).

  **Full run through the real command line**, `credit-g_20nan`, two folds, tracking to a
  throwaway store:

  ```
  main.py --dataset_name credit-g_20nan --task imputation --cv_folds 2           --plot_losses --lr_scheduler cosine
  ```

  | | value |
  |---|---|
  | `cv/test/impute/masked/impute_score/mean` | 0.9772 |
  | `cv/test/impute/induced/impute_score/mean` | 0.9800 |
  | realised rate at nominal 0.2 | 0.166 |
  | runs recorded | parent + `best_fold` + `worst_fold`, all tagged `task=imputation`, `is_optuna=false` |
  | parent run name | `impute_credit-g_20nan_20260910_192737` |
  | artifacts | two previews, two ledgers, one per-column table, `pretrain_losses` and `decode_losses` plots |
  | `hyperparameters.json` | decode keys only; no `EPOCH_FINE`, no `LABELS` |

  Unit suite 119 green; the vehicle regression passes unedited.
