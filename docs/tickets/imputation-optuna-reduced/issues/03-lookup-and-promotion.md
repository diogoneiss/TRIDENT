# 03. Task-keyed lookup and explicit promotion

Status: ready-for-agent
Blocked by: 02
Plan task: 3. ADR 0005 decision 5. Wayfinder ticket 04. Closes backlog I2.

## Goal

`--task imputation` finds `datasets/hiperparams/<base>/<dataset>.imputation.json` before
the shared file; every run logs where its configuration came from; a study writes outside
its own directory only under `--promote_best`, for both tasks, and what it promotes is the
complete configuration.

## Seams under test

- `load_hyperparameters` / `resolve_training_request` in a temporary cwd: imputation
  loads the task-keyed file when present, the shared file otherwise, the defaults
  otherwise; classification loads the shared file or the defaults, never the task-keyed
  one; `TrainingRequest.config_source` is `defaults`, `override` or the relative path.
- The tracker (temporary backend): `params.config_source` on a parent run and on a trial
  run.
- `run_hyperparameter_optimization` (stubbed, temporary backend): a classification study
  without the flag leaves `datasets/hiperparams/` absent and keeps its running best inside
  the study directory; with `--promote_best`, imputation writes the task-keyed file and
  classification the shared file, each holding the task's complete key set (held values
  included) and `LR_SCHEDULER` equal to the study's schedule.

## Acceptance criteria

- [ ] All of the above, in `tests/unit/test_training_config.py`,
      `tests/unit/test_training_tracking.py` and `tests/unit/test_opt_tracking.py`.
- [ ] Unit suite green; both integration fixtures pass unedited.
- [ ] A real two-trial reduced study on `credit-g_20nan --promote_best` (scratch store)
      produces `datasets/hiperparams/credit-g/credit-g_20nan.imputation.json` with the
      complete key set; the file is then deleted, since the real promoted files come from
      ticket 05.
- [ ] `README.md` "Configuration System" documents the lookup order and `--promote_best`;
      `docs/BACKLOG.md` moves I2 to "Fixed already" with the commit.

## Constraints

- `--promote_best` and `--retrain_best` are independent.
- The promoted file is written from the winning trial's `hyperparameters` user attribute
  resolved through `Hyperparameters.from_mapping`, in the key set `logged_hyperparameters`
  uses for the task.
- A general `--hyperparams <path>` flag (backlog I1) is out of scope.

## Comments
