# 03. Task-keyed lookup and explicit promotion

Status: done (`a24dcec`) on `feat/imputation-task`, 2026-09-11
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

- [x] All of the above, in `tests/unit/test_training_config.py`,
      `tests/unit/test_training_tracking.py` and `tests/unit/test_opt_tracking.py`.
- [x] Unit suite green; both integration fixtures pass unedited.
- [x] A real two-trial reduced study on `credit-g_20nan --promote_best` (scratch store)
      produces `datasets/hiperparams/credit-g/credit-g_20nan.imputation.json` with the
      complete key set; the file is then deleted, since the real promoted files come from
      ticket 05.
- [x] `README.md` "Configuration System" documents the lookup order and `--promote_best`;
      `docs/BACKLOG.md` moves I2 to "Fixed already" with the commit.

## Constraints

- `--promote_best` and `--retrain_best` are independent.
- The promoted file is written from the winning trial's `hyperparameters` user attribute
  resolved through `Hyperparameters.from_mapping`, in the key set `logged_hyperparameters`
  uses for the task.
- A general `--hyperparams <path>` flag (backlog I1) is out of scope.

## Comments

- 2026-09-11, seven red-green slices, commit `a24dcec`:

  | Slice | Behaviour pinned |
  |---|---|
  | 1 | An imputation run prefers `<dataset>.imputation.json`, falls back to the shared file, then the defaults; classification never reads the task-keyed file |
  | 2 | Every request says where its configuration came from: `defaults`, `override`, or the posix repository-relative path |
  | 3 | Both trackers log `config_source`, on comparison parents and Optuna trials alike (dense) |
  | 4 | A classification study without the flag writes nothing outside its study directory; running best and final best live there (backlog I2) |
  | 5 | `--promote_best` on an imputation study writes the complete 17-key task-keyed file, held values at their defaults, `LR_SCHEDULER` the study's schedule |
  | 6 | `--promote_best` on a classification study writes the complete 16-key shared file (`LABELS` rides along at 4, as `logged_hyperparameters` does; backlog C3) |
  | 7 | `--promote_best` is accepted at both entry points |

  Slice 3 made the trial record's exact-params test and the runner's exact-kwargs test
  in `test_training_runtime.py` go red as planned, so the runner's pass-through has a
  unit seam after all; both gained the key. `logged_hyperparameters` now delegates to
  `complete_configuration(hyperparameters, task)`, the key set a promoted file holds;
  `hyperparameter_file(dataset, task)` names the per-task path for the loader and the
  promoter alike. Unit suite 158 green; both fixtures pass unedited.

  **Proof** (`credit-g_20nan --task imputation --use_optuna --n_trials 2 --lr_scheduler
  cosine --promote_best`, scratch store `scratchpad/proof03/proof.db`): the two trials
  logged `config_source = override`; the promoted file held the 17 keys with
  `LR_SCHEDULER` `cosine`, `LR_DECODE` 0.00797, `WEIGHT_DECAY_DECODE` 0.00157,
  `LAMBDA_NUM` 0.205 (credit-g is mixed), `PROB_MASCARA` and `DROPOUT` 0.3 (written as
  the float-step artefact `0.30000000000000004`, exactly what the trial trained with),
  and every held key at its default. An ordinary `--task imputation --lr_scheduler
  cosine` run then logged `config_source =
  datasets/hiperparams/credit-g/credit-g_20nan.imputation.json`, the promoted
  `LR_DECODE`, no `validation/` key, and `test/impute/masked/impute_score` 0.8482,
  identical to trial 0's test score. The file and its directory were removed afterwards;
  no artifact folder left under `mlruns/`.

  Notes: `--promote_best` without `--use_optuna` is a silent no-op through `main.py`,
  the same as `--retrain_best`; the study parent logs no `config_source`, since it is
  not a training run; `--search_space` and the validation objective stay out of the
  README until ticket 06, per the plan. The user's own sweep was writing shared files
  under `datasets/hiperparams/` (electricity, letters) while this ticket ran; that
  directory holds no tracked file, and nothing there was touched.
