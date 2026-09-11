# 02. The search objective on the validation split

Status: ready-for-agent
Blocked by: 01
Plan task: 2. ADR 0005 decision 3. Wayfinder ticket 02.

## Goal

An imputation trial is ranked by `validation/impute/masked/impute_score`, computed on the
fixed validation mask the decode stage already holds, only when a programmatic request
field asks for it; classification's study keeps ranking by `f1_macro` on the test split.
Ordinary and cross-validation runs never carry a `validation/` key.

## Seams under test

- `TaskSpec.search_objective`: `validation/impute/masked/impute_score` for imputation,
  `f1_macro` for classification.
- `train_and_evaluate_decoder(..., score_search_objective=True)` on the tiny synthetic
  frame of `tests/unit/test_training_decoding.py`: the fold metrics and the tracker both
  hold `validation/impute/masked/impute_score`; with the default, no key starts with
  `validation/`.
- `run_hyperparameter_optimization` (temporary backend, stub returning distinct
  validation and test scores): `optuna/best_objective_value` and the best trial follow
  the validation score; every imputation trial namespace carries the request field, the
  retrain namespace does not.

## Acceptance criteria

- [ ] All of the above, in `tests/unit/test_training_tasks.py`,
      `tests/unit/test_training_decoding.py` and `tests/unit/test_opt_tracking.py`.
- [ ] Unit suite green; both integration fixtures pass unedited (the field defaults off).
- [ ] A real two-trial reduced study on `credit-g_20nan` against a scratch store:
      each trial's `optuna/objective_value` equals its
      `validation/impute/masked/impute_score` and differs from its
      `test/impute/masked/impute_score`.

## Constraints

- The objective mirrors the ranking metric's population: masked cells only, on the
  validation split; the induced population stays a test-fold report.
- The existing `test/`-prefixing `tracker.log_metrics` call in `decoding.py` stays as it
  is; the `validation/` family is logged in its own call.
- `optuna/objective_value` keeps logging the objective under one key whatever the task.

## Comments

- 2026-09-11 (from ticket 01): `_TrainingStub` in `tests/unit/test_opt_tracking.py` keys the
  metric it returns on `args.task` and returns `impute/masked/impute_score` for imputation.
  Moving the objective to a `validation/` key changes what the stub must return and what
  `test_an_imputation_study_trains_the_imputation_task_in_every_trial` relies on; plan that
  red rather than meeting it by surprise. The fixture also fabricates a vehicle table
  header because a study reads the header before its first trial.
