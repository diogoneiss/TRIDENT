# 02. The search objective on the validation split

Status: done (`9a38b90`) on `feat/imputation-task`, 2026-09-11
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

- [x] All of the above, in `tests/unit/test_training_tasks.py`,
      `tests/unit/test_training_decoding.py` and `tests/unit/test_opt_tracking.py`
      (plus the request seam in `tests/unit/test_training_config.py`).
- [x] Unit suite green; both integration fixtures pass unedited (the field defaults off).
- [x] A real two-trial reduced study on `credit-g_20nan` against a scratch store:
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
- 2026-09-11, four red-green slices, commit `9a38b90`:

  | Slice | Behaviour pinned |
  |---|---|
  | 1 | Each task declares its search objective (`TaskSpec.search_objective`) |
  | 2 | The decode stage scores the validation split only when asked, in its own `validation/` family, never under `test/` |
  | 3 | An imputation trial is ranked by its validation score, not its test score; the retrain never asks for one |
  | 4 | A search can ask a request for its validation score and nothing else can (programmatic field, defaulted off) |

  The stub now returns both keys for imputation, the validation one equal to the test one
  unless a test says otherwise, so the earlier Optuna tests kept passing. The runner's
  pass-through of the field to the decode stage has no unit seam of its own; the proof
  run covers it. Unit suite 152 green; both fixtures pass unedited.

  **Proof run** (`credit-g_20nan --task imputation --use_optuna --n_trials 2
  --lr_scheduler cosine`, scratch store `scratchpad/proof02/proof.db`): the seeded sampler
  drew the same two configurations as ticket 01's proof. Trial 0: objective 0.8533 =
  validation 0.8533, test 0.8482. Trial 1: objective 0.9940 = validation 0.9940, test
  0.9954. No `test/validation/` key on either trial; `datasets/hiperparams/` untouched; no
  artifact folder left under `mlruns/`. The final log line now names the search objective.

  Note for ticket 03 and the README: a study's `optuna/best_objective_value` is a
  validation-split score from here on and is not comparable with any `test/` number.
