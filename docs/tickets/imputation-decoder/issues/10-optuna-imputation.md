# 10. Optuna for the imputation task

Status: done (this commit) on `feat/imputation-task`
Blocked by: 02, 08
Plan task: 10. ADR 0004 decision 13. Wayfinder ticket 13.

## Goal

Make `opt.py` task-aware: a task-switched search space, a failure path that never
returns a score, the head-count constraint, direction-aware best tracking, and
study-directory-only writes for imputation studies.

## Seams under test

- `define_search_space(trial, task)`: imputation samples `EPOCHS_DECODE`, `LR_DECODE`,
  `WEIGHT_DECAY_DECODE`, `LAMBDA_NUM` and never `EPOCH_FINE` / `LR_FINE` /
  `WEIGHT_DECAY_FINE` nor either evaluation rate; every sampled `(DIM, HEADS)` satisfies
  `DIM % HEADS == 0`.
- `ObjectiveFunctionWrapper.__call__`: a failing trial raises `optuna.TrialPruned` and
  still sets `trial_status=failed`; the running best improves on a *lower* score under the
  imputation task.
- Study creation: `direction` follows the task; the study parent carries `task` and
  `is_optuna=true`.
- `save_best_params` under the imputation task writes only inside the study directory
  and leaves `datasets/hiperparams/` untouched.

## Acceptance criteria

- [x] All of the above, in `tests/unit/test_opt_tracking.py` or a sibling test module.
- [x] Classification studies keep their current behaviour except for the head-count
      constraint and the pruning failure path.

## Constraints

- Under a minimised objective, returning `0.0` on failure would score crashes as perfect
  imputers; this is why pruning replaces the sentinel in both directions.
- No Optuna run exists in the store, so changing the sampled space breaks no published
  result.

## Comments

- 2026-09-10, five red-green slices in `tests/unit/test_opt_search_space.py`:

  | Slice | Behaviour pinned |
  |---|---|
  | 1 | A search tunes the stage its task actually runs |
  | 2 | A search never tunes how hard its own exam is |
  | 3 | Every sampled shape is one the model can actually build |
  | 4 | A trial that crashes is discarded rather than scored |
  | 5 | An imputation study never overwrites a dataset's shared config |

  Slice 3 fixes backlog **B3** by construction rather than by rejection: `HEADS` is drawn
  first and `DIM` as a multiple of it, so an invalid pair can no longer be sampled at all.
  40 random trials all satisfy the constraint.

  **Slice 5 caught the bug while proving it.** Its red run wrote
  `datasets/hiperparams/credit-g/credit-g_20nan.json` -- the cross-task clobber happening
  for real, from a test. The green run keeps the result inside the study directory. The
  stray file was removed.

  **One existing assertion changed for a real reason.** A crashed trial used to return
  `0.0` and finish, so `test_each_trial_trains_inside_its_own_nested_run` asserted every
  run was `FINISHED`. Pruning lets the exception leave the run open, so MLflow marks it
  `FAILED`, which is what actually happened. The test now expects the failed trial to say
  so, and the rest to finish.

  Unit suite 128 green; the vehicle regression passes unedited.
