# 10. Optuna for the imputation task

Status: ready-for-agent
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

- [ ] All of the above, in `tests/unit/test_opt_tracking.py` or a sibling test module.
- [ ] Classification studies keep their current behaviour except for the head-count
      constraint and the pruning failure path.

## Constraints

- Under a minimised objective, returning `0.0` on failure would score crashes as perfect
  imputers; this is why pruning replaces the sentinel in both directions.
- No Optuna run exists in the store, so changing the sampled space breaks no published
  result.

## Comments
