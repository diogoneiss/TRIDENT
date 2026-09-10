# 02. Request, hyperparameters and command line

Status: ready-for-agent
Blocked by: 01
Plan task: 2. ADR 0004 decisions 1, 9, 11. Wayfinder tickets 08, 11, 12.

## Goal

Let a run select the imputation task from the command line, carry the five new
hyperparameters (and the extra-rates list) through the config file with defaults that keep
every existing file valid, and log only the parameters the run's task actually uses.

## Seams under test

1. `Hyperparameters.from_mapping` (`src/training/types.py`): the new keys take defaults;
   a fifteen-key mapping shaped like `datasets/hiperparams/vehicle/vehicle_00nan.json`
   still loads unchanged.
2. `build_training_parser` + `validate_parsed_args` (`src/training/config.py`): `--task`
   defaults to `classification` with `choices=TASK_NAMES`; `--score_null_path` is refused
   without `--task imputation` and accepted with it.
3. `resolve_training_request`: `task` and `score_null_path` land on `TrainingRequest`;
   `TrainingRequest.__post_init__` rejects an unknown task name.
4. **Logged parameters**: promote the runner's private `_mlflow_hyperparameters` to a
   public function (e.g. `logged_hyperparameters(request)` in `config.py`) consumed by the
   runner and, later, the Optuna trial tracker. An imputation request yields the
   pre-training and decode keys and omits `EPOCH_FINE`, `LR_FINE`, `WEIGHT_DECAY_FINE`,
   `LABELS`; a classification request yields exactly today's set; `task` is logged as a
   run parameter by `_log_execution_params`.

## Files

Modify `src/training/types.py`, `src/training/config.py`, `src/training/runner.py`,
`tests/unit/test_training_config.py`, `tests/unit/test_training_runtime.py`.

## Acceptance criteria

- [ ] New `Hyperparameters` fields with JSON keys and defaults: `decode_epochs` /
      `EPOCHS_DECODE` 150, `decode_learning_rate` / `LR_DECODE` 0.001,
      `decode_weight_decay` / `WEIGHT_DECAY_DECODE` 0.0019, `lambda_num` / `LAMBDA_NUM` 1.0,
      `eval_mask_rate` / `EVAL_MASK_RATE` 0.2, `eval_mask_rates_extra` /
      `EVAL_MASK_RATES_EXTRA` `()`.
- [ ] `TrainingRequest` gains `task: str = "classification"` and `score_null_path: bool =
      False` (defaulted fields after the undefaulted ones; order between them free).
- [ ] `--task` and `--score_null_path` parse as above; the rejection is a `SystemExit`
      from `validate_parsed_args`, not a runtime error.
- [ ] Logged parameter set is task-specific as above; `task` appears as a parameter.
- [ ] Existing unit suite and the vehicle regression pass with no edit.

## Constraints

- `--task` is never readable from the JSON file (ADR decision 1).
- No new per-hyperparameter command-line flags (backlog I1 is the general fix).
- `EVAL_MASK_RATE` stays a scalar; the sweep list is the separate key.

## Comments
