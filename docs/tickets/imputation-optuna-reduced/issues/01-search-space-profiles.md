# 01. Search-space profiles

Status: ready-for-agent
Blocked by: none
Plan task: 1. ADR 0005 decisions 1 and 2. Wayfinder tickets 01 and 05.

## Goal

A `--search_space {full,reduced}` flag, defaulting per task, with the reduced profile
sampling only `PROB_MASCARA`, `LR_DECODE` (1e-4..1e-2), `WEIGHT_DECAY_DECODE`, `DROPOUT`
and, on mixed tables, `LAMBDA_NUM`; the two range corrections applied to `full` as well;
the profile recorded as a sparse `search_space` tag and param on the study parent and
every trial; the column mix read from the pipeline's declaration through a pure helper
extracted from `prepare_dataset`; and `opt.py`'s `__main__` built on
`build_training_parser` so the two entry points share every flag.

## Seams under test

- `define_search_space(trial, task, profile, mixed_columns)`: the reduced imputation
  profile returns exactly the sampled keys (five on a mixed table, four otherwise) and
  nothing held; the full profile reproduces today's keys with `LR_DECODE` reaching 1e-2
  and `LAMBDA_NUM` only on mixed tables; defaults reproduce today's full space.
- `validate_parsed_args`: `--search_space reduced` with `--task classification` is a
  `SystemExit` naming both flags; the flag defaults to `None`.
- `declared_column_types(columns, label_column, base_dataset_name)`: electricity is mixed
  (its `day` is declared), vehicle is numerical-only, credit-g is 13 + 7; `prepare_dataset`
  unchanged, which the vehicle fixture proves.
- `run_hyperparameter_optimization` (temporary MLflow backend): the study parent and every
  trial carry `search_space` as tag and param; `None` resolves to `reduced` for
  imputation and `full` for classification.
- `opt.build_parser()` accepts `--dataset_name`, `--task`, `--search_space`,
  `--lr_scheduler`, `--disable_mlflow`, `--metrics_dir`, `--n_trials`, `--retrain_best`.

## Acceptance criteria

- [ ] All of the above, in `tests/unit/test_opt_search_space.py`, `tests/unit/test_opt_tracking.py`,
      `tests/unit/test_training_config.py` and `tests/unit/test_training_data.py`.
- [ ] Unit suite green; both integration fixtures pass unedited.
- [ ] A real two-trial reduced study on `credit-g_20nan` against a scratch store shows only
      the sampled keys as `best_` params and `search_space = reduced` on the study parent.

## Constraints

- The column mix is computed once per study from the dataset's CSV header and the
  declared categorical list, never per trial and never from dtypes.
- `SEARCH_SPACE_PROFILES` lives in `src/training/types.py` in the `LR_SCHEDULER_NAMES`
  style; `SEARCH_SPACE_TAG` in `src/mlflow_utils.py`.
- No backfill: the store holds no Optuna run.

## Comments
