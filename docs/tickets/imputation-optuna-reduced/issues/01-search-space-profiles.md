# 01. Search-space profiles

Status: done (`14ceea6`) on `feat/imputation-task`, 2026-09-11
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

- [x] All of the above, in `tests/unit/test_opt_search_space.py`, `tests/unit/test_opt_tracking.py`,
      `tests/unit/test_training_config.py` and `tests/unit/test_training_data.py`.
- [x] Unit suite green; both integration fixtures pass unedited.
- [x] A real two-trial reduced study on `credit-g_20nan` against a scratch store shows only
      the sampled keys as `best_` params and `search_space = reduced` on the study parent.

## Constraints

- The column mix is computed once per study from the dataset's CSV header and the
  declared categorical list, never per trial and never from dtypes.
- `SEARCH_SPACE_PROFILES` lives in `src/training/types.py` in the `LR_SCHEDULER_NAMES`
  style; `SEARCH_SPACE_TAG` in `src/mlflow_utils.py`.
- No backfill: the store holds no Optuna run.

## Comments

- 2026-09-11, seven red-green slices, commit `14ceea6`:

  | Slice | Behaviour pinned |
  |---|---|
  | 1 | The reduced profile samples only what moves the decoder (five keys on a mixed table, four otherwise) |
  | 2 | The decode learning rate can exceed its default in both profiles |
  | 3 | `LAMBDA_NUM` is sampled only where both column types exist, in the full profile too |
  | 4 | `--search_space reduced` is refused for classification at parse time; unspecified stays unspecified |
  | 5 | Column types follow the pipeline's declaration, not dtypes (`declared_column_types`) |
  | 6 | An imputation study records and samples the reduced profile by default; a classification study the full one |
  | 7 | The study entry point accepts every training flag |

  Slice 6 needed a fabricated table header in the `mlflow_backend` fixture: a study now
  reads the table's header before its first trial, and the fixture's temporary working
  directory had no table, so every stubbed study failed with `FileNotFoundError` until the
  fixture wrote one (vehicle, all numerical, no declaration). Failing fast there is the
  intended behaviour for a real user, so no fallback was added.

  Unit suite 148 green; both fixtures pass unedited after the `data.py` extraction.

  **Proof run** (`credit-g_20nan --task imputation --use_optuna --n_trials 2
  --lr_scheduler cosine`, scratch store `scratchpad/proof01/proof.db`): profile resolved
  to `reduced`, table detected as mixed; trial 0 scored 0.848 masked `impute_score`
  (`PROB_MASCARA` 0.3, `LR_DECODE` 8.0e-3, `WEIGHT_DECAY_DECODE` 1.6e-3, `DROPOUT` 0.3,
  `LAMBDA_NUM` 0.21) and trial 1 scored 0.995; the study parent carries
  `search_space = reduced` as tag and param and exactly the five sampled keys as `best_`
  params; both trials carry the tag and param and ran with `DIM = 128`, `EPOCHS_PRE = 300`
  (held at the defaults); `datasets/hiperparams/` holds no file; no artifact folder was
  left under `mlruns/`. Trial wall-clock about 80 s each, consistent with the 66 s
  training measurement plus start-up. For what it is worth at two trials: 0.848 already
  beats the default configuration's 0.896 on the same split (ticket 07 of the map).

  Note for ticket 02: this proof's objective is still the **test**-split score; the
  validation-split objective is ticket 02's work.
