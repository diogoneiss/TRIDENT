# Task 3 report: runtime adapters and regression baseline

## Implementation

- Added explicit enabled/disabled MLflow trackers. Disabled tracking does not call
  `mlflow.autolog`, setup, run, logging, or artifact APIs.
- Added typed training orchestration, filesystem artifact writing, and a shared CLI
  compatibility facade. `train.main(args, return_metrics=False)` keeps the flat
  Optuna return mapping.
- Unified `main.py` and direct `train.py` parsing, including `--metrics_dir` and
  `--disable_mlflow`; propagated both options to Optuna trial and retraining namespaces.
- Preserved raw fold metrics under the configured runtime metrics directory and
  retained optional plots/model files plus their MLflow artifacts.
- Added the opt-in typed `vehicle_00nan` two-fold regression baseline.

## RED evidence

1. `uv run --python 3.10 pytest tests/unit/test_training_runtime.py -q`
   initially failed as expected: `src.training.tracking` did not exist and
   `train.run_training` was absent. Importing the old facade also enabled MLflow
   autologging eagerly.
2. `uv run --python 3.10 pytest tests/integration/test_vehicle_regression.py -q`
   initially failed as expected with `FileNotFoundError` for
   `tests/fixtures/vehicle_00nan_regression.json`.

## Regression capture and GREEN evidence

The typed configuration was run three times with seed 42, automatic CUDA selection,
two folds, disabled MLflow, temporary outputs, and the full small override recorded
in the fixture. All three runs matched exactly:

| Metric | Fold 1 | Fold 2 | Mean |
| --- | ---: | ---: | ---: |
| accuracy | 0.2576832151 | 0.2033096927 | 0.2304964539 |
| f1_micro | 0.2576832151 | 0.2033096927 | 0.2304964539 |
| f1_macro | 0.1024436090 | 0.1280307706 | 0.1152371898 |

Environment: Python 3.10; Windows-10-10.0.19045-SP0; PyTorch 2.5.1+cu121; CUDA.
The observed maximum deviation was 0.0, so the committed tolerance is `max(0.0 +
0.005, 0.01) = 0.01`.

- `uv run --python 3.10 pytest tests/integration/test_vehicle_regression.py -q`:
  1 passed.
- `uv run --python 3.10 main.py --help` and `uv run --python 3.10 train.py --help`:
  both expose the shared runtime options.
- `uv run --python 3.10 pytest -q`: 12 passed.

## Environment notes

Pytest exits successfully, but its output includes two existing scikit-learn
undefined-precision warnings for the deliberately tiny run and a Windows pyarrow
access-violation traceback during interpreter shutdown. Neither changes the zero
exit status or test result, but both should be investigated separately if clean test
output is required.

## Commit

`refactor: modularize training workflow`
