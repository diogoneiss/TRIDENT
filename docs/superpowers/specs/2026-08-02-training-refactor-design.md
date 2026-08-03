# Training Refactor Design

## Purpose

Split `train.py` into focused, testable modules while preserving normal training behavior and existing calling conventions. Establish an isolated end-to-end regression test so structural changes cannot silently alter the small-run metrics contract.

## Architecture

`train.py` becomes a backward-compatible façade: it retains `main(args, return_metrics=False)`, exposes direct script execution, and translates legacy `argparse.Namespace` inputs into typed internal values. The implementation moves to `src/training/`:

- `config.py` defines `Hyperparameters`, `DatasetSpec`, and `RuntimeOptions`, and resolves defaults, JSON configuration, and Optuna overrides.
- `data.py` resolves dataset paths, reads and prepares a dataset, and constructs the legacy or cross-validation folds.
- `pretraining.py` builds and trains the pre-training model.
- `finetuning.py` builds the classifier, trains it, evaluates it, and returns fold metrics.
- `artifacts.py` writes hyperparameters, metrics, plots, and optional models.
- `runner.py` coordinates runs and folds, owns MLflow orchestration, and returns a typed `TrainingResult`.

`main.py` and `train.py` will use one shared argument parser. `opt.py` continues calling the façade and can provide the existing `hyperparams_override` field.

## Runtime outputs and tracking

Normal commands retain their current defaults:

- run artifacts: `results/`
- project metrics: `metrics/`
- MLflow: enabled
- device: automatically selected

Two new public options are added to the shared parser:

- `--metrics_dir`: destination for the project-level metrics CSV, defaulting to `metrics`.
- `--disable_mlflow`: disables all MLflow setup, autologging, metrics, tags, and artifacts.

Tests and programmatic callers can select temporary output and metrics locations with MLflow disabled. This isolates test artifacts without changing the default experiment workflow.

## Compatibility

- Both `uv run main.py ...` and `uv run train.py ...` stay valid.
- `train.main(args, return_metrics=True)` keeps returning the existing dictionary shape expected by Optuna.
- Existing names, paths, split selection, random seed behavior, auto-selected device behavior, output defaults, and MLflow behavior are preserved.
- This refactor intentionally does not address data-scaling order, scheduler cadence, or other training-correctness issues.

## Regression testing

Add `pytest` as a development dependency and configure an `integration` marker. The integration test will run `vehicle_00nan` with:

- seed `42`
- `cv_folds=2`
- automatic device selection
- a tiny explicit override: two pre-training and two fine-tuning epochs
- isolated temporary results and metrics directories
- MLflow disabled

The test reads a version-controlled JSON fixture containing the exact run configuration, each fold's `accuracy`, `f1_micro`, and `f1_macro`, their two-fold means, and metric tolerances. It checks both individual folds and aggregate means. Since the selected device may vary and current seeding does not force deterministic GPU algorithms, tolerances are calibrated from repeated baseline runs and documented in the fixture.

The test is marked `integration`, so it runs deliberately with `uv run pytest -m integration` instead of in the fast default suite. Re-baselining is an explicit reviewable fixture update, not an automatic action.

## Error handling and validation

The existing missing-dataset and missing-split behavior remains intact. Configuration parsing keeps the current default values and accepts only known hyperparameter keys. Runtime options use defaults when legacy callers do not provide their new fields.

## Documentation and tracking

`docs/tickets/0001-training-refactor.md` tracks the work, and ADR 0001 records the package/facade boundary. No tickets, issues, or decisions are created in external services.
