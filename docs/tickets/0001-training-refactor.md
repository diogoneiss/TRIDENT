# Training refactor and regression baseline

## Status

In progress

## Goal

Break the monolithic training script into focused modules without changing the default training behavior, and establish a repeatable regression baseline for that behavior.

## Acceptance criteria

- `train.main(args, return_metrics=False)` and direct `train.py` execution remain supported.
- The internal implementation is organized beneath `src/training/` with typed configuration and result objects.
- MLflow and metrics locations are configurable through public CLI options while retaining legacy defaults.
- A committed, opt-in integration test runs a small two-fold `vehicle_00nan` training job against a checked-in metrics fixture.
- All test artifacts are isolated from the repository's normal `results/` and `metrics/` directories.

## Constraints

- Preserve existing training behavior; correctness fixes are out of scope.
- Keep work tracking and decision records in this repository.
