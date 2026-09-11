# TRIDENT Agent Guide

- Use Python 3.11 through uv: `uv run --python 3.11 ...`.
- Main entry points: `main.py` for normal CLI use, `train.py` for the compatible training API/CLI, and `opt.py` for Optuna.
- Training implementation lives in `src/training/`; keep `train.main(args, return_metrics=False)` and its Optuna-compatible return shape stable.
- Two tasks share the pipeline: `--task classification` (default) and `--task imputation`, which swaps classifier fine-tuning for the decode stage. Every new argument and dataclass field defaults to the classification behaviour, so classification stays bit-identical. See [ADR 0004](docs/adr/0004-imputation-decoder-task.md).
- Preserve training behavior unless the task explicitly authorizes an algorithm change (notably split/scaling order and scheduler cadence).
- Runtime defaults are `results/`, `metrics/`, automatic device selection, and MLflow enabled. Use `--metrics_dir` and `--disable_mlflow` for isolated runs.
- Tests: `uv run --python 3.11 pytest -m "not integration"`; run the regression check with `uv run --python 3.11 pytest -m integration`.
- Two integration fixtures are reviewed regression baselines; change either only for intentional, documented behavior changes: `vehicle_00nan` for the classification task, and `credit-g_20nan` for the imputation task (`tests/fixtures/credit-g_20nan_imputation_regression.json`, the only variant exercising both column types and both scored cell populations).
- Keep tickets, plans, and decisions in repository files under `docs/`; do not create external issues.
- `pyarrow<24` is pinned because PyArrow 24 caused intermittent Windows access violations during pytest collection.
- Preserve unrelated worktree changes, especially generated `metrics/`, `results/`, MLflow files, and user edits.
