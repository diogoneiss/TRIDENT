# TRIDENT Agent Guide

- Use Python 3.10 through uv: `uv run --python 3.10 ...`.
- Main entry points: `main.py` for normal CLI use, `train.py` for the compatible training API/CLI, and `opt.py` for Optuna.
- Training implementation lives in `src/training/`; keep `train.main(args, return_metrics=False)` and its Optuna-compatible return shape stable.
- Preserve training behavior unless the task explicitly authorizes an algorithm change (notably split/scaling order and scheduler cadence).
- Runtime defaults are `results/`, `metrics/`, automatic device selection, and MLflow enabled. Use `--metrics_dir` and `--disable_mlflow` for isolated runs.
- Tests: `uv run --python 3.10 pytest -m "not integration"`; run the regression check with `uv run --python 3.10 pytest -m integration`.
- `vehicle_00nan` integration metrics are a reviewed regression baseline; change its fixture only for intentional, documented behavior changes.
- Keep tickets, plans, and decisions in repository files under `docs/`; do not create external issues.
- `pyarrow<24` is pinned because PyArrow 24 caused intermittent Windows access violations during pytest collection.
- Preserve unrelated worktree changes, especially generated `metrics/`, `results/`, MLflow files, and user edits.
