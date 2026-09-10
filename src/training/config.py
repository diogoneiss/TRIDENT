"""Translation from legacy argparse namespaces to typed training requests."""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

from .types import (
    LR_SCHEDULER_NAMES,
    DatasetSpec,
    Hyperparameters,
    RuntimeOptions,
    TrainingRequest,
)


def load_hyperparameters(args: argparse.Namespace) -> Hyperparameters:
    hyperparameters = _load_base_hyperparameters(args)
    # ``--lr_scheduler`` wins over whatever the override mapping or JSON file said,
    # so one invocation can re-run any stored configuration under another schedule.
    lr_scheduler = getattr(args, "lr_scheduler", None)
    if lr_scheduler is not None:
        hyperparameters = dataclasses.replace(hyperparameters, lr_scheduler=lr_scheduler)
    return hyperparameters


def _load_base_hyperparameters(args: argparse.Namespace) -> Hyperparameters:
    override = getattr(args, "hyperparams_override", None)
    if override is not None:
        return Hyperparameters.from_mapping(override)

    dataset_name = getattr(args, "dataset_name")
    base_dataset_name = dataset_name.split("_", 1)[0]
    hyperparameters_path = Path("datasets/hiperparams") / base_dataset_name / f"{dataset_name}.json"
    if hyperparameters_path.exists():
        values = json.loads(hyperparameters_path.read_text())
        if isinstance(values, Mapping):
            return Hyperparameters.from_mapping(values)

    return Hyperparameters()


def resolve_training_request(args: argparse.Namespace) -> TrainingRequest:
    return TrainingRequest(
        dataset=DatasetSpec.from_name(args.dataset_name, getattr(args, "label_column", None)),
        hyperparameters=load_hyperparameters(args),
        runtime=RuntimeOptions(
            output_dir=Path(getattr(args, "output_dir", "results")),
            metrics_dir=Path(getattr(args, "metrics_dir", "metrics")),
            tracking_enabled=not getattr(args, "disable_mlflow", False),
            # Programmatic only (set by opt.py), like ``hyperparams_override``.
            tracking_run_role=getattr(args, "mlflow_run_role", None) or "parent",
            tracking_tags=dict(getattr(args, "mlflow_tags", None) or {}),
        ),
        seed=getattr(args, "seed", 42),
        cv_folds=getattr(args, "cv_folds", None),
        plot_losses=getattr(args, "plot_losses", False),
        save_model=getattr(args, "save_model", False),
    )


def validate_parsed_args(args: argparse.Namespace) -> argparse.Namespace:
    """Post-parse validation for mutually exclusive and dependent flags."""
    run_all = getattr(args, "all", False)
    dataset_name = getattr(args, "dataset_name", None)

    # Must specify exactly one of --all or --dataset_name.
    if not run_all and not dataset_name:
        raise SystemExit("error: one of --all or --dataset_name is required")

    # --limit and --nan_level only make sense with --all.
    if not run_all:
        if getattr(args, "limit", None) is not None:
            raise SystemExit("error: --limit can only be used with --all")
        if getattr(args, "nan_level", 0) != 0:
            raise SystemExit("error: --nan_level can only be used with --all")

    # --all is incompatible with --use_optuna.
    if run_all and getattr(args, "use_optuna", False):
        raise SystemExit("error: --all and --use_optuna are mutually exclusive")

    return args


def build_training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRIDENT: Training and evaluation of the model")

    # Dataset selection: --all and --dataset_name are mutually exclusive.
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset_name", type=str, default=None)
    dataset_group.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run training on all available datasets.",
    )

    # Companion flags for --all mode.
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="When used with --all, run only the first N datasets (alphabetical).",
    )
    parser.add_argument(
        "--nan_level",
        type=int,
        default=0,
        choices=[0, 20, 40, 60, 80],
        help="When used with --all, select the missingness level (default: 0).",
    )

    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--metrics_dir", type=str, default="metrics")
    parser.add_argument("--disable_mlflow", action="store_true")
    parser.add_argument("--plot_losses", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=None)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default=None,
        choices=LR_SCHEDULER_NAMES,
        help=(
            "Learning-rate schedule for both training stages. Overrides LR_SCHEDULER from the "
            "hyperparameter file. Default: cosine_legacy (the schedule of every run before ADR 0003)."
        ),
    )
    parser.add_argument("--use_optuna", action="store_true")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--retrain_best", action="store_true")
    return parser
