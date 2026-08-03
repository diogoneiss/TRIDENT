"""Translation from legacy argparse namespaces to typed training requests."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .types import DatasetSpec, Hyperparameters, RuntimeOptions, TrainingRequest


def load_hyperparameters(args: argparse.Namespace) -> Hyperparameters:
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
        ),
        seed=getattr(args, "seed", 42),
        cv_folds=getattr(args, "cv_folds", None),
        plot_losses=getattr(args, "plot_losses", False),
        save_model=getattr(args, "save_model", False),
    )


def build_training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRIDENT: Training and evaluation of the model")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--metrics_dir", type=str, default="metrics")
    parser.add_argument("--disable_mlflow", action="store_true")
    parser.add_argument("--plot_losses", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=None)
    parser.add_argument("--use_optuna", action="store_true")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--retrain_best", action="store_true")
    return parser
