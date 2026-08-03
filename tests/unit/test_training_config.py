from argparse import Namespace
from pathlib import Path

from src.training.config import build_training_parser, resolve_training_request
from src.training.types import RuntimeOptions


def test_legacy_namespace_uses_legacy_runtime_defaults() -> None:
    args = Namespace(
        dataset_name="vehicle_00nan",
        label_column="class",
        output_dir="results",
        seed=42,
        plot_losses=False,
        save_model=False,
        cv_folds=None,
    )

    request = resolve_training_request(args)

    assert request.runtime == RuntimeOptions(Path("results"), Path("metrics"), True)


def test_parser_exposes_runtime_options() -> None:
    args = build_training_parser().parse_args(
        ["--dataset_name", "vehicle_00nan", "--metrics_dir", "tmp/metrics", "--disable_mlflow"]
    )

    assert args.metrics_dir == "tmp/metrics"
    assert args.disable_mlflow is True


def test_legacy_namespace_accepts_hyperparameter_override() -> None:
    args = Namespace(dataset_name="vehicle_00nan", hyperparams_override={"DIM": 64})

    request = resolve_training_request(args)

    assert request.hyperparameters.dimension == 64
