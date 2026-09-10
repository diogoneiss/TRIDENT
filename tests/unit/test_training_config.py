from argparse import Namespace
import json
from pathlib import Path

import pytest

from src.training.config import build_training_parser, resolve_training_request
from src.training.types import DEFAULT_LR_SCHEDULER, Hyperparameters, RuntimeOptions


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


def test_parser_rejects_unknown_scheduler() -> None:
    with pytest.raises(SystemExit):
        build_training_parser().parse_args(
            ["--dataset_name", "vehicle_00nan", "--lr_scheduler", "linear"]
        )


def test_scheduler_defaults_to_legacy_when_flag_is_absent() -> None:
    args = build_training_parser().parse_args(["--dataset_name", "vehicle_00nan"])

    request = resolve_training_request(args)

    assert args.lr_scheduler is None
    assert request.hyperparameters.lr_scheduler == DEFAULT_LR_SCHEDULER == "cosine_legacy"


def test_scheduler_flag_overrides_hyperparameter_override_mapping() -> None:
    args = Namespace(
        dataset_name="vehicle_00nan",
        hyperparams_override={"DIM": 64, "LR_SCHEDULER": "plateau"},
        lr_scheduler="warmup_cosine",
    )

    request = resolve_training_request(args)

    assert request.hyperparameters.dimension == 64
    assert request.hyperparameters.lr_scheduler == "warmup_cosine"


def test_scheduler_flag_overrides_dataset_json_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "datasets" / "hiperparams" / "vehicle"
    config_dir.mkdir(parents=True)
    (config_dir / "vehicle_00nan.json").write_text(json.dumps({"DIM": 32, "LR_SCHEDULER": "cosine"}))

    from_file = resolve_training_request(Namespace(dataset_name="vehicle_00nan"))
    from_flag = resolve_training_request(
        Namespace(dataset_name="vehicle_00nan", lr_scheduler="constant")
    )

    assert from_file.hyperparameters.dimension == 32
    assert from_file.hyperparameters.lr_scheduler == "cosine"
    assert from_flag.hyperparameters.dimension == 32
    assert from_flag.hyperparameters.lr_scheduler == "constant"


def test_hyperparameters_reject_unknown_scheduler() -> None:
    with pytest.raises(ValueError, match="Unknown learning-rate scheduler"):
        Hyperparameters(lr_scheduler="linear")
    with pytest.raises(ValueError, match="Unknown learning-rate scheduler"):
        Hyperparameters.from_mapping({"LR_SCHEDULER": "linear"})


def test_legacy_namespace_defaults_to_parent_tracking_without_extra_tags() -> None:
    request = resolve_training_request(Namespace(dataset_name="vehicle_00nan"))

    assert request.runtime.tracking_run_role == "parent"
    assert request.runtime.tracking_tags == {}


def test_legacy_namespace_accepts_programmatic_tracking_role_and_tags() -> None:
    args = Namespace(
        dataset_name="vehicle_00nan",
        mlflow_run_role="optuna_trial",
        mlflow_tags={"optuna_study_run_id": "abc"},
    )

    request = resolve_training_request(args)

    assert request.runtime.tracking_run_role == "optuna_trial"
    assert request.runtime.tracking_tags == {"optuna_study_run_id": "abc"}


def test_runtime_options_reject_unknown_tracking_role() -> None:
    with pytest.raises(ValueError, match="tracking run role"):
        RuntimeOptions(tracking_run_role="child")
