"""Opt-in regression baseline for a short typed vehicle training run."""

import json
from pathlib import Path

import pytest

from src.training.runner import run_training
from src.training.types import DatasetSpec, Hyperparameters, RuntimeOptions, TrainingRequest


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "vehicle_00nan_regression.json"


@pytest.mark.integration
def test_vehicle_typed_runner_matches_regression_fixture(tmp_path) -> None:
    """A deterministic short CPU/CUDA run remains within recorded tolerances."""
    fixture = json.loads(FIXTURE_PATH.read_text())
    request = TrainingRequest(
        dataset=DatasetSpec.from_name("vehicle_00nan", "class"),
        hyperparameters=Hyperparameters.from_mapping(
            {
                "DIM": 16,
                "HIDDEN_DIM": 8,
                "HEADS": 4,
                "LAYERS": 1,
                "DIM_FEED": 16,
                "DROPOUT": 0.1,
                "EPOCHS_PRE": 2,
                "BATCH": 64,
                "LR_PRE": 0.00034,
                "WEIGHT_DECAY_PRE": 0.005,
                "PROB_MASCARA": 0.5,
                "EPOCH_FINE": 2,
                "LR_FINE": 0.001,
                "WEIGHT_DECAY_FINE": 0.0019,
                "LABELS": 4,
            }
        ),
        runtime=RuntimeOptions(
            output_dir=tmp_path / "results",
            metrics_dir=tmp_path / "metrics",
            tracking_enabled=False,
        ),
        seed=42,
        cv_folds=2,
        plot_losses=False,
        save_model=False,
    )

    result = run_training(request)
    for observed, expected in zip(result.fold_results, fixture["folds"], strict=True):
        for metric in ("accuracy", "f1_micro", "f1_macro"):
            assert observed.metrics[metric] == pytest.approx(
                expected[metric], abs=fixture["tolerance"]
            )
    for metric in ("accuracy", "f1_micro", "f1_macro"):
        assert result.mean_metrics[metric] == pytest.approx(
            fixture["mean"][metric], abs=fixture["tolerance"]
        )
