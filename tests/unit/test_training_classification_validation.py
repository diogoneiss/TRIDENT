"""A classification fold carries the validation reading of the epoch it kept (ADR 0029, decision 1)."""

import numpy as np

import src.training.runner as runner
from src.training.types import DatasetSpec, Hyperparameters, RuntimeOptions, TrainingRequest


def test_a_fold_carries_its_kept_epochs_validation_reading(monkeypatch, tmp_path) -> None:
    """``kept_epoch`` is the epoch of the lowest validation loss, the one whose weights the
    test is scored with, and the three validation numbers are read at that epoch. The test
    metrics are unchanged and never carry a validation key under ``test/``."""
    outcomes = []
    original = runner.train_and_evaluate_classifier

    def capture(*args, **kwargs):
        outcome = original(*args, **kwargs)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(runner, "train_and_evaluate_classifier", capture)
    request = TrainingRequest(
        dataset=DatasetSpec.from_name("vehicle_00nan", "class"),
        hyperparameters=Hyperparameters.from_mapping(
            {"DIM": 16, "HIDDEN_DIM": 8, "HEADS": 4, "LAYERS": 1, "DIM_FEED": 16, "EPOCHS_PRE": 1, "BATCH": 64, "EPOCH_FINE": 4}
        ),
        runtime=RuntimeOptions(output_dir=tmp_path / "results", metrics_dir=tmp_path / "metrics", tracking_enabled=False),
        seed=42,
        cv_folds=None,
        plot_losses=False,
        save_model=False,
    )

    result = runner.run_training(request)

    (fold,) = result.fold_results
    (outcome,) = outcomes
    assert fold.metrics["kept_epoch"] == int(np.argmin(outcome.validation_losses))
    for key in ("val_f1_macro", "val_f1_micro", "val_accuracy"):
        assert 0.0 <= float(fold.metrics[key]) <= 1.0
    assert {"accuracy", "f1_micro", "f1_macro"} <= set(fold.metrics)
    assert not any(key.startswith("test/val") for key in fold.metrics)


def test_the_classification_search_objective_is_refused_for_imputation() -> None:
    """The flag chooses between classification's test and validation splits (ADR 0029,
    decision 2); an imputation study ranks by --search_objective instead."""
    import pytest

    from src.training.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--dataset_name", "vehicle_40nan", "--task", "imputation", "--classification_search_objective", "validation"])
    assert parse_args(["--dataset_name", "vehicle_40nan", "--classification_search_objective", "validation"]).classification_search_objective == "validation"
