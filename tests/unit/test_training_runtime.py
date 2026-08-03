from argparse import Namespace
import mlflow


def test_disabled_tracker_never_calls_mlflow_apis(monkeypatch) -> None:
    """Disabled tracking must keep every MLflow API completely untouched."""
    from src.training.tracking import create_tracker

    def fail_if_called(*args, **kwargs):
        raise AssertionError("MLflow must not be called when tracking is disabled")

    monkeypatch.setattr(mlflow, "autolog", fail_if_called)
    monkeypatch.setattr(mlflow, "start_run", fail_if_called)
    monkeypatch.setattr(mlflow, "log_metrics", fail_if_called)

    tracker = create_tracker(enabled=False)
    with tracker.parent_run(
        dataset_name="vehicle_00nan", seed=42, cv_folds=2, hyperparameters={}
    ):
        with tracker.fold_run(fold=1, cv_folds=2, dataset_name="vehicle_00nan"):
            tracker.log_metrics({"test/accuracy": 1.0})
            tracker.log_artifact("unused")


def test_train_main_preserves_legacy_cross_validation_return_shape(monkeypatch) -> None:
    """The compatibility facade returns Optuna's flat mean-metrics mapping."""
    import train
    from src.training.types import FoldResult, TrainingResult

    expected = TrainingResult(
        fold_results=(
            FoldResult(1, "vehicle_00nan", {"accuracy": 0.5, "f1_macro": 0.4}),
            FoldResult(2, "vehicle_00nan", {"accuracy": 1.0, "f1_macro": 0.8}),
        ),
        mean_metrics={"accuracy": 0.75, "f1_macro": 0.6},
    )
    monkeypatch.setattr(train, "run_from_namespace", lambda args, return_metrics: {
        "dataset": "vehicle_00nan", **expected.mean_metrics
    })

    result = train.main(
        Namespace(dataset_name="vehicle_00nan", cv_folds=2), return_metrics=True
    )

    assert result == {"dataset": "vehicle_00nan", "accuracy": 0.75, "f1_macro": 0.6}
