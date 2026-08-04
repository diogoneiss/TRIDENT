from argparse import Namespace
from contextlib import contextmanager
import json
from pathlib import Path

import mlflow
import pandas as pd

from src.training.tracking import BufferedFoldTracker
from src.training.types import (
    DatasetSpec,
    FinetuningOutcome,
    FoldResult,
    Hyperparameters,
    PreparedDataset,
    PretrainingOutcome,
    RuntimeOptions,
    TrainingRequest,
)


class FakeTracker:
    def __init__(self) -> None:
        self.active = False
        self.fold_tracker_ids: list[int] = []
        self.logged_artifacts: list[tuple[Path, str | None]] = []
        self.lineage_datasets: list[PreparedDataset] = []
        self.finalized_records: list[int | str] | None = None
        self.finalized_artifact_paths = None
        self.single_split_records = []
        self.single_split_children = 0

    @contextmanager
    def parent_run(self, **kwargs):
        self.active = True
        try:
            yield self
        finally:
            self.active = False

    @contextmanager
    def fold_run(self, **kwargs):
        assert self.active
        fold_tracker = BufferedFoldTracker()
        self.fold_tracker_ids.append(id(fold_tracker))
        yield fold_tracker

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        assert self.active
        self.logged_artifacts.append((Path(path), artifact_path))

    def log_prepared_dataset(self, dataset: PreparedDataset) -> None:
        assert self.active
        self.lineage_datasets.append(dataset)

    def finalize_cross_validation(self, records, summary, dataset, artifact_paths) -> None:
        assert self.active
        assert self.finalized_records is None
        self.finalized_records = [record.result.fold for record in records]
        self.finalized_artifact_paths = artifact_paths

    def log_single_split_record(self, record) -> None:
        assert self.active
        self.single_split_records.append(record)


def _stub_training_runtime(monkeypatch, tmp_path, cv_folds: int | None):
    import src.training.runner as runner

    frame = pd.DataFrame({"feature": [0.0, 1.0], "class": [0, 1]})
    frame.attrs["dataset_name"] = "vehicle_00nan"
    dataset = PreparedDataset(
        frame=frame,
        label_column="class",
        categorical_columns=(),
        numerical_columns=("feature",),
        label_classes=("car", "van"),
        source_path=tmp_path / "vehicle_00nan.csv",
        splits_path=tmp_path / "vehicle_split.json",
    )
    dataset.source_path.write_text("feature,class\n0,car\n1,van\n")
    dataset.splits_path.write_text("{}")
    folds = list(range(1, (cv_folds or 1) + 1))
    fake_tracker = FakeTracker()
    pretraining_tracker_ids: list[int] = []
    finetuning_tracker_ids: list[int] = []

    def train_pretrainer(dataset, fold, hyperparameters, device, tracker):
        pretraining_tracker_ids.append(id(tracker))
        tracker.log_metrics(
            {
                "pretrain/train_loss": 1.0 + fold,
                "pretrain/val_loss": 2.0 + fold,
            },
            step=0,
        )
        return PretrainingOutcome(object(), (1.0 + fold,), (2.0 + fold,))

    def train_classifier(dataset, fold, pretraining, hyperparameters, device, tracker):
        finetuning_tracker_ids.append(id(tracker))
        tracker.log_metrics(
            {
                "finetune/train_loss": 3.0 + fold,
                "finetune/val_loss": 4.0 + fold,
            },
            step=0,
        )
        metrics = {"accuracy": 0.4 + fold / 10, "f1_macro": 0.3 + fold / 10}
        tracker.log_metrics({f"test/{name}": value for name, value in metrics.items()})
        return FinetuningOutcome(
            object(),
            FoldResult("single_split", "vehicle_00nan", metrics),
            (3.0 + fold,),
            (4.0 + fold,),
        )

    monkeypatch.setattr(runner, "prepare_dataset", lambda spec: dataset)
    monkeypatch.setattr(runner, "build_folds", lambda *args: folds)
    monkeypatch.setattr(runner, "create_tracker", lambda enabled: fake_tracker)
    monkeypatch.setattr(runner, "train_pretrainer", train_pretrainer)
    monkeypatch.setattr(runner, "train_and_evaluate_classifier", train_classifier)

    request = TrainingRequest(
        dataset=DatasetSpec.from_name("vehicle_00nan", "class"),
        hyperparameters=Hyperparameters(),
        runtime=RuntimeOptions(
            output_dir=tmp_path / "results",
            metrics_dir=tmp_path / "metrics",
            tracking_enabled=True,
        ),
        seed=42,
        cv_folds=cv_folds,
        plot_losses=False,
        save_model=False,
    )
    result = runner.run_training(request)
    return (
        result,
        fake_tracker,
        dataset,
        pretraining_tracker_ids,
        finetuning_tracker_ids,
    )


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
    ) as active_tracker:
        with active_tracker.fold_run(
            fold=1, cv_folds=2, dataset_name="vehicle_00nan"
        ) as fold_tracker:
            fold_tracker.log_metrics({"test/accuracy": 1.0})
            fold_tracker.log_artifact("unused")


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


def test_runner_uses_fold_buffers_and_finalizes_cross_validation_once(
    monkeypatch, tmp_path
) -> None:
    (
        result,
        fake_tracker,
        dataset,
        pretraining_tracker_ids,
        finetuning_tracker_ids,
    ) = _stub_training_runtime(monkeypatch, tmp_path, cv_folds=2)

    assert fake_tracker.fold_tracker_ids == pretraining_tracker_ids
    assert fake_tracker.fold_tracker_ids == finetuning_tracker_ids
    assert fake_tracker.finalized_records == [1, 2]
    assert fake_tracker.finalized_artifact_paths.provenance_json.exists()
    assert result.fold_results[0].fold == 1
    assert result.fold_results[1].fold == 2
    assert result.mean_metrics == {"accuracy": 0.55, "f1_macro": 0.45}
    assert fake_tracker.active is False


def test_runner_replays_single_split_into_parent_without_cv_artifacts_or_children(
    monkeypatch, tmp_path
) -> None:
    (
        result,
        fake_tracker,
        dataset,
        pretraining_tracker_ids,
        finetuning_tracker_ids,
    ) = _stub_training_runtime(monkeypatch, tmp_path, cv_folds=None)

    assert fake_tracker.fold_tracker_ids == pretraining_tracker_ids
    assert fake_tracker.fold_tracker_ids == finetuning_tracker_ids
    assert fake_tracker.finalized_records is None
    assert [record.result.fold for record in fake_tracker.single_split_records] == [
        "single_split"
    ]
    assert fake_tracker.single_split_children == 0
    assert fake_tracker.lineage_datasets == [dataset]
    assert {(path.name, artifact_path) for path, artifact_path in fake_tracker.logged_artifacts} == {
        ("hyperparameters.json", "parameters"),
        ("metrics.csv", "metrics"),
        ("provenance.json", "data"),
    }
    provenance_path = next(
        path for path, _ in fake_tracker.logged_artifacts if path.name == "provenance.json"
    )
    provenance = json.loads(provenance_path.read_text())
    assert provenance["split_strategy"] == "single_split"
    assert provenance["cv_folds"] is None
    assert not list((tmp_path / "results").rglob("cv_summary.json"))
    assert not list((tmp_path / "results").rglob("diagnostic_manifest.json"))
    assert result.fold_results[0].fold == "single_split"
    assert fake_tracker.active is False
