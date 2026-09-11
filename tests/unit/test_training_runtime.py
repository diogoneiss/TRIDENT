from argparse import Namespace
from contextlib import contextmanager
import json
from dataclasses import replace
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from src.training.tracking import BufferedFoldTracker
from src.training.runner import run_training
from src.training.types import (
    CrossValidationSummary,
    DecodingOutcome,
    DatasetSpec,
    FinetuningOutcome,
    FoldResult,
    FoldSplit,
    FoldTrackingRecord,
    Hyperparameters,
    PreparedDataset,
    PretrainingOutcome,
    RuntimeOptions,
    TrackingArtifactPaths,
    TrainingRequest,
)


class FakeTracker:
    def __init__(self) -> None:
        self.active = False
        self.fold_tracker_ids: list[int] = []
        self.logged_artifacts: list[tuple[Path, str | None]] = []
        self.lineage_datasets: list[PreparedDataset] = []
        self.parent_run_kwargs: dict[str, object] | None = None
        self.finalized_records: tuple[FoldTrackingRecord, ...] | None = None
        self.finalized_summary: CrossValidationSummary | None = None
        self.finalized_dataset: PreparedDataset | None = None
        self.finalized_artifact_paths: TrackingArtifactPaths | None = None
        self.finalization_calls = 0
        self.single_split_record: FoldTrackingRecord | None = None

    @contextmanager
    def parent_run(self, **kwargs):
        assert self.parent_run_kwargs is None
        self.parent_run_kwargs = kwargs
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

    def finalize_cross_validation(
        self,
        records: list[FoldTrackingRecord],
        summary: CrossValidationSummary,
        dataset: PreparedDataset,
        artifact_paths: TrackingArtifactPaths,
    ) -> None:
        assert self.active
        assert self.finalized_records is None
        self.finalized_records = tuple(records)
        self.finalized_summary = summary
        self.finalized_dataset = dataset
        self.finalized_artifact_paths = artifact_paths
        self.finalization_calls += 1
        self.log_prepared_dataset(dataset)
        for path, artifact_path in (
            (artifact_paths.raw_fold_metrics_csv, "metrics"),
            (artifact_paths.summary_json, "metrics"),
            (artifact_paths.manifest_json, "tracking"),
            (artifact_paths.provenance_json, "data"),
        ):
            self.log_artifact(str(path), artifact_path)

    def log_single_split_record(self, record: FoldTrackingRecord) -> None:
        assert self.active
        assert self.single_split_record is None
        self.single_split_record = record


_ENVIRONMENT = {
    "device": "cpu",
    "gpu_name": "none",
    "torch_version": "0.0.0",
    "cuda_version": "none",
}
_TIMING_KEYS = {"time/pretrain_seconds", "time/finetune_seconds", "time/total_seconds"}


def _timing_events(record: FoldTrackingRecord) -> dict[str, float]:
    events = [event for event in record.metric_events if event.key in _TIMING_KEYS]
    assert all(event.step is None for event in events)
    return {event.key: event.value for event in events}


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
        metrics = {
            1: {"accuracy": 0.5, "f1_macro": 0.4},
            2: {"accuracy": 0.6, "f1_macro": 0.5},
        }[fold]
        tracker.log_metrics(
            {
                **{f"test/{name}": value for name, value in metrics.items()},
                "test/loss": {1: 0.4, 2: 0.2}[fold],
            }
        )
        return FinetuningOutcome(
            object(),
            FoldResult("single_split", "vehicle_00nan", metrics),
            (3.0 + fold,),
            (4.0 + fold,),
        )

    monkeypatch.setattr(runner, "prepare_dataset", lambda spec: dataset)
    monkeypatch.setattr(runner, "build_folds", lambda *args: folds)
    monkeypatch.setattr(
        runner, "create_tracker", lambda enabled, run_role="parent": fake_tracker
    )
    monkeypatch.setattr(runner, "runtime_environment_tags", lambda device: dict(_ENVIRONMENT))
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
        dataset_name="vehicle_00nan",
        seed=42,
        cv_folds=2,
        hyperparameters={},
        lr_scheduler="cosine_legacy",
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

    assert fake_tracker.parent_run_kwargs == {
        "dataset_name": "vehicle_00nan",
        "seed": 42,
        "cv_folds": 2,
        "hyperparameters": {
            "DIM": 128,
            "HIDDEN_DIM": 16,
            "HEADS": 16,
            "LAYERS": 2,
            "DIM_FEED": 32,
            "DROPOUT": 0.2,
            "EPOCHS_PRE": 300,
            "BATCH": 256,
            "LR_PRE": 0.00034,
            "WEIGHT_DECAY_PRE": 0.005,
            "PROB_MASCARA": 0.5,
            "EPOCH_FINE": 150,
            "LR_FINE": 0.001,
            "WEIGHT_DECAY_FINE": 0.0019,
            "LABELS": 4,
            "LR_SCHEDULER": "cosine_legacy",
        },
        "lr_scheduler": "cosine_legacy",
        "environment": _ENVIRONMENT,
        "extra_tags": {},
        "task": "classification",
        # No file and no override behind the minimal request, so the run says so.
        "config_source": "defaults",
    }
    assert fake_tracker.fold_tracker_ids == pretraining_tracker_ids
    assert fake_tracker.fold_tracker_ids == finetuning_tracker_ids
    assert fake_tracker.finalization_calls == 1
    assert fake_tracker.finalized_records is not None
    assert [record.result.fold for record in fake_tracker.finalized_records] == [1, 2]
    assert [record.result.metrics["accuracy"] for record in fake_tracker.finalized_records] == [
        0.5,
        0.6,
    ]
    assert [record.result.metrics["f1_macro"] for record in fake_tracker.finalized_records] == [
        0.4,
        0.5,
    ]
    assert fake_tracker.finalized_dataset is dataset
    assert fake_tracker.lineage_datasets[0] is dataset
    assert fake_tracker.finalized_summary is not None
    assert fake_tracker.finalized_summary.metrics["accuracy"].mean == 0.55
    assert fake_tracker.finalized_summary.metrics["f1_macro"].mean == 0.45
    assert fake_tracker.finalized_summary.metrics["loss"].mean == pytest.approx(0.3)
    assert fake_tracker.finalized_summary.loss_bands["pretrain/train_loss"][0].mean == 2.5
    assert fake_tracker.finalized_summary.loss_bands["finetune/val_loss"][0].mean == 5.5
    assert fake_tracker.finalized_summary.diagnostic_roles == {
        1: "worst_fold",
        2: "best_fold",
    }
    for record in fake_tracker.finalized_records:
        timings = _timing_events(record)
        assert set(timings) == _TIMING_KEYS
        assert all(value >= 0 for value in timings.values())
        assert timings["time/total_seconds"] == pytest.approx(
            timings["time/pretrain_seconds"] + timings["time/finetune_seconds"]
        )
    assert set(fake_tracker.finalized_summary.timings) == {
        "pretrain_seconds",
        "finetune_seconds",
        "total_seconds",
    }
    assert fake_tracker.finalized_summary.timings["total_seconds"].fold_count == 2
    assert fake_tracker.finalized_artifact_paths is not None
    assert all(
        path.exists()
        for path in (
            fake_tracker.finalized_artifact_paths.raw_fold_metrics_csv,
            fake_tracker.finalized_artifact_paths.summary_json,
            fake_tracker.finalized_artifact_paths.manifest_json,
            fake_tracker.finalized_artifact_paths.provenance_json,
        )
    )
    results_dir = fake_tracker.finalized_artifact_paths.raw_fold_metrics_csv.parents[1]
    raw_fold_metrics = pd.read_csv(fake_tracker.finalized_artifact_paths.raw_fold_metrics_csv)
    assert raw_fold_metrics["loss"].tolist() == [0.4, 0.2]
    assert {"pretrain_seconds", "finetune_seconds", "total_seconds"} <= set(raw_fold_metrics.columns)
    assert (raw_fold_metrics["total_seconds"] >= 0).all()
    summary_payload = json.loads(fake_tracker.finalized_artifact_paths.summary_json.read_text())
    assert summary_payload["metrics"]["loss"]["mean"] == pytest.approx(0.3)
    assert summary_payload["timings"]["total_seconds"]["fold_count"] == 2
    assert set(fake_tracker.logged_artifacts) == {
        (results_dir / "hyperparameters.json", "parameters"),
        (results_dir / "metrics.csv", "metrics"),
        (fake_tracker.finalized_artifact_paths.raw_fold_metrics_csv, "metrics"),
        (fake_tracker.finalized_artifact_paths.summary_json, "metrics"),
        (fake_tracker.finalized_artifact_paths.manifest_json, "tracking"),
        (fake_tracker.finalized_artifact_paths.provenance_json, "data"),
    }
    assert result.fold_results[0].fold == 1
    assert result.fold_results[1].fold == 2
    assert all("loss" not in fold_result.metrics for fold_result in result.fold_results)
    # Timing never enters the deterministic fold metrics or the Optuna return shape.
    assert all(
        not any(key.endswith("_seconds") for key in fold_result.metrics)
        for fold_result in result.fold_results
    )
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

    assert fake_tracker.parent_run_kwargs is not None
    assert fake_tracker.parent_run_kwargs["dataset_name"] == "vehicle_00nan"
    assert fake_tracker.parent_run_kwargs["seed"] == 42
    assert fake_tracker.parent_run_kwargs["cv_folds"] is None
    assert fake_tracker.fold_tracker_ids == pretraining_tracker_ids
    assert fake_tracker.fold_tracker_ids == finetuning_tracker_ids
    assert fake_tracker.finalization_calls == 0
    assert fake_tracker.finalized_records is None
    assert fake_tracker.single_split_record is not None
    assert fake_tracker.single_split_record.result is result.fold_results[0]
    assert set(_timing_events(fake_tracker.single_split_record)) == _TIMING_KEYS
    assert fake_tracker.parent_run_kwargs["environment"] == _ENVIRONMENT
    assert fake_tracker.lineage_datasets[0] is dataset
    assert {(path.name, artifact_path) for path, artifact_path in fake_tracker.logged_artifacts} == {
        ("hyperparameters.json", "parameters"),
        ("metrics.csv", "metrics"),
        ("provenance.json", "data"),
    }
    provenance_path = next(
        path for path, _ in fake_tracker.logged_artifacts if path.name == "provenance.json"
    )
    assert all(path.exists() for path, _ in fake_tracker.logged_artifacts)
    provenance = json.loads(provenance_path.read_text())
    assert provenance["split_strategy"] == "single_split"
    assert provenance["cv_folds"] is None
    assert not list((tmp_path / "results").rglob("cv_summary.json"))
    assert not list((tmp_path / "results").rglob("diagnostic_manifest.json"))
    assert result.fold_results[0].fold == "single_split"
    assert fake_tracker.active is False


def _minimal_request(tmp_path, task: str, cv_folds=None) -> TrainingRequest:
    return TrainingRequest(
        dataset=DatasetSpec.from_name("vehicle_00nan", "class"),
        hyperparameters=Hyperparameters(),
        runtime=RuntimeOptions(
            output_dir=tmp_path / "results",
            metrics_dir=tmp_path / "metrics",
            tracking_enabled=False,
        ),
        seed=42,
        cv_folds=cv_folds,
        plot_losses=False,
        save_model=False,
        task=task,
    )


def _stub_stages(monkeypatch, tmp_path, folds=1, scores=(0.5,)):
    """Replace both second stages with recorders, leaving the runner's choice on show."""
    import src.training.runner as runner

    frame = pd.DataFrame({"feature": [0.0, 1.0], "class": [0, 1]})
    frame.attrs["dataset_name"] = "vehicle_00nan"
    dataset = PreparedDataset(
        frame=frame, label_column="class", categorical_columns=(),
        numerical_columns=("feature",), label_classes=("car", "van"),
        source_path=tmp_path / "vehicle_00nan.csv", splits_path=tmp_path / "split.json",
    )
    dataset.source_path.write_text("feature,class\n0,car\n1,van\n")
    called: list[str] = []

    monkeypatch.setattr(runner, "prepare_dataset", lambda spec: dataset)
    monkeypatch.setattr(
        runner, "build_folds", lambda *args, **kwargs: [FoldSplit([0], [1], [1])] * folds
    )
    def pretrainer(dataset, fold, hyperparameters, device, tracker):
        tracker.log_metrics({"pretrain/train_loss": 1.0, "pretrain/val_loss": 2.0}, step=0)
        return PretrainingOutcome(object(), (1.0,), (2.0,))

    monkeypatch.setattr(runner, "train_pretrainer", pretrainer)

    def classifier(dataset, fold, pretraining, hyperparameters, device, tracker):
        called.append("classification")
        tracker.log_metrics({"finetune/train_loss": 1.0, "finetune/val_loss": 1.0}, step=0)
        return FinetuningOutcome(
            object(), FoldResult("single_split", "vehicle_00nan", {"f1_macro": 0.5}), (1.0,), (1.0,)
        )

    def decoder(**kwargs):
        called.append("imputation")
        kwargs["tracker"].log_metrics({"decode/train_loss": 1.0, "decode/val_loss": 1.0}, step=0)
        score = scores[(len(called) - 1) % len(scores)]
        return DecodingOutcome(
            object(),
            FoldResult(
                "single_split", "vehicle_00nan",
                {"impute/masked/impute_score": score, "impute/masked/rmse_num_z": score},
            ),
            (1.0,), (1.0,),
            pd.DataFrame(
                [{"row": 0, "column": "feature", "kind": "numerical", "population": "masked",
                  "actual": 0.0, "imputed": score, "confidence": float("nan")}]
            ),
        )

    monkeypatch.setattr(runner, "train_and_evaluate_classifier", classifier)
    monkeypatch.setattr(runner, "train_and_evaluate_decoder", decoder)
    return called


def test_the_runner_runs_the_stage_its_task_asks_for_and_only_that_one(
    monkeypatch, tmp_path
) -> None:
    """A classification run must never reach the decoder, which is what keeps it untouched."""
    called = _stub_stages(monkeypatch, tmp_path)

    run_training(_minimal_request(tmp_path / "a", "classification"))
    run_training(_minimal_request(tmp_path / "b", "imputation"))

    assert called == ["classification", "imputation"]


def test_an_imputation_run_calls_its_best_fold_the_one_with_the_lowest_score(
    monkeypatch, tmp_path
) -> None:
    """Lower is better for an error ratio, so best and worst are the other way round.

    Selecting by the classification rule would hand back the worst fold for diagnosis
    without raising anything, which is the quietest way for this to go wrong.
    """
    _stub_stages(monkeypatch, tmp_path, folds=2, scores=(0.9, 0.4))

    run_training(_minimal_request(tmp_path / "run", "imputation", cv_folds=2))

    manifest = json.loads(
        next((tmp_path / "run" / "results").rglob("diagnostic_manifest.json")).read_text()
    )
    assert manifest["diagnostic_roles"] == {"1": "worst_fold", "2": "best_fold"}
    assert manifest["impute_score_ranking"] == {"1": 0.9, "2": 0.4}


def test_an_imputation_run_leaves_a_preview_and_a_ledger_for_every_fold(
    monkeypatch, tmp_path
) -> None:
    """The files are the point of the task, so they are written without being asked for."""
    _stub_stages(monkeypatch, tmp_path, folds=2, scores=(0.9, 0.4))

    run_training(_minimal_request(tmp_path / "run", "imputation", cv_folds=2))

    results = tmp_path / "run" / "results"
    assert len(list(results.rglob("*_preview.md"))) == 2
    assert len(list(results.rglob("*_cells.csv"))) == 2
    assert len(list(results.rglob("per_column_imputation.csv"))) == 1


def test_a_classification_run_writes_no_imputation_artifacts(monkeypatch, tmp_path) -> None:
    """Nothing about the new task should appear in a run that never used it."""
    _stub_stages(monkeypatch, tmp_path)

    run_training(_minimal_request(tmp_path / "run", "classification"))

    assert list((tmp_path / "run" / "results").rglob("*_preview.md")) == []


def test_a_loss_plot_is_named_after_the_stage_that_produced_it(monkeypatch, tmp_path) -> None:
    """An imputation run's curves are the decode stage's, so calling them fine-tuning lies."""
    _stub_stages(monkeypatch, tmp_path)
    request = _minimal_request(tmp_path / "run", "imputation")
    request = replace(request, plot_losses=True)

    run_training(request)

    plots = {path.name for path in (tmp_path / "run" / "results").rglob("*.png")}
    assert plots == {"pretrain_losses.png", "decode_losses.png"}
