from pathlib import Path
from typing import Iterator

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import pytest

from src.training.summary import summarize_cross_validation
from src.training.tracking import create_tracker
from src.training.types import FoldResult, PreparedDataset, TrackingArtifactPaths


pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The specified dataset source can be interpreted in multiple ways:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Hint.*Inferred schema contains integer column\\(s\\):UserWarning"
    ),
]


@pytest.fixture
def mlflow_backend(tmp_path, monkeypatch) -> Iterator[str]:
    previous_tracking_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        yield tracking_uri
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking_uri)


def _dataset(tmp_path: Path) -> PreparedDataset:
    frame = pd.DataFrame({"feature": [0.0, 1.0], "class": [0, 1]})
    frame.attrs["dataset_name"] = "vehicle_00nan"
    source_path = tmp_path / "vehicle_00nan.csv"
    frame.to_csv(source_path, index=False)
    splits_path = tmp_path / "vehicle_split.json"
    splits_path.write_text("{}")
    return PreparedDataset(
        frame=frame,
        label_column="class",
        categorical_columns=(),
        numerical_columns=("feature",),
        label_classes=("car", "van"),
        source_path=source_path,
        splits_path=splits_path,
    )


def _artifact_paths(tmp_path: Path) -> TrackingArtifactPaths:
    paths = TrackingArtifactPaths(
        raw_fold_metrics_csv=tmp_path / "results" / "metrics" / "raw_fold_metrics.csv",
        summary_json=tmp_path / "results" / "metrics" / "cv_summary.json",
        manifest_json=tmp_path / "results" / "tracking" / "diagnostic_manifest.json",
        provenance_json=tmp_path / "results" / "data" / "provenance.json",
    )
    for path in (
        paths.raw_fold_metrics_csv,
        paths.summary_json,
        paths.manifest_json,
        paths.provenance_json,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    return paths


def _buffered_record(
    active_tracker,
    tmp_path: Path,
    fold: int,
    f1_macro: float,
    cv_folds: int = 2,
):
    with active_tracker.fold_run(
        fold=fold, cv_folds=cv_folds, dataset_name="vehicle_00nan"
    ) as fold_tracker:
        fold_tracker.log_metrics(
            {
                "pretrain/train_loss": 1.0 + fold,
                "pretrain/val_loss": 2.0 + fold,
                "finetune/train_loss": 3.0 + fold,
                "finetune/val_loss": 4.0 + fold,
            },
            step=7,
        )
        fold_tracker.log_metrics(
            {"test/accuracy": 0.5 + f1_macro / 2, "test/f1_macro": f1_macro}
        )
        artifact = tmp_path / f"fold_{fold}.txt"
        artifact.write_text(f"fold {fold}")
        fold_tracker.log_artifact(str(artifact), artifact_path="diagnostics")
    return fold_tracker.to_record(
        FoldResult(
            fold,
            "vehicle_00nan",
            {"accuracy": 0.5 + f1_macro / 2, "f1_macro": f1_macro},
        )
    )


def _experiment_runs(client: MlflowClient):
    experiment = client.get_experiment_by_name("TRIDENT/vehicle")
    assert experiment is not None
    return client.search_runs([experiment.experiment_id])


def _artifact_files(client: MlflowClient, run_id: str, path: str | None = None) -> set[str]:
    files: set[str] = set()
    for artifact in client.list_artifacts(run_id, path):
        if artifact.is_dir:
            files.update(_artifact_files(client, run_id, artifact.path))
        else:
            files.add(artifact.path)
    return files


def test_finalize_cross_validation_logs_parent_lineage_summary_and_selected_children(
    tmp_path, mlflow_backend
) -> None:
    tracker = create_tracker(enabled=True)
    dataset = _dataset(tmp_path)
    artifact_paths = _artifact_paths(tmp_path)

    with tracker.parent_run(
        dataset_name="vehicle_00nan", seed=42, cv_folds=3, hyperparameters={}
    ) as active_tracker:
        records = [
            _buffered_record(active_tracker, tmp_path, fold=1, f1_macro=0.4, cv_folds=3),
            _buffered_record(active_tracker, tmp_path, fold=2, f1_macro=0.6, cv_folds=3),
            _buffered_record(active_tracker, tmp_path, fold=3, f1_macro=0.8, cv_folds=3),
        ]
        active_tracker.finalize_cross_validation(
            records, summarize_cross_validation(records), dataset, artifact_paths
        )

    client = MlflowClient(tracking_uri=mlflow_backend)
    runs = _experiment_runs(client)
    parents = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    children = [run for run in runs if "mlflow.parentRunId" in run.data.tags]

    assert len(parents) == 1
    parent = client.get_run(parents[0].info.run_id)
    assert parent.data.tags["run_role"] == "parent"
    assert parent.data.tags["dataset_variant"] == "vehicle_00nan"
    assert parent.data.tags["missingness_percent"] == "0"
    assert parent.data.tags["evaluation_mode"] == "cross_validation"
    assert parent.data.metrics["cv/test/f1_macro/mean"] == pytest.approx(0.6)
    assert parent.data.metrics["cv/test/f1_macro/fold_count"] == 3.0
    assert parent.data.metrics["cv/finetune/val_loss/mean"] == pytest.approx(6.0)
    loss_history = client.get_metric_history(
        parent.info.run_id, "cv/finetune/val_loss/mean"
    )
    assert [(metric.step, metric.value) for metric in loss_history] == [(7, 6.0)]
    assert len(parent.inputs.dataset_inputs) == 1
    assert parent.inputs.dataset_inputs[0].dataset.name == "vehicle_00nan"

    assert len(children) == 2
    assert {child.data.tags["run_role"] for child in children} == {
        "best_fold",
        "worst_fold",
    }
    assert {child.data.tags["fold"] for child in children} == {"1", "3"}
    assert all(child.data.tags["fold"] != "2" for child in children)
    child_by_fold = {child.data.tags["fold"]: child for child in children}
    assert child_by_fold["1"].data.metrics["test/f1_macro"] == pytest.approx(0.4)
    assert child_by_fold["3"].data.metrics["test/f1_macro"] == pytest.approx(0.8)
    assert _artifact_files(client, child_by_fold["1"].info.run_id) == {
        "diagnostics/fold_1.txt"
    }
    assert _artifact_files(client, child_by_fold["3"].info.run_id) == {
        "diagnostics/fold_3.txt"
    }


def test_finalize_cross_validation_replays_one_diagnostic_child_for_a_tie(
    tmp_path, mlflow_backend
) -> None:
    tracker = create_tracker(enabled=True)

    with tracker.parent_run(
        dataset_name="vehicle_00nan", seed=42, cv_folds=2, hyperparameters={}
    ) as active_tracker:
        records = [
            _buffered_record(active_tracker, tmp_path, fold=1, f1_macro=0.5),
            _buffered_record(active_tracker, tmp_path, fold=2, f1_macro=0.5),
        ]
        active_tracker.finalize_cross_validation(
            records,
            summarize_cross_validation(records),
            _dataset(tmp_path),
            _artifact_paths(tmp_path),
        )

    client = MlflowClient(tracking_uri=mlflow_backend)
    children = [
        run
        for run in _experiment_runs(client)
        if "mlflow.parentRunId" in run.data.tags
    ]

    assert len(children) == 1
    assert children[0].data.tags["run_role"] == "best_and_worst"
    assert children[0].data.tags["fold"] == "1"


def test_log_single_split_record_replays_into_parent_without_a_child(
    tmp_path, mlflow_backend
) -> None:
    tracker = create_tracker(enabled=True)
    artifact = tmp_path / "single_split.txt"
    artifact.write_text("single split")

    with tracker.parent_run(
        dataset_name="vehicle_00nan", seed=42, cv_folds=None, hyperparameters={}
    ) as active_tracker:
        with active_tracker.fold_run(
            fold=1, cv_folds=None, dataset_name="vehicle_00nan"
        ) as fold_tracker:
            fold_tracker.log_metrics({"pretrain/train_loss": 1.0}, step=7)
            fold_tracker.log_metrics({"test/f1_macro": 0.4})
            fold_tracker.log_artifact(str(artifact), artifact_path="diagnostics")
        active_tracker.log_single_split_record(
            fold_tracker.to_record(
                FoldResult("single_split", "vehicle_00nan", {"f1_macro": 0.4})
            )
        )

    client = MlflowClient(tracking_uri=mlflow_backend)
    runs = _experiment_runs(client)

    assert len(runs) == 1
    parent = runs[0]
    assert parent.data.tags["run_role"] == "parent"
    assert parent.data.tags["evaluation_mode"] == "single_split"
    assert parent.data.metrics["test/f1_macro"] == pytest.approx(0.4)
    history = client.get_metric_history(parent.info.run_id, "pretrain/train_loss")
    assert [(metric.step, metric.value) for metric in history] == [(7, 1.0)]
    assert _artifact_files(client, parent.info.run_id) == {
        "diagnostics/single_split.txt"
    }


def test_finalize_cross_validation_logs_parent_artifacts_at_stable_paths(
    tmp_path, mlflow_backend
) -> None:
    tracker = create_tracker(enabled=True)
    artifact_paths = _artifact_paths(tmp_path)

    with tracker.parent_run(
        dataset_name="vehicle_00nan", seed=42, cv_folds=2, hyperparameters={}
    ) as active_tracker:
        records = [
            _buffered_record(active_tracker, tmp_path, fold=1, f1_macro=0.4),
            _buffered_record(active_tracker, tmp_path, fold=2, f1_macro=0.8),
        ]
        active_tracker.finalize_cross_validation(
            records,
            summarize_cross_validation(records),
            _dataset(tmp_path),
            artifact_paths,
        )

    client = MlflowClient(tracking_uri=mlflow_backend)
    parent = next(
        run
        for run in _experiment_runs(client)
        if "mlflow.parentRunId" not in run.data.tags
    )

    assert _artifact_files(client, parent.info.run_id) == {
        "data/provenance.json",
        "metrics/cv_summary.json",
        "metrics/raw_fold_metrics.csv",
        "tracking/diagnostic_manifest.json",
    }


def test_disabled_tracker_buffers_records_without_creating_mlflow_runs(
    tmp_path, mlflow_backend
) -> None:
    tracker = create_tracker(enabled=False)

    with tracker.parent_run(
        dataset_name="vehicle_00nan", seed=42, cv_folds=2, hyperparameters={}
    ) as active_tracker:
        records = [
            _buffered_record(active_tracker, tmp_path, fold=1, f1_macro=0.4),
            _buffered_record(active_tracker, tmp_path, fold=2, f1_macro=0.8),
        ]
        active_tracker.log_prepared_dataset(_dataset(tmp_path))
        active_tracker.finalize_cross_validation(
            records,
            summarize_cross_validation(records),
            _dataset(tmp_path),
            _artifact_paths(tmp_path),
        )
        active_tracker.log_single_split_record(records[0])

    assert records[0].metric_events[0].key == "pretrain/train_loss"
    assert records[0].artifacts[0].path.name == "fold_1.txt"
    client = MlflowClient(tracking_uri=mlflow_backend)
    assert client.get_experiment_by_name("TRIDENT/vehicle") is None
