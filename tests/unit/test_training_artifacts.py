import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.artifacts import ArtifactWriter
from src.training.types import (
    CrossValidationSummary,
    FoldResult,
    FoldTrackingRecord,
    LoggedArtifact,
    LoggedMetric,
    LossBand,
    MetricSummary,
    PreparedDataset,
)


def _dataset() -> PreparedDataset:
    frame = pd.DataFrame({"feature": [0.0, 1.0], "class": [0, 1]})
    frame.attrs["dataset_name"] = "vehicle_00nan"
    return PreparedDataset(
        frame=frame,
        label_column="class",
        categorical_columns=(),
        numerical_columns=("feature",),
        label_classes=np.array(["car", "van"]),
        source_path=Path("datasets/processed_datasets/vehicle/vehicle_00nan.csv"),
        splits_path=Path("datasets/processed_datasets/splits/vehicle_split.json"),
    )


def _summary() -> CrossValidationSummary:
    return CrossValidationSummary(
        metrics={
            "f1_macro": MetricSummary(
                mean=np.float64(0.6),
                ci95_lower=np.float64(0.1),
                ci95_upper=np.float64(1.1),
                std=np.float64(0.2),
                minimum=np.float64(0.4),
                maximum=np.float64(0.8),
                fold_count=np.int64(2),
            ),
            "loss": MetricSummary(
                mean=np.float64(0.3),
                ci95_lower=np.float64(-0.97),
                ci95_upper=np.float64(1.57),
                std=np.float64(0.14),
                minimum=np.float64(0.2),
                maximum=np.float64(0.4),
                fold_count=np.int64(2),
            ),
        },
        loss_bands={
            "finetune/val_loss": (
                LossBand(step=np.int64(0), mean=np.float64(1.5), ci95_lower=1.0, ci95_upper=2.0),
            )
        },
        diagnostic_roles={1: "worst_fold", 2: "best_fold"},
        timings={
            "total_seconds": MetricSummary(
                mean=np.float64(4.5),
                ci95_lower=np.float64(-14.56),
                ci95_upper=np.float64(23.56),
                std=np.float64(2.12),
                minimum=np.float64(3.0),
                maximum=np.float64(6.0),
                fold_count=np.int64(2),
            ),
        },
    )


def _record(
    fold: int,
    f1_macro: float,
    test_loss: float,
    artifacts: tuple[LoggedArtifact, ...] = (),
) -> FoldTrackingRecord:
    return FoldTrackingRecord(
        result=FoldResult(
            fold,
            "vehicle_00nan",
            {"accuracy": np.float64(0.5 + f1_macro / 2), "f1_macro": np.float64(f1_macro)},
        ),
        metric_events=(
            LoggedMetric("test/loss", test_loss, None),
            LoggedMetric("time/pretrain_seconds", 2.0 * fold, None),
            LoggedMetric("time/finetune_seconds", 1.0 * fold, None),
            LoggedMetric("time/total_seconds", 3.0 * fold, None),
        ),
        artifacts=artifacts,
    )


def test_write_cv_tracking_artifacts_writes_parent_contract_with_builtin_json_values(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path / "results", tmp_path / "project_metrics", "vehicle_00nan")
    records = [
        _record(1, f1_macro=0.4, test_loss=0.2),
        _record(2, f1_macro=0.8, test_loss=0.4),
    ]

    paths = writer.write_cv_tracking_artifacts(records, _summary(), _dataset(), seed=42, cv_folds=2)

    assert paths.raw_fold_metrics_csv.exists()
    assert paths.raw_fold_metrics_csv.relative_to(writer.results_dir) == Path("metrics/raw_fold_metrics.csv")
    assert paths.summary_json.relative_to(writer.results_dir) == Path("metrics/cv_summary.json")
    assert paths.manifest_json.relative_to(writer.results_dir) == Path("tracking/diagnostic_manifest.json")
    assert paths.provenance_json.relative_to(writer.results_dir) == Path("data/provenance.json")

    summary = json.loads(paths.summary_json.read_text())
    assert summary["interval"] == "two-sided 95% Student-t"
    assert summary["metrics"]["f1_macro"]["fold_count"] == 2
    assert summary["metrics"]["loss"]["mean"] == 0.3
    assert summary["loss_bands"]["finetune/val_loss"][0]["step"] == 0
    assert summary["timings"]["total_seconds"]["mean"] == 4.5
    assert summary["timings"]["total_seconds"]["fold_count"] == 2

    raw_fold_metrics = pd.read_csv(paths.raw_fold_metrics_csv)
    assert raw_fold_metrics["loss"].tolist() == [0.2, 0.4]
    assert raw_fold_metrics["pretrain_seconds"].tolist() == [2.0, 4.0]
    assert raw_fold_metrics["finetune_seconds"].tolist() == [1.0, 2.0]
    assert raw_fold_metrics["total_seconds"].tolist() == [3.0, 6.0]

    provenance = json.loads(paths.provenance_json.read_text())
    assert provenance["source_path"].endswith("vehicle_00nan.csv")
    assert provenance["splits_path"].endswith("vehicle_split.json")
    assert provenance["seed"] == 42
    assert provenance["cv_folds"] == 2
    assert provenance["prepared_schema"]["label_classes"] == ["car", "van"]

    manifest = json.loads(paths.manifest_json.read_text())
    assert manifest["diagnostic_roles"] == {"1": "worst_fold", "2": "best_fold"}
    assert manifest["f1_macro_ranking"] == {"1": 0.4, "2": 0.8}


def test_diagnostic_manifest_maps_selected_fold_artifacts_to_mlflow_destinations(
    tmp_path,
) -> None:
    writer = ArtifactWriter(tmp_path / "results", tmp_path / "project_metrics", "vehicle_00nan")
    worst_plot = writer.results_dir / "plots" / "pretrain_losses_fold_1.png"
    best_plot = writer.results_dir / "plots" / "finetune_losses_fold_3.png"
    best_model = writer.results_dir / "final_model_fold_3.pt"
    skipped_model = writer.results_dir / "final_model_fold_2.pt"
    for artifact_path in (worst_plot, best_plot, best_model, skipped_model):
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"retained diagnostic artifact")
    records = [
        _record(
            1,
            f1_macro=0.4,
            test_loss=0.5,
            artifacts=(LoggedArtifact(worst_plot, "plots"),),
        ),
        _record(
            2,
            f1_macro=0.6,
            test_loss=0.3,
            artifacts=(LoggedArtifact(skipped_model, "models"),),
        ),
        _record(
            3,
            f1_macro=0.8,
            test_loss=0.2,
            artifacts=(
                LoggedArtifact(best_plot, "plots"),
                LoggedArtifact(best_model, "models"),
            ),
        ),
    ]
    summary = CrossValidationSummary(
        metrics=_summary().metrics,
        loss_bands=_summary().loss_bands,
        diagnostic_roles={1: "worst_fold", 3: "best_fold"},
    )

    paths = writer.write_cv_tracking_artifacts(records, summary, _dataset(), seed=42, cv_folds=3)

    manifest = json.loads(paths.manifest_json.read_text())
    assert manifest["retained_artifact_paths"] == {
        "1": [{"source_path": str(worst_plot), "artifact_path": "plots"}],
        "3": [
            {"source_path": str(best_plot), "artifact_path": "plots"},
            {"source_path": str(best_model), "artifact_path": "models"},
        ],
    }
