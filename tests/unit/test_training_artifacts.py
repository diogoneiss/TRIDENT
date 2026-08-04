import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.artifacts import ArtifactWriter
from src.training.types import CrossValidationSummary, FoldResult, LossBand, MetricSummary, PreparedDataset


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
            )
        },
        loss_bands={
            "finetune/val_loss": (
                LossBand(step=np.int64(0), mean=np.float64(1.5), ci95_lower=1.0, ci95_upper=2.0),
            )
        },
        diagnostic_roles={1: "worst_fold", 2: "best_fold"},
    )


def test_write_cv_tracking_artifacts_writes_parent_contract_with_builtin_json_values(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path / "results", tmp_path / "project_metrics", "vehicle_00nan")
    fold_results = [
        FoldResult(1, "vehicle_00nan", {"accuracy": np.float64(0.5), "f1_macro": np.float64(0.4)}),
        FoldResult(2, "vehicle_00nan", {"accuracy": np.float64(1.0), "f1_macro": np.float64(0.8)}),
    ]

    paths = writer.write_cv_tracking_artifacts(fold_results, _summary(), _dataset(), seed=42, cv_folds=2)

    assert paths.raw_fold_metrics_csv.exists()
    assert paths.raw_fold_metrics_csv.relative_to(writer.results_dir) == Path("metrics/raw_fold_metrics.csv")
    assert paths.summary_json.relative_to(writer.results_dir) == Path("metrics/cv_summary.json")
    assert paths.manifest_json.relative_to(writer.results_dir) == Path("tracking/diagnostic_manifest.json")
    assert paths.provenance_json.relative_to(writer.results_dir) == Path("data/provenance.json")

    summary = json.loads(paths.summary_json.read_text())
    assert summary["interval"] == "two-sided 95% Student-t"
    assert summary["metrics"]["f1_macro"]["fold_count"] == 2
    assert summary["loss_bands"]["finetune/val_loss"][0]["step"] == 0

    provenance = json.loads(paths.provenance_json.read_text())
    assert provenance["source_path"].endswith("vehicle_00nan.csv")
    assert provenance["splits_path"].endswith("vehicle_split.json")
    assert provenance["seed"] == 42
    assert provenance["cv_folds"] == 2
    assert provenance["prepared_schema"]["label_classes"] == ["car", "van"]

    manifest = json.loads(paths.manifest_json.read_text())
    assert manifest["diagnostic_roles"] == {"1": "worst_fold", "2": "best_fold"}
    assert manifest["f1_macro_ranking"] == {"1": 0.4, "2": 0.8}
