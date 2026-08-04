"""Filesystem artifacts emitted by a training runtime."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .types import (
    CrossValidationSummary,
    FoldResult,
    Hyperparameters,
    PreparedDataset,
    TrackingArtifactPaths,
)


class ArtifactWriter:
    def __init__(self, output_dir: Path, metrics_dir: Path, dataset_name: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = output_dir / dataset_name / timestamp
        self.metrics_dir = metrics_dir
        self.dataset_name = dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def write_hyperparameters(self, hyperparameters: Hyperparameters) -> Path:
        values = {
            "DIM": hyperparameters.dimension,
            "HIDDEN_DIM": hyperparameters.hidden_dimension,
            "HEADS": hyperparameters.heads,
            "LAYERS": hyperparameters.layers,
            "DIM_FEED": hyperparameters.feedforward_dimension,
            "DROPOUT": hyperparameters.dropout,
            "EPOCHS_PRE": hyperparameters.pretraining_epochs,
            "BATCH": hyperparameters.batch_size,
            "LR_PRE": hyperparameters.pretraining_learning_rate,
            "WEIGHT_DECAY_PRE": hyperparameters.pretraining_weight_decay,
            "PROB_MASCARA": hyperparameters.mask_probability,
            "EPOCH_FINE": hyperparameters.finetuning_epochs,
            "LR_FINE": hyperparameters.finetuning_learning_rate,
            "WEIGHT_DECAY_FINE": hyperparameters.finetuning_weight_decay,
            "LABELS": hyperparameters.labels,
        }
        path = self.results_dir / "hyperparameters.json"
        path.write_text(json.dumps(values, indent=4))
        print(f"Hyperparameters saved to: {path}")
        return path

    def write_metrics(self, fold_results: list[FoldResult]) -> pd.DataFrame:
        rows = [{"fold": result.fold, "dataset": result.dataset_name, **result.metrics} for result in fold_results]
        frame = pd.DataFrame(rows)
        result_path = self.results_dir / "metrics.csv"
        frame.to_csv(result_path, index=False)
        print(f"Metrics saved to: {result_path}")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        root_path = self.metrics_dir / f"{self.dataset_name}_metrics.csv"
        frame.to_csv(root_path, index=False)
        print(f"Metrics also saved to: {root_path}")
        return frame

    def write_cv_tracking_artifacts(
        self,
        fold_results: Sequence[FoldResult],
        summary: CrossValidationSummary,
        dataset: PreparedDataset,
        seed: int,
        cv_folds: int,
    ) -> TrackingArtifactPaths:
        """Write the parent-run CSV, summary, diagnostic manifest, and lineage."""
        paths = TrackingArtifactPaths(
            raw_fold_metrics_csv=self.results_dir / "metrics" / "raw_fold_metrics.csv",
            summary_json=self.results_dir / "metrics" / "cv_summary.json",
            manifest_json=self.results_dir / "tracking" / "diagnostic_manifest.json",
            provenance_json=self.write_tracking_provenance(dataset, seed, cv_folds),
        )
        for path in (
            paths.raw_fold_metrics_csv,
            paths.summary_json,
            paths.manifest_json,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {"fold": result.fold, "dataset": result.dataset_name, **result.metrics}
            for result in fold_results
        ]
        pd.DataFrame(rows).to_csv(paths.raw_fold_metrics_csv, index=False)

        summary_payload = {
            "interval": "two-sided 95% Student-t",
            "interval_interpretation": (
                "Internal CV uncertainty; not an independent-test generalization guarantee."
            ),
            "formula": "mean ± t(0.975, n - 1) * sample_std / sqrt(n)",
            "metrics": summary.metrics,
            "loss_bands": summary.loss_bands,
        }
        _write_json(paths.summary_json, summary_payload)

        f1_macro_ranking = {
            str(result.fold): result.metrics["f1_macro"]
            for result in fold_results
            if "f1_macro" in result.metrics
        }
        manifest_payload = {
            "diagnostic_roles": {str(fold): role for fold, role in summary.diagnostic_roles.items()},
            "f1_macro_ranking": f1_macro_ranking,
            "selected_folds": [
                {
                    "fold": fold,
                    "role": role,
                    "f1_macro": f1_macro_ranking.get(str(fold)),
                }
                for fold, role in summary.diagnostic_roles.items()
            ],
            "retained_artifact_paths": {},
        }
        _write_json(paths.manifest_json, manifest_payload)

        return paths

    def write_tracking_provenance(
        self,
        dataset: PreparedDataset,
        seed: int,
        cv_folds: int | None,
    ) -> Path:
        """Write prepared-dataset lineage for a parent training run."""
        path = self.results_dir / "data" / "provenance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_path": dataset.source_path,
            "splits_path": dataset.splits_path,
            "dataset_name": dataset.frame.attrs.get("dataset_name", self.dataset_name),
            "prepared_schema": {
                "label_column": dataset.label_column,
                "categorical_columns": dataset.categorical_columns,
                "numerical_columns": dataset.numerical_columns,
                "label_classes": dataset.label_classes,
            },
            "preparation": {
                "label_encoding": "LabelEncoder",
                "numerical_scaling": "StandardScaler",
            },
            "split_strategy": (
                "cross_validation" if cv_folds is not None else "single_split"
            ),
            "cv_folds": cv_folds,
            "seed": seed,
        }
        _write_json(path, payload)
        return path

    def write_loss_plot(self, name: str, train_losses: list[float], validation_losses: list[float]) -> Path:
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        path = plots_dir / name
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(name.removesuffix(".png").replace("_", " ").title())
        plt.legend()
        plt.savefig(path)
        plt.close()
        return path

    def save_model(self, name: str, model: object) -> Path:
        path = self.results_dir / name
        torch.save(model, path)
        print(f"Final model saved to: {path}")
        return path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(_to_builtin(payload), indent=2))


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return _to_builtin(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value
