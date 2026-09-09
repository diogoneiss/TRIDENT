"""Immutable values exchanged between the training workflow stages."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

#TODO store those defaults elsewhere
@dataclass(frozen=True)
class Hyperparameters:
    dimension: int = 128
    hidden_dimension: int = 16
    heads: int = 16
    layers: int = 2
    feedforward_dimension: int = 32
    dropout: float = 0.2
    pretraining_epochs: int = 100
    finetuning_epochs: int = 75

    batch_size: int = 256
    pretraining_learning_rate: float = 0.00034
    pretraining_weight_decay: float = 0.005
    mask_probability: float = 0.5
    finetuning_learning_rate: float = 0.001
    finetuning_weight_decay: float = 0.0019
    labels: int = 4

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Hyperparameters":
        return cls(
            dimension=int(values.get("DIM", values.get("dimension", 128))),
            hidden_dimension=int(values.get("HIDDEN_DIM", values.get("hidden_dimension", 16))),
            heads=int(values.get("HEADS", values.get("heads", 16))),
            layers=int(values.get("LAYERS", values.get("layers", 2))),
            feedforward_dimension=int(values.get("DIM_FEED", values.get("feedforward_dimension", 32))),
            dropout=float(values.get("DROPOUT", values.get("dropout", 0.2))),
            pretraining_epochs=int(values.get("EPOCHS_PRE", values.get("pretraining_epochs", 100))),
            batch_size=int(values.get("BATCH", values.get("batch_size", 256))),
            pretraining_learning_rate=float(
                values.get("LR_PRE", values.get("pretraining_learning_rate", 0.00034))
            ),
            pretraining_weight_decay=float(
                values.get("WEIGHT_DECAY_PRE", values.get("pretraining_weight_decay", 0.005))
            ),
            mask_probability=float(values.get("PROB_MASCARA", values.get("mask_probability", 0.5))),
            finetuning_epochs=int(values.get("EPOCH_FINE", values.get("finetuning_epochs", 75))),
            finetuning_learning_rate=float(
                values.get("LR_FINE", values.get("finetuning_learning_rate", 0.001))
            ),
            finetuning_weight_decay=float(
                values.get("WEIGHT_DECAY_FINE", values.get("finetuning_weight_decay", 0.0019))
            ),
            labels=int(values.get("LABELS", values.get("labels", 4))),
        )


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    base_dataset_name: str
    label_column: str

    @classmethod
    def from_name(cls, dataset_name: str, label_column: str | None) -> "DatasetSpec":
        return cls(
            dataset_name=dataset_name,
            base_dataset_name=dataset_name.split("_", 1)[0],
            label_column=label_column or "class",
        )


@dataclass(frozen=True)
class RuntimeOptions:
    output_dir: Path = Path("results")
    metrics_dir: Path = Path("metrics")
    tracking_enabled: bool = True


@dataclass(frozen=True)
class FoldSplit:
    train_indices: Sequence[int]
    validation_indices: Sequence[int]
    test_indices: Sequence[int]


@dataclass(frozen=True)
class PreparedDataset:
    frame: Any
    label_column: str
    categorical_columns: Sequence[str]
    numerical_columns: Sequence[str]
    label_classes: Sequence[Any]
    source_path: Path
    splits_path: Path


@dataclass(frozen=True)
class PretrainingOutcome:
    model: Any
    train_losses: Sequence[float]
    validation_losses: Sequence[float]


@dataclass(frozen=True)
class FoldResult:
    fold: int | str
    dataset_name: str
    metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class LoggedMetric:
    key: str
    value: float
    step: int | None


@dataclass(frozen=True)
class LoggedArtifact:
    path: Path
    artifact_path: str | None


@dataclass(frozen=True)
class FoldTrackingRecord:
    result: FoldResult
    metric_events: Sequence[LoggedMetric]
    artifacts: Sequence[LoggedArtifact]


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    ci95_lower: float
    ci95_upper: float
    std: float
    minimum: float
    maximum: float
    fold_count: int


@dataclass(frozen=True)
class LossBand:
    step: int
    mean: float
    ci95_lower: float
    ci95_upper: float


@dataclass(frozen=True)
class CrossValidationSummary:
    metrics: Mapping[str, MetricSummary]
    loss_bands: Mapping[str, Sequence[LossBand]]
    diagnostic_roles: Mapping[int, str]


@dataclass(frozen=True)
class TrackingArtifactPaths:
    raw_fold_metrics_csv: Path
    summary_json: Path
    manifest_json: Path
    provenance_json: Path


@dataclass(frozen=True)
class FinetuningOutcome:
    model: Any
    result: FoldResult
    train_losses: Sequence[float]
    validation_losses: Sequence[float]


@dataclass(frozen=True)
class TrainingResult:
    fold_results: Sequence[FoldResult]
    mean_metrics: dict[str, float]


@dataclass(frozen=True)
class TrainingRequest:
    dataset: DatasetSpec
    hyperparameters: Hyperparameters
    runtime: RuntimeOptions
    seed: int
    cv_folds: int | None
    plot_losses: bool
    save_model: bool


class TrainingTracker(Protocol):
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics at an optional training step."""
