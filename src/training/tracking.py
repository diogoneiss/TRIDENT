"""MLflow adapters for the training runtime."""

from contextlib import contextmanager
import datetime
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import mlflow

from src.mlflow_utils import (
    build_fold_tags,
    build_run_tags,
    get_or_create_experiment,
    parse_missingness_percent,
    setup_mlflow,
)
from src.training.types import (
    CrossValidationSummary,
    FoldResult,
    FoldTrackingRecord,
    LoggedArtifact,
    LoggedMetric,
    PreparedDataset,
    TrackingArtifactPaths,
)


class BufferedFoldTracker:
    """Collect one fold's MLflow events until its diagnostic role is known."""

    def __init__(self) -> None:
        self.metric_events: list[LoggedMetric] = []
        self.artifacts: list[LoggedArtifact] = []

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metric_events.extend(
            LoggedMetric(key, float(value), step) for key, value in metrics.items()
        )

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append(LoggedArtifact(Path(path), artifact_path))

    def to_record(self, result: FoldResult) -> FoldTrackingRecord:
        return FoldTrackingRecord(result, tuple(self.metric_events), tuple(self.artifacts))


class DisabledTracker:
    """A tracker whose methods deliberately avoid importing MLflow side effects."""

    @contextmanager
    def parent_run(self, **kwargs) -> Iterator["DisabledTracker"]:
        yield self

    @contextmanager
    def fold_run(self, **kwargs) -> Iterator[BufferedFoldTracker]:
        yield BufferedFoldTracker()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        return None

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        return None

    def log_prepared_dataset(self, dataset: PreparedDataset) -> None:
        return None

    def finalize_cross_validation(
        self,
        records: Sequence[FoldTrackingRecord],
        summary: CrossValidationSummary,
        dataset: PreparedDataset,
        artifact_paths: TrackingArtifactPaths,
    ) -> None:
        return None

    def log_single_split_record(self, record: FoldTrackingRecord) -> None:
        return None


class MlflowTracker:
    """Log one comparison parent and only selected diagnostic fold runs."""

    def __init__(self) -> None:
        self.experiment_id: str | None = None
        self.cv_folds: int | None = None

    @contextmanager
    def parent_run(
        self,
        *,
        dataset_name: str,
        seed: int,
        cv_folds: int | None,
        hyperparameters: dict[str, object],
    ) -> Iterator["MlflowTracker"]:
        setup_mlflow()
        self.experiment_id = get_or_create_experiment(dataset_name)
        self.cv_folds = cv_folds
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tags = build_run_tags(
            dataset_name=dataset_name,
            run_type="train",
            seed=seed,
            cv_folds=cv_folds,
            extra={
                "run_role": "parent",
                "dataset_variant": dataset_name,
                "missingness_percent": parse_missingness_percent(dataset_name),
                "evaluation_mode": (
                    "cross_validation" if cv_folds is not None else "single_split"
                ),
            },
        )
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"train_{dataset_name}_{timestamp}",
            tags=tags,
        ):
            mlflow.log_params(hyperparameters)
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("seed", seed)
            if cv_folds is not None:
                mlflow.log_param("cv_folds", cv_folds)
            yield self

    @contextmanager
    def fold_run(
        self, *, fold: int, cv_folds: int | None, dataset_name: str
    ) -> Iterator[BufferedFoldTracker]:
        yield BufferedFoldTracker()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_prepared_dataset(self, dataset: PreparedDataset) -> None:
        dataset_input = mlflow.data.from_pandas(
            dataset.frame,
            source=str(dataset.source_path),
            name=dataset.frame.attrs["dataset_name"],
        )
        mlflow.log_input(dataset_input, context="training")

    def finalize_cross_validation(
        self,
        records: Sequence[FoldTrackingRecord],
        summary: CrossValidationSummary,
        dataset: PreparedDataset,
        artifact_paths: TrackingArtifactPaths,
    ) -> None:
        self.log_prepared_dataset(dataset)
        self._log_parent_summary(summary, artifact_paths)
        self._log_diagnostic_children(records, summary.diagnostic_roles)

    def log_single_split_record(self, record: FoldTrackingRecord) -> None:
        self._replay_record(record)

    def _log_parent_summary(
        self,
        summary: CrossValidationSummary,
        artifact_paths: TrackingArtifactPaths,
    ) -> None:
        for metric_name, statistics in summary.metrics.items():
            prefix = f"cv/test/{metric_name}"
            mlflow.log_metrics(
                {
                    f"{prefix}/mean": float(statistics.mean),
                    f"{prefix}/ci95_lower": float(statistics.ci95_lower),
                    f"{prefix}/ci95_upper": float(statistics.ci95_upper),
                    f"{prefix}/std": float(statistics.std),
                    f"{prefix}/min": float(statistics.minimum),
                    f"{prefix}/max": float(statistics.maximum),
                    f"{prefix}/fold_count": float(statistics.fold_count),
                }
            )

        for loss_name, bands in summary.loss_bands.items():
            prefix = f"cv/{loss_name}"
            for band in bands:
                mlflow.log_metrics(
                    {
                        f"{prefix}/mean": float(band.mean),
                        f"{prefix}/ci95_lower": float(band.ci95_lower),
                        f"{prefix}/ci95_upper": float(band.ci95_upper),
                    },
                    step=int(band.step),
                )

        parent_artifacts = (
            (artifact_paths.raw_fold_metrics_csv, "metrics"),
            (artifact_paths.summary_json, "metrics"),
            (artifact_paths.manifest_json, "tracking"),
            (artifact_paths.provenance_json, "data"),
        )
        for path, artifact_path in parent_artifacts:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def _log_diagnostic_children(
        self,
        records: Sequence[FoldTrackingRecord],
        diagnostic_roles: Mapping[int, str],
    ) -> None:
        records_by_fold = {record.result.fold: record for record in records}
        for fold, role in diagnostic_roles.items():
            record = records_by_fold[fold]
            tags = build_fold_tags(fold - 1, self.cv_folds)
            tags.update({"dataset": record.result.dataset_name, "run_role": role})
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=f"{role}_fold_{fold}",
                tags=tags,
                nested=True,
            ):
                self._replay_record(record)

    @staticmethod
    def _replay_record(record: FoldTrackingRecord) -> None:
        for event in record.metric_events:
            mlflow.log_metric(event.key, event.value, step=event.step)
        for artifact in record.artifacts:
            mlflow.log_artifact(str(artifact.path), artifact_path=artifact.artifact_path)


def create_tracker(enabled: bool):
    """Return an MLflow-backed tracker only when tracking has been requested."""
    return MlflowTracker() if enabled else DisabledTracker()
