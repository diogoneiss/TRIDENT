"""MLflow adapters for the training runtime."""

from contextlib import contextmanager
import datetime
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import mlflow

from src.mlflow_utils import (
    LR_SCHEDULER_TAG,
    build_fold_tags,
    build_run_tags,
    get_or_create_experiment,
    parse_missingness_percent,
    setup_mlflow,
)
from src.training.summary import fold_timings_for_tracking
from src.training.types import (
    CrossValidationSummary,
    FoldResult,
    FoldTrackingRecord,
    LoggedArtifact,
    LoggedMetric,
    MetricSummary,
    PreparedDataset,
    TrackingArtifactPaths,
)

# Total training wall-clock for one parent run, whatever its evaluation mode.
TRAINING_SECONDS_KEY = "time/training_seconds"


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
        self.lr_scheduler: str | None = None

    @contextmanager
    def parent_run(
        self,
        *,
        dataset_name: str,
        seed: int,
        cv_folds: int | None,
        hyperparameters: dict[str, object],
        lr_scheduler: str,
        environment: Mapping[str, str] | None = None,
        extra_tags: Mapping[str, str] | None = None,
    ) -> Iterator["MlflowTracker"]:
        setup_mlflow()
        self.experiment_id = get_or_create_experiment(dataset_name)
        self.cv_folds = cv_folds
        self.lr_scheduler = lr_scheduler
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tags = _execution_tags(
            dataset_name=dataset_name,
            run_role="parent",
            run_type="train",
            seed=seed,
            cv_folds=cv_folds,
            lr_scheduler=lr_scheduler,
            environment=environment,
            extra_tags=extra_tags,
        )
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"train_{dataset_name}_{timestamp}",
            tags=tags,
        ):
            _log_execution_params(hyperparameters, dataset_name, seed, cv_folds)
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
        timings = fold_timings_for_tracking(record)
        if "total_seconds" in timings:
            mlflow.log_metric(TRAINING_SECONDS_KEY, timings["total_seconds"])

    def _log_parent_summary(
        self,
        summary: CrossValidationSummary,
        artifact_paths: TrackingArtifactPaths,
    ) -> None:
        _log_summary_metrics(summary)
        self._log_loss_bands(summary)
        self._log_parent_artifacts(artifact_paths)

    @staticmethod
    def _log_loss_bands(summary: CrossValidationSummary) -> None:
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

    @staticmethod
    def _log_parent_artifacts(artifact_paths: TrackingArtifactPaths) -> None:
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
            if self.lr_scheduler is not None:
                tags[LR_SCHEDULER_TAG] = self.lr_scheduler
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


class OptunaTrialTracker(MlflowTracker):
    """Log a lightweight trial record into the MLflow run the caller has active.

    ``opt.py`` opens one nested run per trial under the study run. Opening a
    second top-level run from inside it is what MLflow refuses, so this tracker
    adopts that trial run instead: it tags it, logs the parameters and the
    final summary metrics, and leaves it open for the caller to close. Loss
    histories, artifacts, dataset lineage and diagnostic children are skipped
    on purpose; a study of hundreds of trials would otherwise dwarf every
    other run in the store. ``--retrain_best`` produces the full record for
    the chosen configuration.
    """

    @contextmanager
    def parent_run(
        self,
        *,
        dataset_name: str,
        seed: int,
        cv_folds: int | None,
        hyperparameters: dict[str, object],
        lr_scheduler: str,
        environment: Mapping[str, str] | None = None,
        extra_tags: Mapping[str, str] | None = None,
    ) -> Iterator["OptunaTrialTracker"]:
        active = mlflow.active_run()
        if active is None:
            raise RuntimeError(
                "optuna_trial tracking logs into the caller's active MLflow run; "
                "open the trial run (nested under the study) before training."
            )
        self.experiment_id = active.info.experiment_id
        self.cv_folds = cv_folds
        self.lr_scheduler = lr_scheduler
        mlflow.set_tags(
            _execution_tags(
                dataset_name=dataset_name,
                run_role="optuna_trial",
                run_type="optuna_trial",
                seed=seed,
                cv_folds=cv_folds,
                lr_scheduler=lr_scheduler,
                environment=environment,
                extra_tags=extra_tags,
            )
        )
        _log_execution_params(hyperparameters, dataset_name, seed, cv_folds)
        yield self

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
        _log_summary_metrics(summary)

    def log_single_split_record(self, record: FoldTrackingRecord) -> None:
        # Only the step-less final metrics and timings, not the epoch histories.
        for event in record.metric_events:
            if event.step is None:
                mlflow.log_metric(event.key, event.value)
        timings = fold_timings_for_tracking(record)
        if "total_seconds" in timings:
            mlflow.log_metric(TRAINING_SECONDS_KEY, timings["total_seconds"])


def _execution_tags(
    *,
    dataset_name: str,
    run_role: str,
    run_type: str,
    seed: int,
    cv_folds: int | None,
    lr_scheduler: str,
    environment: Mapping[str, str] | None,
    extra_tags: Mapping[str, str] | None,
) -> dict[str, str]:
    return build_run_tags(
        dataset_name=dataset_name,
        run_type=run_type,
        seed=seed,
        cv_folds=cv_folds,
        extra={
            "run_role": run_role,
            "dataset_variant": dataset_name,
            "missingness_percent": parse_missingness_percent(dataset_name),
            "evaluation_mode": (
                "cross_validation" if cv_folds is not None else "single_split"
            ),
            # A tag rather than only a param so runs can be filtered and grouped
            # by schedule in the comparison table; see ADR 0003.
            LR_SCHEDULER_TAG: lr_scheduler,
            # Hardware and library descriptors (device, gpu_name, ...) so
            # the ``time/`` metrics are only compared within one environment.
            **(environment or {}),
            **(extra_tags or {}),
        },
    )


def _log_execution_params(
    hyperparameters: dict[str, object], dataset_name: str, seed: int, cv_folds: int | None
) -> None:
    mlflow.log_params(hyperparameters)
    mlflow.log_param("dataset_name", dataset_name)
    mlflow.log_param("seed", seed)
    if cv_folds is not None:
        mlflow.log_param("cv_folds", cv_folds)


def _log_summary_metrics(summary: CrossValidationSummary) -> None:
    """Log the final CV statistics and timings; shared by parents and trials."""
    for metric_name, statistics in summary.metrics.items():
        mlflow.log_metrics(_summary_metrics(f"cv/test/{metric_name}", statistics))

    for timing_name, statistics in summary.timings.items():
        mlflow.log_metrics(_summary_metrics(f"cv/time/{timing_name}", statistics))
    if "total_seconds" in summary.timings:
        # One mode-independent column: the same key a single-split parent
        # logs, here as the sum of every fold's total.
        total = summary.timings["total_seconds"]
        mlflow.log_metric(TRAINING_SECONDS_KEY, float(total.mean) * int(total.fold_count))


def _summary_metrics(prefix: str, statistics: MetricSummary) -> dict[str, float]:
    return {
        f"{prefix}/mean": float(statistics.mean),
        f"{prefix}/ci95_lower": float(statistics.ci95_lower),
        f"{prefix}/ci95_upper": float(statistics.ci95_upper),
        f"{prefix}/std": float(statistics.std),
        f"{prefix}/min": float(statistics.minimum),
        f"{prefix}/max": float(statistics.maximum),
        f"{prefix}/fold_count": float(statistics.fold_count),
    }


def create_tracker(enabled: bool, run_role: str = "parent"):
    """Return an MLflow-backed tracker only when tracking has been requested."""
    if not enabled:
        return DisabledTracker()
    if run_role == "optuna_trial":
        return OptunaTrialTracker()
    if run_role == "parent":
        return MlflowTracker()
    raise ValueError(f"Unknown tracking run role {run_role!r}")
