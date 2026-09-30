"""MLflow adapters for the training runtime."""

from contextlib import contextmanager
import datetime
import logging
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.time import get_current_time_millis

from src.mlflow_utils import (
    IS_MIRROR_TAG,
    IS_OPTUNA_TAG,
    LR_SCHEDULER_TAG,
    TASK_TAG,
    build_fold_tags,
    build_run_tags,
    get_or_create_experiment,
    parse_missingness_percent,
    setup_mlflow,
)
from src.training.mirroring import METRICS_PER_BATCH, chunks, metric, mirror_run_tree
from src.training.summary import fold_timings_for_tracking
from src.training.types import (
    DEFAULT_TASK,
    CrossValidationSummary,
    FoldKey,
    FoldResult,
    FoldTrackingRecord,
    LoggedArtifact,
    LoggedMetric,
    MetricSummary,
    PreparedDataset,
    TrackingArtifactPaths,
)

logger = logging.getLogger(__name__)

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

    def __init__(self, mirror_runs: bool = True) -> None:
        self.experiment_id: str | None = None
        self.cv_folds: int | None = None
        self.lr_scheduler: str | None = None
        self.task: str = DEFAULT_TASK
        self.is_optuna: bool = False
        self.mirror_runs = mirror_runs

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
        task: str = DEFAULT_TASK,
        is_optuna: bool = False,
        config_source: str = "defaults",
    ) -> Iterator["MlflowTracker"]:
        setup_mlflow()
        self.experiment_id = get_or_create_experiment(dataset_name)
        self.cv_folds = cv_folds
        self.lr_scheduler = lr_scheduler
        self.task = task
        self.is_optuna = is_optuna
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tags = execution_tags(
            dataset_name=dataset_name,
            run_role="parent",
            run_type="train",
            seed=seed,
            cv_folds=cv_folds,
            lr_scheduler=lr_scheduler,
            environment=environment,
            extra_tags=extra_tags,
            task=task,
            is_optuna=is_optuna,
        )
        # An imputation run reads first in a run list and sorts beside its own kind.
        prefix = "impute" if task == "imputation" else "train"
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"{prefix}_{dataset_name}_{timestamp}",
            tags=tags,
        ) as run:
            _log_execution_params(hyperparameters, dataset_name, seed, cv_folds, task, config_source)
            yield self
        # Only reached when the block above closed normally: an exception through the
        # ``yield`` skips it, so a failed run is left to ``scripts/mirror_runs.py``. The
        # diagnostic children are logged by then, so the whole tree goes at once.
        if self.mirror_runs:
            mirror_root_safely(run.info.run_id)

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
            _log_metric_events([LoggedMetric(TRAINING_SECONDS_KEY, timings["total_seconds"], None)])

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
        _log_metric_events(
            [
                LoggedMetric(f"cv/{loss_name}/{statistic}", float(value), int(band.step))
                for loss_name, bands in summary.loss_bands.items()
                for band in bands
                for statistic, value in (
                    ("mean", band.mean),
                    ("ci95_lower", band.ci95_lower),
                    ("ci95_upper", band.ci95_upper),
                )
            ]
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
        diagnostic_roles: Mapping[FoldKey, str],
    ) -> None:
        records_by_fold = {record.result.fold: record for record in records}
        for fold, role in diagnostic_roles.items():
            record = records_by_fold[fold]
            # Diagnostic children exist only for a cross-validated parent, whose folds are
            # numbered; the single-split path reports ``"single_split"`` and never reaches
            # here. Asserted rather than branched: a string arriving here already raised
            # ``TypeError`` on the arithmetic below, so this states the invariant without
            # changing which inputs succeed.
            assert isinstance(fold, int)
            tags = build_fold_tags(fold - 1, self.cv_folds)
            tags.update({"dataset": record.result.dataset_name, "run_role": role})
            if self.lr_scheduler is not None:
                tags[LR_SCHEDULER_TAG] = self.lr_scheduler
            tags[TASK_TAG] = self.task
            tags[IS_OPTUNA_TAG] = _flag(self.is_optuna)
            tags[IS_MIRROR_TAG] = _flag(False)
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=f"{role}_fold_{fold}",
                tags=tags,
                nested=True,
            ):
                self._replay_record(record)

    @staticmethod
    def _replay_record(record: FoldTrackingRecord) -> None:
        _log_metric_events(record.metric_events)
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
        task: str = DEFAULT_TASK,
        is_optuna: bool = True,
        config_source: str = "defaults",
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
        self.task = task
        self.is_optuna = is_optuna
        mlflow.set_tags(
            execution_tags(
                dataset_name=dataset_name,
                run_role="optuna_trial",
                run_type="optuna_trial",
                seed=seed,
                cv_folds=cv_folds,
                lr_scheduler=lr_scheduler,
                environment=environment,
                extra_tags=extra_tags,
                task=task,
                is_optuna=is_optuna,
            )
        )
        _log_execution_params(hyperparameters, dataset_name, seed, cv_folds, task, config_source)
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
        events = [event for event in record.metric_events if event.step is None]
        timings = fold_timings_for_tracking(record)
        if "total_seconds" in timings:
            events.append(LoggedMetric(TRAINING_SECONDS_KEY, timings["total_seconds"], None))
        _log_metric_events(events)


def execution_tags(
    *,
    dataset_name: str,
    run_role: str,
    run_type: str,
    seed: int,
    cv_folds: int | None,
    lr_scheduler: str,
    environment: Mapping[str, str] | None,
    extra_tags: Mapping[str, str] | None,
    task: str = DEFAULT_TASK,
    is_optuna: bool = False,
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
            # Which task the run trained, and whether a search produced it. Both are
            # dense -- a real value on every run kind -- so either can be filtered on
            # without a gap silently swallowing runs.
            TASK_TAG: task,
            IS_OPTUNA_TAG: _flag(is_optuna),
            # A run that trains is never a mirror; the mirror of it says "true" (ADR 0006).
            IS_MIRROR_TAG: _flag(False),
            # Hardware and library descriptors (device, gpu_name, ...) so
            # the ``time/`` metrics are only compared within one environment.
            **(environment or {}),
            **(extra_tags or {}),
        },
    )


def _flag(value: bool) -> str:
    """MLflow tag values are strings, so a boolean has to be spelled out."""
    return "true" if value else "false"


def _log_execution_params(
    hyperparameters: dict[str, object],
    dataset_name: str,
    seed: int,
    cv_folds: int | None,
    task: str = DEFAULT_TASK,
    config_source: str = "defaults",
) -> None:
    mlflow.log_params(hyperparameters)
    mlflow.log_param("dataset_name", dataset_name)
    mlflow.log_param("seed", seed)
    mlflow.log_param("task", task)
    # Dense: on comparison parents and Optuna trials alike, so tuned and default runs
    # are told apart without a new tag (ADR 0005, decision 5).
    mlflow.log_param("config_source", config_source)
    if cv_folds is not None:
        mlflow.log_param("cv_folds", cv_folds)


def _log_summary_metrics(summary: CrossValidationSummary) -> None:
    """Log the final CV statistics and timings; shared by parents and trials."""
    events = [
        LoggedMetric(key, value, None)
        for metric_name, statistics in summary.metrics.items()
        for key, value in _summary_metrics(f"cv/test/{metric_name}", statistics).items()
    ]
    # Which baseline the copied ``baseline/best`` statistics belong to (ADR 0007).
    if summary.best_baselines:
        mlflow.set_tags(
            {f"best_baseline/{population}": name for population, name in summary.best_baselines.items()}
        )

    events.extend(
        LoggedMetric(key, value, None)
        for timing_name, statistics in summary.timings.items()
        for key, value in _summary_metrics(f"cv/time/{timing_name}", statistics).items()
    )
    if "total_seconds" in summary.timings:
        # One mode-independent column: the same key a single-split parent
        # logs, here as the sum of every fold's total.
        total = summary.timings["total_seconds"]
        events.append(
            LoggedMetric(TRAINING_SECONDS_KEY, float(total.mean) * int(total.fold_count), None)
        )
    _log_metric_events(events)


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


class _MetricClock:
    """Hand out metric timestamps that only ever increase within this process.

    A flush claims one millisecond per metric, running ahead of the wall clock, so the next
    flush can start before the last one's final millisecond; the clock stepping back (NTP)
    does the same. Either would tie or reorder rows that ``get_metric_history`` sorts by
    timestamp first.
    """

    def __init__(self) -> None:
        self._last = -1

    def reserve(self, count: int) -> int:
        """Claim ``count`` consecutive milliseconds; return the first."""
        # MLflow's own clock, the one ``mlflow.log_metric`` stamps with; it is untyped.
        now = int(get_current_time_millis())  # type: ignore[no-untyped-call]
        start = max(now, self._last + 1)
        self._last = start + count - 1
        return start


_METRIC_CLOCK = _MetricClock()


def _log_metric_events(events: Sequence[LoggedMetric]) -> None:
    """Write metrics into the active run in the order given, a thousand per commit.

    ``mlflow.log_metric`` commits every value on its own, which on a hard disk cost about
    200 ms each. Each metric here takes its own millisecond from ``_METRIC_CLOCK``, so
    ``get_metric_history``, which sorts by timestamp first, returns them in logged order.
    The timestamps mark when the run was written, not the epoch that produced the value.
    """
    if not events:
        return
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("metrics are written into the active MLflow run, and none is open")
    start = _METRIC_CLOCK.reserve(len(events))
    metrics = [
        # A step-less metric is stored at step 0, as ``mlflow.log_metric`` stores it.
        metric(event.key, event.value, start + offset, event.step if event.step is not None else 0)
        for offset, event in enumerate(events)
    ]
    client = MlflowClient()
    for batch in chunks(metrics, METRICS_PER_BATCH):
        client.log_batch(run.info.run_id, metrics=batch)


def mirror_root_safely(run_id: str) -> None:
    """Mirror a finished root run's tree; a failure is a warning, never a failed training.

    The run stays intact in its family, unmirrored, and the script repairs it later.
    """
    try:
        mirror_run_tree(MlflowClient(), run_id)
    except Exception as error:  # noqa: BLE001 - anything here must not reach the trainer
        logger.warning(f"Run {run_id} was not mirrored (scripts/mirror_runs.py repairs it): {error}")


def create_tracker(enabled: bool, run_role: str = "parent", mirror_runs: bool = True):
    """Return an MLflow-backed tracker only when tracking has been requested."""
    if not enabled:
        return DisabledTracker()
    if run_role == "optuna_trial":
        # A trial is nested under its study, which mirrors the whole tree when it closes.
        return OptunaTrialTracker()
    if run_role == "parent":
        return MlflowTracker(mirror_runs=mirror_runs)
    raise ValueError(f"Unknown tracking run role {run_role!r}")
