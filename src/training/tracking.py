"""MLflow adapters for the training runtime."""

from contextlib import contextmanager, nullcontext
import datetime
from typing import Iterator

import mlflow

from src.mlflow_utils import (
    build_fold_tags,
    build_run_tags,
    get_or_create_experiment,
    setup_mlflow,
)


class DisabledTracker:
    """A tracker whose methods deliberately avoid importing MLflow side effects."""

    @contextmanager
    def parent_run(self, **kwargs) -> Iterator["DisabledTracker"]:
        yield self

    @contextmanager
    def fold_run(self, **kwargs) -> Iterator["DisabledTracker"]:
        yield self

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        return None

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        return None


class MlflowTracker:
    """Preserve the original parent-run and nested-fold MLflow layout."""

    def __init__(self) -> None:
        self.experiment_id: str | None = None

    @contextmanager
    def parent_run(
        self,
        *,
        dataset_name: str,
        seed: int,
        cv_folds: int | None,
        hyperparameters: dict[str, object],
    ) -> Iterator["MlflowTracker"]:
        mlflow.autolog()
        setup_mlflow()
        self.experiment_id = get_or_create_experiment(dataset_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tags = build_run_tags(dataset_name=dataset_name, run_type="train", seed=seed, cv_folds=cv_folds)
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
    ) -> Iterator["MlflowTracker"]:
        run_name = f"fold_{fold}_of_{cv_folds}" if cv_folds is not None else "single_split"
        tags = build_fold_tags(fold - 1, cv_folds)
        tags["dataset"] = dataset_name
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags,
            nested=True,
        ):
            yield self

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(path, artifact_path=artifact_path)


def create_tracker(enabled: bool):
    """Return an MLflow-backed tracker only when tracking has been requested."""
    return MlflowTracker() if enabled else DisabledTracker()
