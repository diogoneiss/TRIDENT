"""
src/mlflow_utils.py
-------------------
Centralised MLflow utilities for TRIDENT.

Responsibilities:
  - Configure the tracking URI (SQLite by default, overridable via the
    MLFLOW_TRACKING_URI environment variable).
  - Get-or-create an MLflow experiment named after the dataset.
  - Build flat param / tag dictionaries ready for mlflow.log_params / set_tags.

Usage example (in train.py):
    from src.mlflow_utils import setup_mlflow, get_or_create_experiment, build_run_tags

    setup_mlflow()                                   # call once at startup
    exp_id = get_or_create_experiment(dataset_name)  # one experiment per dataset
    with mlflow.start_run(experiment_id=exp_id, run_name="...", tags=build_run_tags(...)):
        mlflow.log_params(hyperparameters)
        ...
"""

import os
import re
import mlflow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

# Top-level experiment prefix so all TRIDENT experiments are grouped together
EXPERIMENT_PREFIX = "TRIDENT"
# Where the mirror experiments live (ADR 0006): one per task, ``TRIDENT/mirror/<task>``,
# holding a mirror run of every run in every experiment family.
MIRROR_EXPERIMENT_PREFIX = f"{EXPERIMENT_PREFIX}/mirror"

# Tag recording which learning-rate schedule a run trained with (ADR 0003).
# Runs created before the tag existed are backfilled by
# ``scripts/backfill_lr_scheduler_tag.py`` and additionally carry
# ``LR_SCHEDULER_BACKFILLED_TAG`` so provenance stays visible.
LR_SCHEDULER_TAG = "lr_scheduler"
LR_SCHEDULER_BACKFILLED_TAG = "lr_scheduler_backfilled"

# Which task a run trained (ADR 0004), and whether a hyper-parameter search produced it.
# Both are dense: every run kind carries a real value, so either can be filtered on
# without a gap swallowing runs. Runs recorded before the task existed are backfilled as
# "classification" and marked with TASK_BACKFILLED_TAG; is_optuna needs no such marker,
# because the store itself proves the value.
TASK_TAG = "task"
TASK_BACKFILLED_TAG = "task_backfilled"
IS_OPTUNA_TAG = "is_optuna"
# Which search-space profile a study sampled (ADR 0005). Sparse: only Optuna runs carry
# it, and the dense is_optuna tag already gates any filter on it, so no backfill.
SEARCH_SPACE_TAG = "search_space"
# Whether a run is a mirror run (ADR 0006). Dense: "true" on every mirror run, "false" on
# every other run, because a mirror copies ``run_role`` verbatim and a cross-experiment
# query could not tell the copy from its source otherwise. Backfilled as "false" onto
# runs recorded before the tag existed by ``scripts/mirror_runs.py``.
IS_MIRROR_TAG = "is_mirror"
# The run a mirror run copies; its unique key inside the mirror experiment.
SOURCE_RUN_ID_TAG = "source_run_id"


def parse_missingness_percent(dataset_name: str) -> str:
    """Return the normalized percentage encoded by a ``_<n>nan`` suffix."""
    match = re.search(r"_(\d+)nan$", dataset_name)
    return str(int(match.group(1))) if match is not None else "unknown"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_mlflow(tracking_uri: str | None = None) -> str:
    """
    Configure the MLflow tracking URI.

    Priority order:
        1. ``tracking_uri`` argument (explicit).
        2. ``MLFLOW_TRACKING_URI`` environment variable.
        3. ``sqlite:///mlflow.db`` at the current working directory.

    Returns the URI that was set.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)

    return uri


# ---------------------------------------------------------------------------
# Experiment management
# ---------------------------------------------------------------------------

def get_or_create_experiment(dataset_name: str) -> str:
    """
    Return the experiment_id for the given dataset, creating it if absent.

    Experiment name format: ``TRIDENT/<dataset_name>``
    (e.g. ``TRIDENT/vehicle``, ``TRIDENT/breast_cancer``).

    Parameters
    ----------
    dataset_name:
        The full dataset variant name (e.g. ``vehicle_20nan``).
        The *base* name (everything before the first ``_``) is used as the
        experiment sub-folder so that all NaN-percentage variants of the same
        dataset land in the same experiment.

    Returns
    -------
    str
        The MLflow experiment ID.
    """
    base_name = dataset_name.split("_")[0]
    experiment_name = f"{EXPERIMENT_PREFIX}/{base_name}"

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id

    experiment_id = mlflow.create_experiment(experiment_name)
    return experiment_id


def mirror_experiment_name(task: str) -> str:
    """The mirror experiment for a task, e.g. ``TRIDENT/mirror/imputation``."""
    return f"{MIRROR_EXPERIMENT_PREFIX}/{task}"


def is_mirror_experiment(experiment_name: str) -> bool:
    """Whether an experiment holds mirror runs, so it is never mirrored itself."""
    return experiment_name.startswith(f"{MIRROR_EXPERIMENT_PREFIX}/")


def get_or_create_mirror_experiment(task: str) -> str:
    """Return the experiment_id of the task's mirror experiment, creating it if absent.

    Separate from ``get_or_create_experiment``, whose ``split("_")[0]`` on a dataset name
    cannot produce ``TRIDENT/mirror/<task>``. The task is the literal value of the
    ``task`` tag, so the experiment name and the tag can never disagree.
    """
    experiment_name = mirror_experiment_name(task)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return str(experiment.experiment_id)
    return str(mlflow.create_experiment(experiment_name))


# ---------------------------------------------------------------------------
# Parameter / tag helpers
# ---------------------------------------------------------------------------

def build_hyperparams_dict(
    DIM, HIDDEN_DIM, HEADS, LAYERS, DIM_FEED, DROPOUT,
    EPOCHS_PRE, BATCH, LR_PRE, WEIGHT_DECAY_PRE, PROB_MASCARA,
    EPOCH_FINE, LR_FINE, WEIGHT_DECAY_FINE, LABELS,
) -> dict:
    """
    Build a flat dict of all TRIDENT hyperparameters suitable for
    ``mlflow.log_params``.  Keys use the canonical names from the JSON
    hyperparameter files so they are consistent across runs.
    """
    return {
        "DIM": DIM,
        "HIDDEN_DIM": HIDDEN_DIM,
        "HEADS": HEADS,
        "LAYERS": LAYERS,
        "DIM_FEED": DIM_FEED,
        "DROPOUT": DROPOUT,
        "EPOCHS_PRE": EPOCHS_PRE,
        "BATCH": BATCH,
        "LR_PRE": LR_PRE,
        "WEIGHT_DECAY_PRE": WEIGHT_DECAY_PRE,
        "PROB_MASCARA": PROB_MASCARA,
        "EPOCH_FINE": EPOCH_FINE,
        "LR_FINE": LR_FINE,
        "WEIGHT_DECAY_FINE": WEIGHT_DECAY_FINE,
        "LABELS": LABELS,
    }


def build_run_tags(
    dataset_name: str,
    run_type: str = "train",
    seed: int | None = None,
    cv_folds: int | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Build a flat dict of MLflow tags for a training run.

    Parameters
    ----------
    dataset_name:
        Full dataset variant name (e.g. ``vehicle_20nan``).
    run_type:
        One of ``"train"``, ``"optuna_study"``, ``"optuna_trial"``.
    seed:
        Random seed used for the run.
    cv_folds:
        Number of CV folds (``None`` means pre-defined single split).
    extra:
        Any additional key/value tags to merge in.

    Returns
    -------
    dict
        A flat string→string mapping suitable for ``mlflow.set_tags``.
    """
    tags: dict = {
        "dataset": dataset_name,
        "base_dataset": dataset_name.split("_")[0],
        "run_type": run_type,
        "cv_folds": str(cv_folds) if cv_folds is not None else "single_split",
    }
    if seed is not None:
        tags["seed"] = str(seed)
    if extra:
        tags.update({k: str(v) for k, v in extra.items()})
    return tags


def build_fold_tags(fold_idx: int, total_folds: int | None) -> dict:
    """
    Build tags for a per-fold child run.

    Parameters
    ----------
    fold_idx:
        Zero-based fold index.
    total_folds:
        Total number of folds, or ``None`` for a single-split run.
    """
    if total_folds is None:
        return {"fold": "single_split", "fold_index": "0"}
    return {
        "fold": str(fold_idx + 1),
        "fold_index": str(fold_idx),
        "total_folds": str(total_folds),
    }
