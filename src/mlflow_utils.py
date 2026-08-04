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
