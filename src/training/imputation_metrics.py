"""Imputation error, and the score that ranks folds by it.

Pure functions over a table of scored cells, so they can be exercised without training
anything. See ADR 0004 decision 6.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

NUMERICAL = "numerical"
CATEGORICAL = "categorical"
# ``impute_score`` with the class-conditional mean/mode fill as its denominators
# (``class_conditional_score``). A run logs it, for itself and its baselines, only when it ranks by it.
CLASS_SCORE = "impute_score_classcond"


@dataclass(frozen=True)
class ImputationScores:
    """Everything one population of scored cells says about a fold.

    ``metrics`` are pooled over the fold and rank it; ``per_column`` is the same errors
    column by column, in long form, which says where a poor fold went wrong without
    crowding the tracking store on a table of fifty-seven columns.
    """

    metrics: dict[str, float]
    per_column: pd.DataFrame


def score_cells(cells: pd.DataFrame, baselines: Mapping[str, float | str]) -> ImputationScores:
    """Score a fold's reconstructed cells against their true values.

    ``cells`` carries one row per scored cell with its ``column``, its ``kind``, the
    ``actual`` value and the ``imputed`` one. Numbers are scored by how far off the guess
    was, categories by whether it was right. Each kind is pooled over its own cells, so
    the two never dilute each other.
    """
    numerical = cells[cells["kind"] == NUMERICAL]
    categorical = cells[cells["kind"] == CATEGORICAL]
    metrics = _error_metrics(numerical, categorical)
    # How much of the naive imputer's error is left, weighted by how many cells of each
    # kind were scored, so a table of one kind reduces to that kind's ratio alone.
    total = len(numerical) + len(categorical)
    if total:
        score = 0.0
        if len(numerical):
            naive = numerical["column"].map(baselines).to_numpy(dtype=float)
            naive_rmse = float(
                np.sqrt(np.mean((naive - numerical["actual"].to_numpy(dtype=float)) ** 2))
            )
            score += (len(numerical) / total) * _ratio(metrics["rmse_num_z"], naive_rmse)
        if len(categorical):
            naive = categorical["column"].map(baselines).to_numpy()
            naive_error = float(np.mean(naive != categorical["actual"].to_numpy()))
            score += (len(categorical) / total) * _ratio(1.0 - metrics["acc_cat"], naive_error)
        metrics["impute_score"] = score

    rows = [
        {"column": column, "metric": name, "value": value}
        for column, group in cells.groupby("column", sort=True)
        for name, value in _error_metrics(
            group[group["kind"] == NUMERICAL], group[group["kind"] == CATEGORICAL]
        ).items()
    ]
    return ImputationScores(
        metrics=metrics,
        per_column=pd.DataFrame(rows, columns=["column", "metric", "value"]),
    )


def _error_metrics(numerical: pd.DataFrame, categorical: pd.DataFrame) -> dict[str, float]:
    """How far off the guesses were, pooled within each kind of cell."""
    metrics: dict[str, float] = {
        "n_num_cells": float(len(numerical)),
        "n_cat_cells": float(len(categorical)),
    }
    if len(numerical):
        error = numerical["imputed"].to_numpy(dtype=float) - numerical["actual"].to_numpy(
            dtype=float
        )
        metrics["rmse_num_z"] = float(np.sqrt(np.mean(error**2)))
        metrics["mae_num_z"] = float(np.mean(np.abs(error)))
    if len(categorical):
        metrics["acc_cat"] = float(
            np.mean(categorical["imputed"].to_numpy() == categorical["actual"].to_numpy())
        )
        # Per column, then averaged: accuracy alone flatters a model that always answers
        # with the common category, and at high missingness a rare class can vanish from
        # a fold, so a class the model never reaches for scores zero.
        metrics["macro_f1_cat"] = float(
            np.mean(
                [
                    f1_score(
                        group["actual"].to_numpy(),
                        group["imputed"].to_numpy(),
                        average="macro",
                        zero_division=0,
                    )
                    for _, group in categorical.groupby("column", sort=True)
                ]
            )
        )
    return metrics


def _ratio(model_error: float, naive_error: float) -> float:
    """Model error as a share of the naive imputer's, guarding a flawless baseline.

    A constant column leaves the naive imputer nothing to get wrong, so the ratio would
    divide by zero. Matching a flawless baseline is parity rather than a win, and falling
    short of one is worse than parity by however much the model got wrong. Both stay
    finite and keep folds ordered by model error.
    """
    if naive_error == 0.0:
        return 1.0 + model_error
    return model_error / naive_error


def mean_mode_baselines(
    training_fold: pd.DataFrame,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Mapping[str, float | str]:
    """What a naive imputer would fill each column's cells with.

    The mean of a numerical column and the mode of a categorical one, learned from the
    training fold alone so that nothing about the scored cells leaks into their own
    baseline. Missing training cells are not values, so they neither pull the mean nor
    win the vote.
    """
    baselines: dict[str, float | str] = {
        column: float(training_fold[column].mean()) for column in numerical_columns
    }
    for column in categorical_columns:
        baselines[column] = str(training_fold[column].mode().iloc[0])
    return baselines


def class_mean_mode_baselines(
    training_fold: pd.DataFrame,
    classes: Sequence[str],
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> dict[str, Mapping[str, float | str]]:
    """``mean_mode_baselines`` once per class, from the training rows of that class alone.

    Class-conditional mean imputation: a cell of a row of class A is filled with the mean or the
    mode its column shows among the training rows of class A. ``classes`` holds the class of each
    row of ``training_fold``. A column a class never shows there keeps the whole fold's fill.
    """
    whole = mean_mode_baselines(training_fold, numerical_columns, categorical_columns)
    labels = np.asarray(classes)
    fills: dict[str, Mapping[str, float | str]] = {}
    for label in np.unique(labels):
        rows = training_fold[labels == label]
        shown = [column for column in rows.columns if bool(rows[column].notna().any())]
        fills[str(label)] = {
            **whole,
            **mean_mode_baselines(
                rows,
                [column for column in numerical_columns if column in shown],
                [column for column in categorical_columns if column in shown],
            ),
        }
    return fills


def class_naive_errors(
    cells: pd.DataFrame,
    classes: Sequence[str],
    class_baselines: Mapping[str, Mapping[str, float | str]],
    baselines: Mapping[str, float | str],
) -> tuple[float, float]:
    """The class-conditional fill's own errors on the scored cells: the RMSE of its numbers and
    the share of its categories that miss.

    ``classes`` holds the class of each row of the frame the cells' ``row`` points into. A class
    the training fold never shows falls back on the whole fold's fill.
    """
    labels = np.asarray(classes)[cells["row"].to_numpy(dtype=int)]
    naive = np.array(
        [class_baselines.get(str(label), baselines)[column] for label, column in zip(labels, cells["column"])],
        dtype=object,
    )
    actual = cells["actual"].to_numpy(dtype=object)
    numerical = (cells["kind"] == NUMERICAL).to_numpy()
    categorical = (cells["kind"] == CATEGORICAL).to_numpy()
    rmse = error = 0.0
    if numerical.any():
        rmse = float(np.sqrt(np.mean((naive[numerical].astype(float) - actual[numerical].astype(float)) ** 2)))
    if categorical.any():
        error = float(np.mean(naive[categorical] != actual[categorical]))
    return rmse, error


def class_conditional_score(
    errors: Mapping[str, float], n_numerical: int, n_categorical: int, naive_rmse: float, naive_error: float
) -> float:
    """``impute_score`` measured against the class-conditional fill (``CLASS_SCORE``).

    A method's own pooled errors (``rmse_num_z``, ``acc_cat``) over that fill's, weighted by the
    cells of each kind as ``score_cells`` weights them: the class-conditional fill scores 1. Only
    the denominators differ from ``impute_score``, so any method scored on the same cells can be
    put on this scale from its errors. The fill reads the row's class, which no model or baseline
    sees: it is a stricter bar, not a deployable imputer.
    """
    total = n_numerical + n_categorical
    score = 0.0
    if n_numerical:
        score += (n_numerical / total) * _ratio(errors["rmse_num_z"], naive_rmse)
    if n_categorical:
        score += (n_categorical / total) * _ratio(1.0 - errors["acc_cat"], naive_error)
    return score
