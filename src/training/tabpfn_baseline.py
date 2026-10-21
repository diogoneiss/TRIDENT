"""TabPFN v2 (Hollmann et al., *Nature* 2025) as a column-wise baseline imputer (E55).

An external foundation-model baseline, offline only; no training path of TRIDENT reads it. The
package is the optional extra ``tabpfn`` (``uv sync --extra tabpfn``), pinned to 9.2.0, and is
imported only when an imputer is fitted, so the rest of the project never needs it.

- **The model:** TabPFN v2's own checkpoints, pinned through
  ``create_default_for_version(ModelVersion.V2)``. The package defaults to v3.5, whose weights are
  gated. The v2 weights (Prior Labs License 1.1: Apache 2.0 with an attribution requirement) are
  cached under ``TABPFN_MODEL_CACHE_DIR``.
- **Column-wise, inductive.**
  - Each column gets one estimator, fitted on the training fold's rows where that column is
    observed, with every other column as its features: a regressor for a number, a classifier
    for a category.
  - A categorical feature enters as its index in the training fold's vocabulary, declared to
    TabPFN as categorical. A gap enters as NaN, which TabPFN reads natively.
  - Hidden cells are NaN in the features too, so no fill ever feeds another.
- **The context is capped at ``CONTEXT_ROWS``,** v2's pre-training limit. A fixed-seed subsample
  of the observed rows is taken where a column has more: letter and electricity.
- **The fill is a point estimate:** the predictive mean for a number, the most probable class for
  a category.
- **Only the asked-for cells are predicted.** A cell that is missing but not hidden keeps the
  column's constant: no score reads it, and no fill enters a feature.
  - The first call on a frame also predicts the frame's own gaps, in the same forward pass. The
    induced population's call then reuses them, which saves a second pass over the context.
  - TabPFN's fills depend slightly on which test rows share a ``predict``. In a check, stacking
    rows moved one fill by up to 0.002, against a difference of 0 on a repeat. On
    electricity_20nan at seed 7, a fold's induced ``impute_score`` moved by at most 1.6e-4 against
    separate calls, far inside the 1% SESOI. The grouping is fixed, so a rerun gives the same
    fills.
- **Fixed settings:** 8 estimators (v2's default, named so that no wide-table rule raises it),
  ``random_state`` 0, the GPU when there is one.
- **Random streams:** everything runs inside ``torch.random.fork_rng``, with numpy's global
  stream saved and restored, so the streams every seeded result depends on are untouched (ADR
  0007 decision 6).
- **Constant fills:** a column with no observed value, one class, or no other column is filled
  with its mean or commonest category.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from src.embedder import as_category_strings

from .imputation_baselines import BaselineImputer

CONTEXT_ROWS = 10_000
N_ESTIMATORS = 8
TABPFN_SEED = 0
PREDICT_ROWS = 5_000
_NOT_A_CLASS = {"nan", "[MASK]", "[NULL]"}

Kind = Literal["regressor", "classifier"]
# Builds one unfitted estimator: its kind and the positions of its categorical features.
Factory = Callable[[Kind, list[int]], Any]


def tabpfn_v2_factory(device: str) -> Factory:
    """The real estimators, TabPFN v2's checkpoints, imported on first use."""
    tabpfn = importlib.import_module("tabpfn")
    version = importlib.import_module("tabpfn.constants").ModelVersion.V2

    def build(kind: Kind, categorical: list[int]) -> Any:
        estimator = tabpfn.TabPFNRegressor if kind == "regressor" else tabpfn.TabPFNClassifier
        return estimator.create_default_for_version(
            version,
            n_estimators=N_ESTIMATORS,
            categorical_features_indices=categorical or None,
            device=device,
            random_state=TABPFN_SEED,
        )

    return build


@dataclass
class _Column:
    estimator: Any | None
    constant: float | str
    features: list[int]


class TabPFNImputer:
    """One TabPFN v2 estimator a column; see the module docstring."""

    name = "tabpfn"

    def __init__(
        self,
        numerical_columns: Sequence[str],
        categorical_columns: Sequence[str],
        factory: Factory | None = None,
        context_rows: int = CONTEXT_ROWS,
    ) -> None:
        self._numerical = list(numerical_columns)
        self._categorical = list(categorical_columns)
        self._factory = factory
        self._context_rows = context_rows
        self._order: list[str] = []
        self._classes: dict[str, list[str]] = {}
        self._fitted: dict[str, _Column] = {}
        self._views: dict[tuple[int, tuple[int, int], int], dict[str, tuple[NDArray[np.int64], NDArray[Any]]]] = {}

    def _matrix(self, frame: pd.DataFrame, missing: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Every column as floats in ``self._order``: numbers as they are, categories as their
        training-fold index; a missing, hidden or unseen cell as NaN."""
        matrix = np.full((len(frame), len(self._order)), np.nan)
        for j, column in enumerate(self._order):
            gaps = missing[:, frame.columns.get_loc(column)]
            if column in self._classes:
                index = {label: float(k) for k, label in enumerate(self._classes[column])}
                values = np.array([index.get(label, np.nan) for label in as_category_strings(frame[column])])
            else:
                values = frame[column].to_numpy(dtype=float).copy()
            values[gaps] = np.nan
            matrix[:, j] = values
        return matrix

    def fit(self, frame: pd.DataFrame) -> None:
        self._order = [*self._numerical, *self._categorical]
        self._classes = {}
        for column in self._categorical:
            classes = sorted(set(as_category_strings(frame[column]).tolist()) - _NOT_A_CLASS)
            if not classes:
                raise ValueError(f"column '{column}' has no observed category in the training fold")
            self._classes[column] = classes
        factory = self._factory or tabpfn_v2_factory("cuda" if torch.cuda.is_available() else "cpu")
        matrix = self._matrix(frame, frame.isna().to_numpy())
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
                self._fitted = {column: self._fit_column(matrix, j, factory) for j, column in enumerate(self._order)}
        finally:
            np.random.set_state(numpy_state)

    def _fit_column(self, matrix: NDArray[np.float64], j: int, factory: Factory) -> _Column:
        column = self._order[j]
        categorical = column in self._classes
        observed = np.flatnonzero(~np.isnan(matrix[:, j]))
        target = matrix[observed, j]
        if categorical:
            codes, counts = np.unique(target.astype(np.int64), return_counts=True)
            constant: float | str = self._classes[column][int(codes[np.argmax(counts)])] if len(codes) else self._classes[column][0]
        else:
            constant = float(target.mean()) if len(target) else 0.0
        features = [k for k in range(len(self._order)) if k != j]
        if not len(observed) or not features or (categorical and len(np.unique(target)) < 2):
            return _Column(None, constant, features)
        if len(observed) > self._context_rows:
            chosen = np.random.default_rng(TABPFN_SEED + j).choice(observed, self._context_rows, replace=False)
            observed = np.sort(chosen)
        positions = [p for p, k in enumerate(features) if self._order[k] in self._classes]
        estimator = factory("classifier" if categorical else "regressor", positions)
        y = matrix[observed, j]
        estimator.fit(matrix[observed][:, features], y.astype(np.int64) if categorical else y)
        return _Column(estimator, constant, features)

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        assert self._fitted, "fit before impute"
        gaps = frame.isna().to_numpy()
        missing = hidden | gaps
        key = self._key(frame, missing)
        if key not in self._views:
            # One forward pass per column serves this call and the frame's own gaps: the
            # induced population's view, which a fold scores right after its masked one.
            views = {key: (missing, hidden)}
            views.setdefault(self._key(frame, gaps), (gaps, gaps))
            self._views = self._predict(frame, views)
        predicted = self._views[key]
        columns: dict[str, NDArray[np.float64] | NDArray[np.object_]] = {}
        for column in frame.columns:
            name = str(column)
            fitted = self._fitted[name]
            values: NDArray[np.float64] | NDArray[np.object_] = (
                as_category_strings(frame[name]).astype(object) if name in self._classes else frame[name].to_numpy(dtype=float).copy()
            )
            values[missing[:, frame.columns.get_loc(name)]] = fitted.constant
            rows, guesses = predicted.get(name, (np.empty(0, dtype=np.int64), np.empty(0)))
            if len(rows):
                values[rows] = guesses
            columns[name] = values
        return pd.DataFrame(columns, index=frame.index).reindex(columns=frame.columns)

    @staticmethod
    def _key(frame: pd.DataFrame, missing: NDArray[np.bool_]) -> tuple[int, tuple[int, int], int]:
        return id(frame), (int(frame.shape[0]), int(frame.shape[1])), hash(missing.tobytes())

    def _predict(
        self, frame: pd.DataFrame, views: dict[tuple[int, tuple[int, int], int], tuple[NDArray[np.bool_], NDArray[np.bool_]]]
    ) -> dict[tuple[int, tuple[int, int], int], dict[str, tuple[NDArray[np.int64], NDArray[Any]]]]:
        """Per view (what is missing, what is asked for), each column's asked rows and fills.

        Every view's rows of a column go through its estimator in one ``predict``, which saves a
        pass over the context. Stacking moves a fill slightly (the module docstring). A gap a
        view does not ask for keeps the column's constant: no score reads it, and no fill ever
        enters a feature.
        """
        matrices = {key: self._matrix(frame, missing) for key, (missing, _) in views.items()}
        out: dict[tuple[int, tuple[int, int], int], dict[str, tuple[NDArray[np.int64], NDArray[Any]]]] = {key: {} for key in views}
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
                for column, fitted in self._fitted.items():
                    if fitted.estimator is None:
                        continue
                    position = frame.columns.get_loc(column)
                    asked = {key: np.flatnonzero(views[key][1][:, position]) for key in views}
                    stacked = np.concatenate([matrices[key][rows][:, fitted.features] for key, rows in asked.items()])
                    if not len(stacked):
                        continue
                    guesses = np.concatenate([
                        np.asarray(fitted.estimator.predict(stacked[start:start + PREDICT_ROWS]))
                        for start in range(0, len(stacked), PREDICT_ROWS)
                    ])
                    if column in self._classes:
                        fills: NDArray[Any] = np.array(self._classes[column], dtype=object)[guesses.astype(np.int64)]
                    else:
                        fills = guesses.astype(float)
                    start = 0
                    for key, rows in asked.items():
                        out[key][column] = (rows, fills[start:start + len(rows)])
                        start += len(rows)
        finally:
            np.random.set_state(numpy_state)
        return out


def fit_tabpfn_imputers(
    train_frame: pd.DataFrame,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> list[BaselineImputer]:
    """TabPFN v2, fit on this fold's training split. Once per fold."""
    imputer = TabPFNImputer(numerical_columns, categorical_columns)
    imputer.fit(train_frame)
    return [imputer]
