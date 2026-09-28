"""Baseline imputers, scored on exactly the cells the decoder is scored on.

Three baseline imputers travel beside the model on every scored population: what filling
the column mean or mode would have got, and what a k-nearest-neighbours imputer would have
got at five and at ten neighbours. Each is fit on the training fold alone and scored
through ``score_cells`` with the same naive denominators, so each KNN baseline carries an
``impute_score`` on the model's own scale while the mean/mode baseline sits at 1.0 by
construction. The better of the two KNN baselines is the bar a comparison reads. See
ADR 0007.

Cells are hidden with a boolean mask rather than by writing NaN into the frame: an
integer-coded categorical column would be upcast to float on the way, and its categories
would then stringify as ``"1.0"`` where the scored truth says ``"1"``.
"""

from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.impute import KNNImputer

from src.embedder import as_category_strings

from .imputation_metrics import NUMERICAL, mean_mode_baselines, score_cells

# The counts are the model's own; repeating them under every baseline would only crowd
# the tracking store.
_COUNTS = frozenset({"n_num_cells", "n_cat_cells"})
# What ``as_category_strings`` renders a gap as, and the two tokens the model's frames
# carry: none of them is a category a baseline imputer may answer with.
MISSING_MARKER = "nan"
_NOT_A_CLASS = frozenset({MISSING_MARKER, "[MASK]", "[NULL]"})
# Two neighbourhood sizes, each its own baseline: five is the literature's default, and
# ten was the stronger bar on 20 of 27 variants once missingness grows (ADR 0007). A
# constant rather than a configuration key: a baseline is a fixed bar, not something a
# run tunes.
KNN_NEIGHBOURS = (5, 10)


class BaselineImputer(Protocol):
    """A baseline imputer: learns from the training fold, fills whatever is hidden."""

    name: str

    def fit(self, frame: pd.DataFrame) -> None: ...

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        """The frame with every hidden or missing cell filled, columns in the frame's order.

        ``hidden`` is one boolean per cell of ``frame``; a cell the frame is already
        missing counts as hidden whether or not the mask says so. Numerical columns come
        back as floats in the frame's own (scaled) space, categorical ones as the strings
        ``as_category_strings`` renders. The frame itself is never rewritten.
        """
        ...


def _position(frame: pd.DataFrame, column: str) -> int:
    """Where a named column sits, as the positional int a mask or matrix is indexed by."""
    position = frame.columns.get_loc(column)
    assert isinstance(position, int), f"feature names are unique, so '{column}' has one position"
    return position


class MeanModeImputer:
    """The naive baseline: each column's training mean or mode, whatever was hidden."""

    name: str = "mean_mode"

    def __init__(self, numerical_columns: Sequence[str], categorical_columns: Sequence[str]) -> None:
        self._numerical = list(numerical_columns)
        self._categorical = list(categorical_columns)
        self._fill: Mapping[str, float | str] = {}

    def fit(self, frame: pd.DataFrame) -> None:
        self._fill = mean_mode_baselines(frame, self._numerical, self._categorical)

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        missing = hidden | frame.isna().to_numpy()
        columns: dict[str, NDArray[np.float64] | NDArray[np.object_]] = {}
        for column in self._numerical:
            values = frame[column].to_numpy(dtype=float)
            columns[column] = np.where(missing[:, _position(frame, column)], float(self._fill[column]), values)
        for column in self._categorical:
            strings = as_category_strings(frame[column]).astype(object)
            columns[column] = np.where(missing[:, _position(frame, column)], str(self._fill[column]), strings)
        return pd.DataFrame(columns, index=frame.index).reindex(columns=frame.columns)


class KnnImputer:
    """The stronger baseline: each hidden cell filled from its nearest training rows.

    Numerical columns enter the distance as they are (already scaled); a categorical
    column enters as a one-hot block over the categories the training fold shows, the
    whole block blank where the cell is hidden, so that a hidden category neither pulls
    the distance nor votes. A filled block decodes to the category with the most support.
    """

    name: str

    def __init__(
        self,
        numerical_columns: Sequence[str],
        categorical_columns: Sequence[str],
        n_neighbors: int,
    ) -> None:
        # Named by its size, so each neighbourhood is its own baseline in the metrics.
        self.name = f"knn{n_neighbors}"
        self._numerical = list(numerical_columns)
        self._categorical = list(categorical_columns)
        self._n_neighbors = n_neighbors
        self._classes: dict[str, list[str]] = {}
        self._imputer: KNNImputer | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        # The vocabulary is fixed here, from the training fold alone, because it fixes the
        # width of the matrix. A category the training fold never shows can never be
        # answered, exactly as it never could be by a model fit on that fold.
        self._classes = {}
        for column in self._categorical:
            classes = sorted(set(as_category_strings(frame[column]).tolist()) - _NOT_A_CLASS)
            if not classes:
                raise ValueError(
                    f"column '{column}' has no observed category in the training fold, "
                    "so the KNN baseline has nothing to fill it with"
                )
            self._classes[column] = classes
        # An all-missing training column must keep its place, or every block after it
        # would shift and every fill would be read from the wrong column.
        self._imputer = KNNImputer(
            n_neighbors=self._n_neighbors, weights="uniform", keep_empty_features=True
        ).fit(self._matrix(frame, frame.isna().to_numpy()))

    def _matrix(self, frame: pd.DataFrame, missing: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Numerical columns, then one one-hot block per categorical column; NaN where hidden."""
        blocks: list[NDArray[np.float64]] = []
        for column in self._numerical:
            values = frame[column].to_numpy(dtype=float)
            blocks.append(np.where(missing[:, _position(frame, column)], np.nan, values)[:, None])
        for column in self._categorical:
            strings = as_category_strings(frame[column]).astype(object)
            classes = np.array(self._classes[column], dtype=object)
            block = (strings[:, None] == classes[None, :]).astype(float)
            block[missing[:, _position(frame, column)], :] = np.nan
            blocks.append(block)
        return np.hstack(blocks)

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        assert self._imputer is not None, "fit before impute"
        missing = hidden | frame.isna().to_numpy()
        filled = self._imputer.transform(self._matrix(frame, missing))
        columns: dict[str, NDArray[np.float64] | NDArray[np.object_]] = {}
        offset = 0
        for column in self._numerical:
            columns[column] = filled[:, offset]
            offset += 1
        for column in self._categorical:
            classes = np.array(self._classes[column], dtype=object)
            block = filled[:, offset : offset + len(classes)]
            # Ties, and a row whose block came back all zero, go to the first category.
            columns[column] = classes[block.argmax(axis=1)]
            offset += len(classes)
        return pd.DataFrame(columns, index=frame.index).reindex(columns=frame.columns)


def fit_baseline_imputers(
    train_frame: pd.DataFrame,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> list[BaselineImputer]:
    """The baseline imputers, fit on this fold's training split. Once per fold."""
    imputers: list[BaselineImputer] = [
        MeanModeImputer(numerical_columns, categorical_columns),
        *(
            KnnImputer(numerical_columns, categorical_columns, n_neighbors=size)
            for size in KNN_NEIGHBOURS
        ),
    ]
    for imputer in imputers:
        imputer.fit(train_frame)
    return imputers


def score_baselines(
    imputers: Sequence[BaselineImputer],
    frame: pd.DataFrame,
    cells: pd.DataFrame,
    baselines: Mapping[str, float | str],
) -> dict[str, float]:
    """Each baseline imputer's error on the scored cells, keyed ``baseline/<name>/<metric>``.

    The cells are hidden from the imputers exactly as they were hidden from the model, the
    fill is read back at the same (row, column) positions, and it is scored by the same
    ``score_cells`` with the same naive denominators. The scored table is left untouched:
    it goes on to the cell ledger and the per-column artifact.
    """
    if cells.empty:
        return {}
    rows = cells["row"].to_numpy(dtype=int)
    columns = frame.columns.get_indexer(pd.Index(cells["column"]))
    if (columns < 0).any():
        unknown = sorted(set(cells["column"].to_numpy()[columns < 0].tolist()))
        raise ValueError(f"scored cells name columns the frame does not have: {unknown}")
    hidden = np.zeros(frame.shape, dtype=bool)
    hidden[rows, columns] = True
    kinds = cells["kind"].to_numpy()

    metrics: dict[str, float] = {}
    for imputer in imputers:
        filled = imputer.impute(frame, hidden)
        if not filled.columns.equals(frame.columns):
            raise ValueError(f"{imputer.name} returned columns in a different order than the frame's")
        guesses = filled.to_numpy(dtype=object)[rows, columns]
        imputed = [
            float(guess) if kind == NUMERICAL else str(guess) for guess, kind in zip(guesses, kinds)
        ]
        scored = score_cells(cells.assign(imputed=imputed), baselines)
        metrics.update(
            {
                f"baseline/{imputer.name}/{name}": value
                for name, value in scored.metrics.items()
                if name not in _COUNTS
            }
        )
    return metrics
