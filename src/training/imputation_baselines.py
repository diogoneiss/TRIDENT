"""Baseline imputers, scored on exactly the cells the decoder is scored on.

Four baseline imputers travel beside the model on every scored population: what filling
the column mean or mode would have got, what a k-nearest-neighbours imputer would have got
at five and at ten neighbours, and what one gradient-boosting model per column would have
got. Each is fit on the training fold alone and scored through ``score_cells`` with the
same naive denominators, so each carries an ``impute_score`` on the model's own scale
while the mean/mode baseline sits at 1.0 by construction. The best of the three learned
baselines is the bar a comparison reads. See ADR 0007.

Cells are hidden with a boolean mask rather than by writing NaN into the frame: an
integer-coded categorical column would be upcast to float on the way, and its categories
would then stringify as ``"1.0"`` where the scored truth says ``"1"``.
"""

from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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
# The gradient-boosting baseline's own seed. Boosting draws a validation split on large
# folds; an integer seed gives that draw a stream of its own, so the fill is the same on
# every run and the global streams every seeded result depends on are never touched.
BOOSTING_SEED = 0
# The most categories histogram boosting takes as one categorical feature; a column with
# more enters as its vocabulary index, read as a number.
_MAX_BOOSTING_CATEGORIES = 255
# Each column's model also trains on a copy of its rows with this share of the other
# cells hidden, the evaluation mask's default rate (ADR 0004). A booster that never saw a
# feature missing sends every gap in it down whichever branch held more rows, which on a
# complete training fold means the rule it learned is simply not applied.
_TRAINING_HIDE_RATE = 0.2


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


class _Constant:
    """A column with nothing to learn from: every fill is the same value."""

    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(len(features), self._value)


_ColumnModel = HistGradientBoostingRegressor | HistGradientBoostingClassifier | _Constant


class GradientBoostingImputer:
    """A learned baseline: one gradient-boosting model per column, in a single pass.

    Each column's model is trained on the training rows where that column is observed,
    with every other column as its features, and fills a hidden cell from the rest of its
    row. Hidden and missing cells enter as gaps, which histogram boosting routes natively,
    so no fill ever feeds another (missForest would iterate; this does not). To learn
    where a gap should go, each model also trains on a copy of its rows with a fifth of
    the feature cells hidden, drawn from a stream of its own. A categorical column enters
    as its index in the training fold's vocabulary, declared categorical to the booster,
    and a categorical target is predicted as that index. A feature with no observed value
    is left out. A column with no observed value, a single observed value, or nothing
    left to learn from is filled with a constant: zero (the scaled mean), that value, or
    its mean or commonest category.
    """

    name: str = "hgb"

    def __init__(self, numerical_columns: Sequence[str], categorical_columns: Sequence[str]) -> None:
        self._numerical = list(numerical_columns)
        self._categorical = list(categorical_columns)
        self._classes: dict[str, list[str]] = {}
        # Per column, its model and the positions of the matrix columns it reads.
        self._models: dict[str, tuple[_ColumnModel, NDArray[np.intp]]] = {}

    def fit(self, frame: pd.DataFrame) -> None:
        self._classes = {}
        for column in self._categorical:
            classes = sorted(set(as_category_strings(frame[column]).tolist()) - _NOT_A_CLASS)
            if not classes:
                raise ValueError(
                    f"column '{column}' has no observed category in the training fold, "
                    "so the gradient-boosting baseline has nothing to fill it with"
                )
            self._classes[column] = classes
        matrix = self._matrix(frame, frame.isna().to_numpy())
        declared = np.array(
            [False] * len(self._numerical)
            + [len(self._classes[column]) <= _MAX_BOOSTING_CATEGORIES for column in self._categorical],
            dtype=bool,
        )
        stream = np.random.default_rng(BOOSTING_SEED)
        self._models = {}
        for index, column in enumerate(self._numerical + self._categorical):
            observed = ~np.isnan(matrix[:, index])
            others = np.delete(np.arange(matrix.shape[1]), index)
            features = matrix[np.ix_(observed, others)]
            target = matrix[observed, index]
            hidden = features.copy()
            hidden[stream.random(hidden.shape) < _TRAINING_HIDE_RATE] = np.nan
            features = np.vstack([features, hidden])
            target = np.concatenate([target, target])
            kept = ~np.isnan(features).all(axis=0) if len(features) else np.zeros(len(others), dtype=bool)
            model = self._fit_one(
                numerical=column in self._numerical,
                features=features[:, kept],
                target=target,
                categorical=declared[others][kept],
            )
            self._models[column] = (model, others[kept])

    @staticmethod
    def _fit_one(
        numerical: bool,
        features: NDArray[np.float64],
        target: NDArray[np.float64],
        categorical: NDArray[np.bool_],
    ) -> _ColumnModel:
        if len(target) == 0:
            return _Constant(0.0)
        values, counts = np.unique(target, return_counts=True)
        if len(values) == 1:
            return _Constant(float(values[0]))
        if features.shape[1] == 0:
            return _Constant(float(target.mean()) if numerical else float(values[counts.argmax()]))
        declared = categorical if categorical.any() else None
        if numerical:
            return HistGradientBoostingRegressor(
                categorical_features=declared, random_state=BOOSTING_SEED
            ).fit(features, target)
        return HistGradientBoostingClassifier(
            categorical_features=declared, random_state=BOOSTING_SEED
        ).fit(features, target.astype(int))

    def _matrix(self, frame: pd.DataFrame, missing: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Numerical columns, then categorical ones as vocabulary indices; NaN where hidden,
        missing, or a category the training fold never showed."""
        blocks: list[NDArray[np.float64]] = []
        for column in self._numerical:
            values = frame[column].to_numpy(dtype=float)
            blocks.append(np.where(missing[:, _position(frame, column)], np.nan, values))
        for column in self._categorical:
            lookup = {category: float(code) for code, category in enumerate(self._classes[column])}
            codes = np.array(
                [lookup.get(value, np.nan) for value in as_category_strings(frame[column]).tolist()],
                dtype=float,
            )
            blocks.append(np.where(missing[:, _position(frame, column)], np.nan, codes))
        if not blocks:
            return np.empty((len(frame), 0))
        return np.column_stack(blocks)

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        missing = hidden | frame.isna().to_numpy()
        matrix = self._matrix(frame, missing)
        columns: dict[str, NDArray[np.float64] | NDArray[np.object_]] = {}
        for column in self._numerical + self._categorical:
            gaps = missing[:, _position(frame, column)]
            model, reads = self._models[column]
            features = matrix[np.ix_(gaps, reads)]
            if column in self._numerical:
                numbers = frame[column].to_numpy(dtype=float).copy()
                if gaps.any():
                    numbers[gaps] = model.predict(features)
                columns[column] = numbers
                continue
            strings = as_category_strings(frame[column]).astype(object)
            if gaps.any():
                codes = np.asarray(model.predict(features)).astype(int)
                strings[gaps] = np.array(self._classes[column], dtype=object)[codes]
            columns[column] = strings
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
        GradientBoostingImputer(numerical_columns, categorical_columns),
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
