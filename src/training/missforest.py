"""missForest baseline imputers: one model per column, iterated until the fills settle.

Stekhoven and Bühlmann's missForest (Bioinformatics, 2012) fills every gap with its
column's mean or mode, then, column by column in order of increasing missingness, fits a
model of that column on its observed rows with every other column as features and
predicts its gaps, and repeats the round until the fills stop settling. Two learners:
random forests with the R package's defaults (``missforest``), and LightGBM with its own
defaults (``missforest_lgbm``). See ADR 0016.
"""

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.embedder import as_category_strings

from .imputation_baselines import _NOT_A_CLASS, BaselineImputer, _Constant, _position


@dataclass(frozen=True)
class RoundChange:
    """How much one round moved the fills, per kind of column; None where that kind had
    no gap to fill.

    ``numerical`` is the paper's Δ_N, the squared change of the numerical fills over their
    squared size; ``categorical`` is Δ_F, the share of categorical fills that changed.
    """

    numerical: float | None
    categorical: float | None


def still_converging(previous: RoundChange, latest: RoundChange) -> bool:
    """Whether another round is worth running: the change of some kind of column shrank.

    The first round in which neither shrank ends the iteration, and the fills of the round
    before it are the answer (``stopCriterion`` in the R package).
    """
    pairs = [
        (previous.numerical, latest.numerical),
        (previous.categorical, latest.categorical),
    ]
    return any(
        before is not None and after is not None and after < before for before, after in pairs
    )


# The most rounds either fit or fill runs; the paper's and the R package's ``maxiter``.
MAX_ROUNDS = 10
# Every model's own seed: a forest draws its bootstraps and split candidates from it, so
# two fits of one fold fill alike and the global streams every seeded result depends on
# are never touched. LightGBM's defaults draw nothing, but it gets the same seed.
MISSFOREST_SEED = 0
# The forest of the R package: 100 trees; a regression split weighs a third of the
# features and stops at five rows a leaf; a classification split weighs their square
# root and grows to single rows. scikit-learn's own regressor defaults (every feature,
# one row a leaf) would make a different, and on this protocol stronger-fitting, forest.
TREES = 100
_REGRESSION_FEATURES = 1 / 3
_REGRESSION_LEAF = 5
# Threads per model. Neither learner's result depends on it (each tree is seeded from
# the forest's seed; LightGBM runs ``deterministic`` and row-wise), so it only weighs
# speed against the cores a training run shares with it.
THREADS = 8

Learner = Literal["random_forest", "lightgbm"]
_NAMES: dict[str, str] = {"random_forest": "missforest", "lightgbm": "missforest_lgbm"}
_ColumnModel = (
    RandomForestRegressor | RandomForestClassifier | LGBMRegressor | LGBMClassifier | _Constant
)


class MissForestImputer:
    """missForest, fit on the training fold and replayed on whatever it is asked to fill.

    The fit is the paper's algorithm on the training fold: every gap starts at its
    column's mean or mode, then each round visits the columns in order of increasing
    missingness, fits each column's model on the rows where it is observed with every
    other column's current fills as features, and writes its predictions into that
    column's gaps before the next column is visited. The rounds stop by
    ``still_converging``, and the models of the round whose fills were kept are the
    ones kept. Every column gets a model, gaps or not: any of them can be hidden on the
    test fold. A fill starts the hidden and missing cells at the training means and
    modes and runs the same rounds with those models, under the same rule.

    Categories enter as their index in the training fold's vocabulary: numbers to the
    forest, declared categorical to LightGBM. A category the training fold never shows
    is a gap. A column with no observed value, a single one, or nothing to learn from is
    filled with a constant, as the gradient-boosting baseline does.
    """

    name: str

    def __init__(
        self, numerical_columns: Sequence[str], categorical_columns: Sequence[str], learner: Learner
    ) -> None:
        self.name = _NAMES[learner]
        self._learner = learner
        self._numerical = list(numerical_columns)
        self._categorical = list(categorical_columns)
        self._classes: dict[str, list[str]] = {}
        # Per matrix column: the fill every gap starts from, and its model.
        self._start: NDArray[np.float64] = np.empty(0)
        self._models: list[_ColumnModel] = []
        # The matrix columns in the order each round visits them.
        self._order: list[int] = []
        self.rounds = 0

    def fit(self, frame: pd.DataFrame) -> None:
        self._classes = {}
        self.rounds = 0
        for column in self._categorical:
            classes = sorted(set(as_category_strings(frame[column]).tolist()) - _NOT_A_CLASS)
            if not classes:
                raise ValueError(
                    f"column '{column}' has no observed category in the training fold, "
                    f"so the {self.name} baseline has nothing to fill it with"
                )
            self._classes[column] = classes
        matrix = self._matrix(frame, frame.isna().to_numpy())
        gaps = np.isnan(matrix)
        self._start = np.array(
            [self._first_guess(matrix[~gaps[:, index], index], index) for index in range(matrix.shape[1])]
        )
        # Stable, so columns missing equally often keep the frame's order.
        self._order = np.argsort(gaps.sum(axis=0), kind="stable").tolist()

        fills = np.where(gaps, self._start, matrix)
        kept: list[_ColumnModel] = []
        previous: RoundChange | None = None
        for _ in range(MAX_ROUNDS):
            models: list[_ColumnModel] = [_Constant(0.0)] * matrix.shape[1]
            updated = fills.copy()
            for index in self._order:
                observed = ~gaps[:, index]
                models[index] = self._fit_one(
                    index, updated[observed][:, self._others(index)], matrix[observed, index]
                )
                if gaps[:, index].any():
                    updated[gaps[:, index], index] = models[index].predict(
                        updated[gaps[:, index]][:, self._others(index)]
                    )
            change = self._change(fills, updated, gaps)
            if previous is not None and not still_converging(previous, change):
                break
            kept, fills, previous = models, updated, change
            self.rounds += 1
        self._models = kept

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        missing = hidden | frame.isna().to_numpy()
        matrix = self._matrix(frame, missing)
        gaps = np.isnan(matrix)
        fills = np.where(gaps, self._start, matrix)
        previous: RoundChange | None = None
        for _ in range(MAX_ROUNDS):
            updated = fills.copy()
            for index in self._order:
                if gaps[:, index].any():
                    updated[gaps[:, index], index] = self._models[index].predict(
                        updated[gaps[:, index]][:, self._others(index)]
                    )
            change = self._change(fills, updated, gaps)
            if previous is not None and not still_converging(previous, change):
                break
            fills, previous = updated, change

        columns: dict[str, NDArray[np.float64] | NDArray[np.object_]] = {}
        for index, column in enumerate(self._numerical):
            numbers = frame[column].to_numpy(dtype=float).copy()
            at = missing[:, _position(frame, column)]
            numbers[at] = fills[at, index]
            columns[column] = numbers
        for offset, column in enumerate(self._categorical):
            index = len(self._numerical) + offset
            strings = as_category_strings(frame[column]).astype(object)
            at = missing[:, _position(frame, column)]
            codes = fills[at, index].astype(int)
            strings[at] = np.array(self._classes[column], dtype=object)[codes]
            columns[column] = strings
        return pd.DataFrame(columns, index=frame.index).reindex(columns=frame.columns)

    def _others(self, index: int) -> NDArray[np.intp]:
        return np.delete(np.arange(len(self._numerical) + len(self._categorical)), index)

    def _is_numerical(self, index: int) -> bool:
        return index < len(self._numerical)

    def _first_guess(self, observed: NDArray[np.float64], index: int) -> float:
        """The column's training mean, or its commonest category's index (ties to the
        first in the vocabulary); zero, the scaled mean, where nothing is observed."""
        if len(observed) == 0:
            return 0.0
        if self._is_numerical(index):
            return float(observed.mean())
        values, counts = np.unique(observed, return_counts=True)
        return float(values[counts.argmax()])

    def _fit_one(
        self, index: int, features: NDArray[np.float64], target: NDArray[np.float64]
    ) -> _ColumnModel:
        numerical = self._is_numerical(index)
        if len(target) == 0:
            return _Constant(self._start[index])
        values, counts = np.unique(target, return_counts=True)
        if len(values) == 1:
            return _Constant(float(values[0]))
        if features.shape[1] == 0:
            return _Constant(float(target.mean()) if numerical else float(values[counts.argmax()]))
        if self._learner == "random_forest":
            if numerical:
                return RandomForestRegressor(
                    n_estimators=TREES,
                    max_features=_REGRESSION_FEATURES,
                    min_samples_leaf=_REGRESSION_LEAF,
                    random_state=MISSFOREST_SEED,
                    n_jobs=THREADS,
                ).fit(features, target)
            return RandomForestClassifier(
                n_estimators=TREES,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=MISSFOREST_SEED,
                n_jobs=THREADS,
            ).fit(features, target.astype(int))
        declared = [
            position
            for position, other in enumerate(self._others(index).tolist())
            if not self._is_numerical(other)
        ]
        # ``deterministic`` with a fixed histogram layout is what makes the fill the same at
        # any thread count; LightGBM documents the pair.
        if numerical:
            return LGBMRegressor(
                random_state=MISSFOREST_SEED, n_jobs=THREADS, deterministic=True, force_row_wise=True, verbose=-1
            ).fit(features, target, categorical_feature=declared)
        return LGBMClassifier(
            random_state=MISSFOREST_SEED, n_jobs=THREADS, deterministic=True, force_row_wise=True, verbose=-1
        ).fit(features, target.astype(int), categorical_feature=declared)

    def _change(
        self, before: NDArray[np.float64], after: NDArray[np.float64], gaps: NDArray[np.bool_]
    ) -> RoundChange:
        """The paper's Δ_N and Δ_F between two rounds' fills; None for a kind with no gap."""
        split = len(self._numerical)
        numerical: float | None = None
        if gaps[:, :split].any():
            moved = float(((after[:, :split] - before[:, :split]) ** 2).sum())
            size = float((after[:, :split] ** 2).sum())
            numerical = moved / size if size > 0 else (0.0 if moved == 0 else float("inf"))
        categorical: float | None = None
        if gaps[:, split:].any():
            changed = int((after[:, split:] != before[:, split:]).sum())
            categorical = changed / after[:, split:].size
        return RoundChange(numerical, categorical)

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


def fit_missforest_imputers(
    train_frame: pd.DataFrame,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> list[BaselineImputer]:
    """The two missForest baselines, fit on this fold's training split. Once per fold."""
    imputers: list[BaselineImputer] = [
        MissForestImputer(numerical_columns, categorical_columns, learner)
        for learner in ("random_forest", "lightgbm")
    ]
    for imputer in imputers:
        imputer.fit(train_frame)
    return imputers
