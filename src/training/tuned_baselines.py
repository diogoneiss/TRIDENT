"""The baselines with a tuning budget (E42; ticket imputation-protocol/02).

Each family of the bar, and missForest, gets a declared grid of settings; a study chooses one per
fold on the validation rows and scores only that one on the test split. This module only builds
the imputers of the grid, as subclasses that change nothing but the learner's settings. The ADR
0007 module and its cache key are untouched, and nothing here runs in a live training run.

The grids (fixed in the ticket before anything was scored), each family's current setting
included:

- knn: k in {5, 10, 20, 50};
- hgb: learning rate {0.05, 0.1} x iterations {100, 200, 400} x depth {3, unlimited}. Early stopping
  follows scikit-learn's ``auto`` rule (on above 10,000 training rows, the hidden copy included),
  but its validation split is drawn by original row, so a row and its hidden copy never sit on
  both sides of it;
- lgbm: learning rate {0.05, 0.1} x rounds {100, 200, 400} x leaves {7, 31};
- missforest: features per split {the R defaults, a half} x trees {100, 300}.

``shrink`` pulls numerical fills toward the training mean by a share alpha, the grid's last axis at
60% missing and above.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import cast

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import NDArray
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from . import lightgbm_baseline, missforest
from .imputation_baselines import (
    BOOSTING_SEED,
    BaselineImputer,
    GradientBoostingImputer,
    KnnImputer,
    _ColumnModel,
    _Constant,
)
from .lightgbm_baseline import LightGbmImputer
from .missforest import MISSFOREST_SEED, MissForestImputer

# scikit-learn's ``early_stopping="auto"`` turns it on above this many training rows.
_AUTO_EARLY_STOPPING_ROWS = 10_000
_EARLY_STOPPING_SHARE = 0.1


@dataclass(frozen=True)
class Setting:
    """One point of a family's grid, named so a study can log the choice."""

    family: str
    label: str
    values: Mapping[str, float | int | str | None]


def _constant_or_none(numerical: bool, features: NDArray[np.float64], target: NDArray[np.float64]) -> _Constant | None:
    if len(target) == 0:
        return _Constant(0.0)
    values, counts = np.unique(target, return_counts=True)
    if len(values) == 1:
        return _Constant(float(values[0]))
    if features.shape[1] == 0:
        return _Constant(float(target.mean()) if numerical else float(values[counts.argmax()]))
    return None


def by_original_row(rows: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Training and validation positions in a stack of ``rows`` original rows followed by their
    hidden copies: a tenth of the original rows, drawn from the booster's own seed, and their
    copies go to validation; no row sits on both sides."""
    stream = np.random.default_rng(BOOSTING_SEED)
    held = np.sort(stream.choice(rows, size=max(1, int(round(_EARLY_STOPPING_SHARE * rows))), replace=False))
    kept = np.setdiff1d(np.arange(rows), held)
    return np.concatenate([kept, kept + rows]), np.concatenate([held, held + rows])


class TunedGradientBoostingImputer(GradientBoostingImputer):
    """``hgb`` at one point of its grid."""

    def __init__(
        self, numerical_columns: Sequence[str], categorical_columns: Sequence[str],
        learning_rate: float, max_iter: int, max_depth: int | None,
    ) -> None:
        super().__init__(numerical_columns, categorical_columns)
        self._settings = (learning_rate, max_iter, max_depth)
        self.name = "hgb_tuned"

    def _fit_one(  # type: ignore[override]
        self, numerical: bool, features: NDArray[np.float64], target: NDArray[np.float64], categorical: NDArray[np.bool_]
    ) -> _ColumnModel:
        constant = _constant_or_none(numerical, features, target)
        if constant is not None:
            return constant
        learning_rate, max_iter, max_depth = self._settings
        declared = categorical if categorical.any() else None
        early = len(target) > _AUTO_EARLY_STOPPING_ROWS
        model: HistGradientBoostingRegressor | HistGradientBoostingClassifier
        if numerical:
            model = HistGradientBoostingRegressor(
                learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth,
                categorical_features=declared, early_stopping=early, random_state=BOOSTING_SEED,
            )
        else:
            target = target.astype(int)
            model = HistGradientBoostingClassifier(
                learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth,
                categorical_features=declared, early_stopping=early, random_state=BOOSTING_SEED,
            )
        if not early:
            return model.fit(features, target)
        # The parent stacks every observed row, then its hidden copy, in that order.
        train, held = by_original_row(len(target) // 2)
        return model.fit(features[train], target[train], X_val=features[held], y_val=target[held])


class TunedLightGbmImputer(LightGbmImputer):
    """``lgbm`` at one point of its grid."""

    def __init__(
        self, numerical_columns: Sequence[str], categorical_columns: Sequence[str],
        learning_rate: float, n_estimators: int, num_leaves: int,
    ) -> None:
        super().__init__(numerical_columns, categorical_columns)
        self._settings = (learning_rate, n_estimators, num_leaves)
        self.name = "lgbm_tuned"

    def _fit_one(  # type: ignore[override]
        self, numerical: bool, features: NDArray[np.float64], target: NDArray[np.float64], categorical: NDArray[np.bool_]
    ) -> _ColumnModel:
        constant = _constant_or_none(numerical, features, target)
        if constant is not None:
            return constant
        learning_rate, n_estimators, num_leaves = self._settings
        declared = np.flatnonzero(categorical).tolist()
        model: LGBMRegressor | LGBMClassifier
        if numerical:
            model = LGBMRegressor(
                learning_rate=learning_rate, n_estimators=n_estimators, num_leaves=num_leaves,
                random_state=BOOSTING_SEED, n_jobs=lightgbm_baseline.THREADS, deterministic=True,
                force_row_wise=True, verbose=-1,
            ).fit(features, target, categorical_feature=declared)
        else:
            model = LGBMClassifier(
                learning_rate=learning_rate, n_estimators=n_estimators, num_leaves=num_leaves,
                random_state=BOOSTING_SEED, n_jobs=lightgbm_baseline.THREADS, deterministic=True,
                force_row_wise=True, verbose=-1,
            ).fit(features, target.astype(int), categorical_feature=declared)
        return cast(_ColumnModel, model)


class TunedMissForestImputer(MissForestImputer):
    """missForest at one point of its grid: features per split and trees."""

    def __init__(
        self, numerical_columns: Sequence[str], categorical_columns: Sequence[str], half_features: bool, trees: int
    ) -> None:
        super().__init__(numerical_columns, categorical_columns)
        self._half, self._trees = half_features, trees
        self.name = "missforest_tuned"

    def _fit_one(
        self, index: int, features: NDArray[np.float64], target: NDArray[np.float64]
    ) -> missforest._ColumnModel:
        constant = _constant_or_none(self._is_numerical(index), features, target)
        if constant is not None:
            return constant if len(target) else _Constant(self._start[index])
        forest: RandomForestRegressor | RandomForestClassifier
        if self._is_numerical(index):
            forest = RandomForestRegressor(
                n_estimators=self._trees, max_features=0.5 if self._half else 1 / 3, min_samples_leaf=5,
                random_state=MISSFOREST_SEED, n_jobs=missforest.THREADS,
            ).fit(features, target)
        else:
            forest = RandomForestClassifier(
                n_estimators=self._trees, max_features=0.5 if self._half else "sqrt", min_samples_leaf=1,
                random_state=MISSFOREST_SEED, n_jobs=missforest.THREADS,
            ).fit(features, target.astype(int))
        forest.n_jobs = 1
        return forest


def grid(family: str) -> list[Setting]:
    """The family's settings, its current one first (a tie on the validation rows keeps it)."""
    if family == "knn":
        return [Setting("knn", f"k{k}", {"k": k}) for k in (5, 10, 20, 50)]
    if family == "hgb":
        depths: list[tuple[float, int, int | None]] = [(0.1, 100, None)]
        depths += [point for point in product((0.05, 0.1), (100, 200, 400), (3, None)) if point != (0.1, 100, None)]
        return [Setting("hgb", f"lr{lr}_it{it}_d{d}", {"learning_rate": lr, "max_iter": it, "max_depth": d}) for lr, it, d in depths]
    if family == "lgbm":
        leaves: list[tuple[float, int, int]] = [(0.1, 100, 31)]
        leaves += [point for point in product((0.05, 0.1), (100, 200, 400), (7, 31)) if point != (0.1, 100, 31)]
        return [Setting("lgbm", f"lr{lr}_n{n}_l{size}", {"learning_rate": lr, "n_estimators": n, "num_leaves": size}) for lr, n, size in leaves]
    if family == "missforest":
        forests = [(False, 100), (False, 300), (True, 100), (True, 300)]
        return [Setting("missforest", f"{'half' if half else 'default'}_t{trees}", {"half_features": half, "trees": trees}) for half, trees in forests]
    raise KeyError(family)


def build(setting: Setting, numerical_columns: Sequence[str], categorical_columns: Sequence[str]) -> BaselineImputer:
    values = setting.values
    if setting.family == "knn":
        return KnnImputer(numerical_columns, categorical_columns, n_neighbors=int(cast(int, values["k"])))
    if setting.family == "hgb":
        depth = values["max_depth"]
        return TunedGradientBoostingImputer(
            numerical_columns, categorical_columns, float(cast(float, values["learning_rate"])),
            int(cast(int, values["max_iter"])), None if depth is None else int(cast(int, depth)),
        )
    if setting.family == "lgbm":
        return TunedLightGbmImputer(
            numerical_columns, categorical_columns, float(cast(float, values["learning_rate"])),
            int(cast(int, values["n_estimators"])), int(cast(int, values["num_leaves"])),
        )
    if setting.family == "missforest":
        return TunedMissForestImputer(numerical_columns, categorical_columns, bool(values["half_features"]), int(cast(int, values["trees"])))
    raise KeyError(setting.family)


SHRINKAGE = (0.0, 0.25, 0.5)


def shrink(guesses: NDArray[np.float64], means: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Numerical fills pulled toward the training mean by ``alpha``."""
    return (1.0 - alpha) * guesses + alpha * means
