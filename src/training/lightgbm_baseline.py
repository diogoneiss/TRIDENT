"""The ``lgbm`` baseline imputer: the gradient-boosting baseline's recipe, with LightGBM.

One LightGBM model per column, trained exactly as ``hgb`` trains its booster (ADR 0007):
on the training rows where the column is observed plus a copy of them with a fifth of the
feature cells hidden, every other column a feature, gaps routed natively, filling a hidden
cell from the rest of its row in a single pass. Only the learner differs: LightGBM's
defaults where ``hgb`` has scikit-learn's ``HistGradientBoosting``, which was modelled on
it. The one default that differs in practice: scikit-learn stops early on folds above
10,000 rows, LightGBM always trains its 100 rounds. Part of the baseline bar (ADR 0017).

It lives apart from ``imputation_baselines.py`` so the baseline cache keeps that module's
entries: their key reads its source.
"""

from typing import Sequence, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import NDArray

from .imputation_baselines import BOOSTING_SEED, BaselineImputer, GradientBoostingImputer, _ColumnModel, _Constant

# Threads per model. The fill does not depend on it (``deterministic`` with a fixed
# row-wise histogram layout), so it only weighs speed against the cores a training run
# shares with it.
THREADS = 8


class LightGbmImputer(GradientBoostingImputer):
    """``hgb``'s recipe, every step of it inherited, with a LightGBM model per column."""

    name: str = "lgbm"

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
        declared = np.flatnonzero(categorical).tolist()
        model: LGBMRegressor | LGBMClassifier
        if numerical:
            model = LGBMRegressor(
                random_state=BOOSTING_SEED, n_jobs=THREADS, deterministic=True, force_row_wise=True, verbose=-1
            ).fit(features, target, categorical_feature=declared)
        else:
            model = LGBMClassifier(
                random_state=BOOSTING_SEED, n_jobs=THREADS, deterministic=True, force_row_wise=True, verbose=-1
            ).fit(features, target.astype(int), categorical_feature=declared)
        # The parent types its models as scikit-learn's boosters, and only ever calls
        # ``predict`` on them, which a LightGBM model answers the same way.
        return cast(_ColumnModel, model)


def fit_lightgbm_imputers(
    train_frame: pd.DataFrame,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> list[BaselineImputer]:
    """The ``lgbm`` baseline, fit on this fold's training split. Once per fold."""
    imputer = LightGbmImputer(numerical_columns, categorical_columns)
    imputer.fit(train_frame)
    return [imputer]
