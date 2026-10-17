"""Linear and low-rank baseline imputers: condexp, inductive SoftImpute and ICE (E41).

Three classical imputers the 2026-10-16 review found missing from the comparison, each fit on the
training fold and asked to fill whatever ADR 0007's protocol hides (``BaselineImputer``):

- ``condexp``: each row's hidden numbers get their Gaussian conditional expectation given the
  row's observed numbers, under a shrunk pairwise-complete covariance;
- ``softimpute``: a rank-k fit of the training fold by hard-impute, and each test row's hidden
  numbers from the PPCA posterior of its observed ones (an inductive SoftImpute);
- ``ice``: missForest's rounds with linear learners, BayesianRidge for numbers and a logistic
  regression for categories, over one-hot categorical features, applied to the test rows round by
  round as scikit-learn's ``IterativeImputer`` transforms. missForest's own fill replays the kept
  round's models from the training means, which is invalid for linear learners (scores up to
  about 19x the mean/mode fill in the review's probe).

condexp and SoftImpute fill categorical columns with the training mode. Every numerical fill is
clipped to the column's range on the training fold. The rules were fixed in
``docs/tickets/imputation-protocol/01-linear-baselines.md`` before anything was scored. Nothing
here draws from a random stream. This module is new so that ``imputation_baselines.py`` and
``imputation_metrics.py``, whose bytes key every cached baseline (ADR 0007 decision 10), stay as
they are.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import BayesianRidge, LogisticRegression

from src.embedder import as_category_strings

from .imputation_baselines import _NOT_A_CLASS, BaselineImputer, MeanModeImputer, _Constant, _position
from .missforest import MAX_ROUNDS, MissForestImputer, RoundChange, still_converging

# condexp: the correlation is shrunk toward the identity by this share and its eigenvalues are
# floored, so a covariance estimated from few co-observed rows stays invertible.
CONDEXP_SHRINK = 0.1
CONDEXP_EIGENVALUE_FLOOR = 1e-3
# SoftImpute: the rank keeps this share of the pairwise covariance's positive eigenvalues, at
# most p - 1 (no p // 2 cap, which broke the probe on spambase); the hard-impute fit stops
# at this relative change or this many iterations.
SOFT_ENERGY = 0.9
SOFT_MAX_ITERATIONS = 100
SOFT_TOLERANCE = 1e-5
# A numerical guard on the PPCA noise variance, binding only when the discarded spectrum is empty.
SOFT_NOISE_FLOOR = 1e-6
# ICE's logistic regressions: lbfgs, multinomial on more than two classes.
LOGISTIC_MAX_ITERATIONS = 1000


class _Predicts(Protocol):
    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...


def _pairwise_covariance(matrix: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The observed means and the pairwise-complete covariance around them.

    Each entry sums the centred products over the rows where both columns are observed and
    divides by that count less one; an off-diagonal entry with fewer than two such rows is 0, and
    a variance from fewer than two values (or none above 0) is 1.
    """
    observed = ~np.isnan(matrix)
    counts = observed.sum(axis=0)
    means = np.where(counts > 0, np.nansum(matrix, axis=0) / np.maximum(counts, 1), 0.0)
    centred = np.where(observed, matrix - means, 0.0)
    indicator = observed.astype(float)
    pairs = indicator.T @ indicator
    covariance = (centred.T @ centred) / np.maximum(pairs - 1.0, 1.0)
    covariance[pairs < 2] = 0.0
    diagonal = np.diag(covariance).copy()
    diagonal[(counts < 2) | (diagonal <= 0)] = 1.0
    np.fill_diagonal(covariance, diagonal)
    return means, covariance


class _NumericalFill:
    """Categorical columns by the training mode; numerical ones by the subclass's model, clipped."""

    name: str = "?"

    def __init__(self, numerical_columns: Sequence[str], categorical_columns: Sequence[str]) -> None:
        self._numerical = list(numerical_columns)
        self._categorical = list(categorical_columns)
        self._mean_mode = MeanModeImputer(numerical_columns, categorical_columns)
        self._means: NDArray[np.float64] = np.empty(0)
        self._low: NDArray[np.float64] = np.empty(0)
        self._high: NDArray[np.float64] = np.empty(0)

    def fit(self, frame: pd.DataFrame) -> None:
        self._mean_mode.fit(frame)
        if not self._numerical:
            return
        matrix = frame[self._numerical].to_numpy(dtype=float)
        observed = ~np.isnan(matrix)
        self._low = np.array([matrix[observed[:, j], j].min() if observed[:, j].any() else -np.inf for j in range(matrix.shape[1])])
        self._high = np.array([matrix[observed[:, j], j].max() if observed[:, j].any() else np.inf for j in range(matrix.shape[1])])
        self._fit_numerical(matrix)

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        filled = self._mean_mode.impute(frame, hidden)
        if not self._numerical:
            return filled
        missing = hidden | frame.isna().to_numpy()
        gaps = missing[:, [_position(frame, column) for column in self._numerical]]
        # The hidden cells still hold their truth in the frame; they are blanked before anything
        # reads the row.
        matrix = frame[self._numerical].to_numpy(dtype=float).copy()
        matrix[gaps] = np.nan
        numbers = np.clip(self._fill(matrix, gaps), self._low, self._high)
        for index, column in enumerate(self._numerical):
            filled[column] = np.where(gaps[:, index], numbers[:, index], frame[column].to_numpy(dtype=float))
        return filled.reindex(columns=frame.columns)

    def _fill(self, matrix: NDArray[np.float64], gaps: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Every row's hidden numbers from its observed ones, one solve per missing pattern."""
        filled = matrix.copy()
        patterns, inverse = np.unique(gaps, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        for index, pattern in enumerate(patterns):
            if not pattern.any():
                continue
            rows = np.nonzero(inverse == index)[0]
            hidden_columns, seen_columns = np.nonzero(pattern)[0], np.nonzero(~pattern)[0]
            filled[np.ix_(rows, hidden_columns)] = self._predict(
                matrix[np.ix_(rows, seen_columns)], pattern, ~pattern
            )
        return filled

    def _fit_numerical(self, matrix: NDArray[np.float64]) -> None:
        raise NotImplementedError

    def _predict(
        self, seen: NDArray[np.float64], hidden: NDArray[np.bool_], observed: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        raise NotImplementedError


class ConditionalExpectationImputer(_NumericalFill):
    """condexp: E[x_hidden | x_observed] under a shrunk pairwise-complete Gaussian (no EM)."""

    name = "condexp"

    def __init__(self, numerical_columns: Sequence[str], categorical_columns: Sequence[str]) -> None:
        super().__init__(numerical_columns, categorical_columns)
        self._covariance: NDArray[np.float64] = np.empty((0, 0))

    def _fit_numerical(self, matrix: NDArray[np.float64]) -> None:
        means, covariance = _pairwise_covariance(matrix)
        spread = np.sqrt(np.diag(covariance))
        correlation = np.clip(covariance / np.outer(spread, spread), -1.0, 1.0)
        np.fill_diagonal(correlation, 1.0)
        correlation = (1 - CONDEXP_SHRINK) * correlation + CONDEXP_SHRINK * np.eye(len(correlation))
        eigenvalues, vectors = np.linalg.eigh(correlation)
        correlation = (vectors * np.maximum(eigenvalues, CONDEXP_EIGENVALUE_FLOOR)) @ vectors.T
        self._means = means
        self._covariance = correlation * np.outer(spread, spread)

    def _predict(
        self, seen: NDArray[np.float64], hidden: NDArray[np.bool_], observed: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        if not observed.any():
            return np.tile(self._means[hidden], (len(seen), 1))
        coefficients = np.linalg.solve(
            self._covariance[np.ix_(observed, observed)], self._covariance[np.ix_(observed, hidden)]
        )
        result: NDArray[np.float64] = self._means[hidden] + (seen - self._means[observed]) @ coefficients
        return result


class SoftImputeImputer(_NumericalFill):
    """Inductive SoftImpute: a rank-k hard-impute fit, and a PPCA posterior for each test row."""

    name = "softimpute"

    def __init__(self, numerical_columns: Sequence[str], categorical_columns: Sequence[str]) -> None:
        super().__init__(numerical_columns, categorical_columns)
        self._loadings: NDArray[np.float64] = np.empty((0, 0))
        self._noise = SOFT_NOISE_FLOOR
        # Diagnostics: the rank chosen, the share of energy it keeps, the iterations it took.
        self.rank = 0
        self.energy = 0.0
        self.iterations = 0

    def _fit_numerical(self, matrix: NDArray[np.float64]) -> None:
        rows, width = matrix.shape
        means, covariance = _pairwise_covariance(matrix)
        spectrum = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
        total = float(spectrum.sum())
        if width == 1:
            rank = 1
        else:
            cumulative = np.cumsum(spectrum) / total if total > 0 else np.ones(width)
            rank = max(1, min(int(np.searchsorted(cumulative, SOFT_ENERGY)) + 1, width - 1))
        observed = ~np.isnan(matrix)
        target = np.where(observed, matrix - means, 0.0)
        completed = target.copy()
        for iteration in range(1, SOFT_MAX_ITERATIONS + 1):
            left, singular, right = np.linalg.svd(completed, full_matrices=False)
            low_rank = (left[:, :rank] * singular[:rank]) @ right[:rank]
            updated = np.where(observed, target, low_rank)
            change = float(((updated - completed) ** 2).sum()) / max(float((completed**2).sum()), 1e-12)
            completed = updated
            self.iterations = iteration
            if change < SOFT_TOLERANCE:
                break
        _, singular, right = np.linalg.svd(completed, full_matrices=False)
        self._loadings = right[:rank].T * (singular[:rank] / np.sqrt(rows))
        # The PPCA noise: the mean of the fitted matrix's covariance eigenvalues the rank leaves
        # out, over all p - k of them (any beyond the rows' count are zero).
        discarded = float((singular[rank:] ** 2).sum()) / rows
        self._noise = max(discarded / (width - rank), SOFT_NOISE_FLOOR) if width > rank else SOFT_NOISE_FLOOR
        self._means = means
        self.rank = rank
        self.energy = float(spectrum[:rank].sum() / total) if total > 0 else 1.0

    def _predict(
        self, seen: NDArray[np.float64], hidden: NDArray[np.bool_], observed: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        if not observed.any():
            return np.tile(self._means[hidden], (len(seen), 1))
        on_seen, on_hidden = self._loadings[observed], self._loadings[hidden]
        rank = self._loadings.shape[1]
        projection = np.linalg.solve(on_seen.T @ on_seen + self._noise * np.eye(rank), on_seen.T)
        result: NDArray[np.float64] = self._means[hidden] + ((seen - self._means[observed]) @ projection.T) @ on_hidden.T
        return result


class SequentialRoundsImputer(MissForestImputer):
    """missForest's fit, kept round by round and applied to new rows as ``IterativeImputer`` does.

    The fit is missForest's: start at the training means and modes, visit the columns by
    increasing missingness (Gauss-Seidel), stop by ``still_converging``. Every kept round's models
    are kept. A fill starts the hidden and missing cells at the training means and modes and
    applies rounds 1 to k in order, each column by column, k being the fit's kept rounds; there is
    no stopping rule at fill time. With ``one_hot`` the categorical features enter as one-hot
    blocks over the training vocabulary, the target's own block left out; otherwise as vocabulary
    codes, as missForest feeds its forests. With ``clip`` every numerical fill, in the fit and in
    the fill, is clipped to the column's training range inside every round.
    """

    name = "missforest_seq"
    one_hot = False
    clip = False

    def __init__(self, numerical_columns: Sequence[str], categorical_columns: Sequence[str]) -> None:
        super().__init__(numerical_columns, categorical_columns)
        self._rounds: list[list[_Predicts]] = []
        self._low: NDArray[np.float64] = np.empty(0)
        self._high: NDArray[np.float64] = np.empty(0)

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
        width = matrix.shape[1]
        self._start = np.array([self._first_guess(matrix[~gaps[:, index], index], index) for index in range(width)])
        self._low = np.array([matrix[~gaps[:, j], j].min() if (~gaps[:, j]).any() else -np.inf for j in range(width)])
        self._high = np.array([matrix[~gaps[:, j], j].max() if (~gaps[:, j]).any() else np.inf for j in range(width)])
        self._order = np.argsort(gaps.sum(axis=0), kind="stable").tolist()

        fills = np.where(gaps, self._start, matrix)
        history: list[list[_Predicts]] = []
        previous: RoundChange | None = None
        for _ in range(MAX_ROUNDS):
            models: list[_Predicts] = [_Constant(0.0)] * width
            updated = fills.copy()
            for index in self._order:
                observed = ~gaps[:, index]
                models[index] = self._fit_one(index, self._design(updated[observed], index), matrix[observed, index])
                if gaps[:, index].any():
                    updated[gaps[:, index], index] = self._bounded(
                        index, models[index].predict(self._design(updated[gaps[:, index]], index))
                    )
            change = self._change(fills, updated, gaps)
            if previous is not None and not still_converging(previous, change):
                break
            history.append(models)
            fills, previous = updated, change
            self.rounds += 1
        self._rounds = history

    def impute(self, frame: pd.DataFrame, hidden: NDArray[np.bool_]) -> pd.DataFrame:
        missing = hidden | frame.isna().to_numpy()
        matrix = self._matrix(frame, missing)
        gaps = np.isnan(matrix)
        fills = np.where(gaps, self._start, matrix)
        for models in self._rounds:
            for index in self._order:
                if gaps[:, index].any():
                    fills[gaps[:, index], index] = self._bounded(
                        index, models[index].predict(self._design(fills[gaps[:, index]], index))
                    )
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
            strings[at] = np.array(self._classes[column], dtype=object)[fills[at, index].astype(int)]
            columns[column] = strings
        return pd.DataFrame(columns, index=frame.index).reindex(columns=frame.columns)

    def _design(self, fills: NDArray[np.float64], index: int) -> NDArray[np.float64]:
        """The features column ``index`` is predicted from: every other column's current fills."""
        if not self.one_hot:
            return fills[:, self._others(index)]
        split = len(self._numerical)
        blocks = [fills[:, [j for j in range(split) if j != index]]]
        for offset, column in enumerate(self._categorical):
            position = split + offset
            if position == index:
                continue
            codes = fills[:, position].astype(int)
            blocks.append(np.eye(len(self._classes[column]))[codes])
        return np.column_stack(blocks) if blocks else np.empty((len(fills), 0))

    def _bounded(self, index: int, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.clip and self._is_numerical(index):
            return np.clip(values, self._low[index], self._high[index])
        return np.asarray(values, dtype=float)


class IceImputer(SequentialRoundsImputer):
    """ICE: chained linear conditionals, one-hot categorical features, the sequential fill, clipped."""

    name = "ice"
    one_hot = True
    clip = True

    def _fit_one(  # type: ignore[override]
        self, index: int, features: NDArray[np.float64], target: NDArray[np.float64]
    ) -> _Predicts:
        if len(target) == 0:
            return _Constant(self._start[index])
        values, counts = np.unique(target, return_counts=True)
        if len(values) == 1:
            return _Constant(float(values[0]))
        if features.shape[1] == 0:
            return _Constant(float(target.mean()) if self._is_numerical(index) else float(values[counts.argmax()]))
        if self._is_numerical(index):
            return BayesianRidge().fit(features, target)
        return _Labels(LogisticRegression(max_iter=LOGISTIC_MAX_ITERATIONS).fit(features, target.astype(int)))


class _Labels:
    """A classifier read as numbers: its predicted vocabulary codes, as floats."""

    def __init__(self, model: LogisticRegression) -> None:
        self._model = model

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self._model.predict(features), dtype=float)


def fit_linear_imputers(
    train_frame: pd.DataFrame,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> list[BaselineImputer]:
    """condexp, SoftImpute and ICE, fit on this fold's training split. Once per fold."""
    imputers: list[BaselineImputer] = [
        ConditionalExpectationImputer(numerical_columns, categorical_columns),
        SoftImputeImputer(numerical_columns, categorical_columns),
        IceImputer(numerical_columns, categorical_columns),
    ]
    for imputer in imputers:
        imputer.fit(train_frame)
    return imputers
