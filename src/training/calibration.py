"""Calibrating the decoder's guesses on the validation split's own gaps.

A diagnostic (``--score_calibrated``, ticket imputation-token-shape/04): fitted on the validation
rows' induced cells, scored against the complete sibling as the test ones are, and applied to the
test split's induced cells. Pure functions over tables of scored cells, in the layout
``_score_population`` produces (``column``, ``kind``, ``actual``, ``imputed``, ``confidence``).

- **Numbers** are shrunk toward the column's training mean ``b``: the guess ``p`` becomes
  ``b + alpha (p - b)``. ``alpha`` is the least-squares factor of the truth ``a`` on the guess,
  ``sum (a - b)(p - b) / sum (p - b)^2``, clipped to [0, 1] so it never amplifies or flips a
  guess, then pulled toward 1 as ``(n alpha + n0) / (n + n0)`` with ``n0`` cells of prior, so a
  column the validation split barely covers is left nearly as it was.
- **Categories** fall back to the training mode wherever the guess's confidence is below a
  threshold. The threshold is chosen from a fixed grid (none, or the 0.1, 0.2, 0.3 and 0.5
  quantiles of the column's validation confidences) as the one with the lowest validation error,
  and only in a column with at least ``MIN_CATEGORICAL_CELLS`` validation cells; a tie keeps the
  guesses.

The constants were fixed in the ticket before any calibrated score was seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .imputation_metrics import CATEGORICAL, NUMERICAL

SHRINK_PRIOR_CELLS = 30
MIN_CATEGORICAL_CELLS = 30
THRESHOLD_QUANTILES = (0.1, 0.2, 0.3, 0.5)


@dataclass(frozen=True)
class Calibration:
    """Per column: the shrinking factor of a numerical one, the confidence threshold of a
    categorical one (None keeps every guess)."""

    alpha: Mapping[str, float]
    threshold: Mapping[str, float | None]


def fit_calibration(validation_cells: pd.DataFrame, baselines: Mapping[str, float | str]) -> Calibration:
    alpha: dict[str, float] = {}
    threshold: dict[str, float | None] = {}
    for column, group in validation_cells.groupby("column", sort=True):
        name = str(column)
        if group["kind"].iloc[0] == NUMERICAL:
            mean = float(baselines[name])
            actual = group["actual"].to_numpy(dtype=float) - mean
            guess = group["imputed"].to_numpy(dtype=float) - mean
            spread = float(np.dot(guess, guess))
            fitted = float(np.clip(np.dot(actual, guess) / spread, 0.0, 1.0)) if spread > 0 else 1.0
            cells = len(group)
            alpha[name] = (cells * fitted + SHRINK_PRIOR_CELLS) / (cells + SHRINK_PRIOR_CELLS)
        elif group["kind"].iloc[0] == CATEGORICAL:
            threshold[name] = _threshold(group, str(baselines[name]))
    return Calibration(alpha=alpha, threshold=threshold)


def _threshold(group: pd.DataFrame, mode: str) -> float | None:
    if len(group) < MIN_CATEGORICAL_CELLS:
        return None
    actual = group["actual"].to_numpy().astype(str)
    guess = group["imputed"].to_numpy().astype(str)
    confidence = group["confidence"].to_numpy(dtype=float)
    best: float | None = None
    best_error = float(np.mean(guess != actual))
    for quantile in THRESHOLD_QUANTILES:
        candidate = float(np.quantile(confidence, quantile))
        answer = np.where(confidence < candidate, mode, guess)
        error = float(np.mean(answer != actual))
        if error < best_error:
            best, best_error = candidate, error
    return best


def apply_calibration(
    cells: pd.DataFrame, calibration: Calibration, baselines: Mapping[str, float | str]
) -> pd.DataFrame:
    """The same cells with calibrated guesses; a column the calibration never saw is left alone."""
    calibrated = cells.copy()
    imputed = calibrated["imputed"].astype(object).to_numpy()
    for column, factor in calibration.alpha.items():
        rows = ((calibrated["column"] == column) & (calibrated["kind"] == NUMERICAL)).to_numpy()
        if rows.any():
            mean = float(baselines[column])
            guess = calibrated.loc[rows, "imputed"].to_numpy(dtype=float)
            imputed[rows] = mean + factor * (guess - mean)
    for column, cut in calibration.threshold.items():
        if cut is None:
            continue
        rows = (
            (calibrated["column"] == column)
            & (calibrated["kind"] == CATEGORICAL)
            & (calibrated["confidence"].to_numpy(dtype=float) < cut)
        ).to_numpy()
        imputed[rows] = str(baselines[column])
    calibrated["imputed"] = imputed
    return calibrated
