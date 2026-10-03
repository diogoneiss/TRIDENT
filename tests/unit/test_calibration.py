"""Calibrating the decoder's guesses on the validation split's own gaps (imputation-token-shape/04).

Numbers are shrunk toward the column's training mean by a factor fitted on the validation cells
and pulled toward 1 when they are few; categories fall back to the training mode below a
confidence threshold chosen on the validation cells, only where a column has enough of them.
"""

import numpy as np
import pandas as pd
import pytest

from src.training.calibration import MIN_CATEGORICAL_CELLS, SHRINK_PRIOR_CELLS, apply_calibration, fit_calibration


def _numerical(column: str, actual: np.ndarray, imputed: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"column": column, "kind": "numerical", "actual": actual, "imputed": imputed, "confidence": np.nan}
    )


def _categorical(column: str, actual: list[str], imputed: list[str], confidence: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"column": column, "kind": "categorical", "actual": actual, "imputed": imputed, "confidence": confidence}
    )


def test_an_overconfident_number_is_shrunk_toward_the_training_mean_by_the_fitted_factor() -> None:
    """A guess twice as far from the mean as the truth fits a factor of one half; with 3000 cells
    the pull toward 1 barely moves it."""
    rng = np.random.default_rng(0)
    actual = rng.normal(1.0, 1.0, 3000)
    mean = 1.0
    validation = _numerical("size", actual, mean + 2 * (actual - mean))

    calibration = fit_calibration(validation, {"size": mean})

    expected = (3000 * 0.5 + SHRINK_PRIOR_CELLS) / (3000 + SHRINK_PRIOR_CELLS)
    assert calibration.alpha["size"] == pytest.approx(expected, abs=1e-9)
    test = _numerical("size", np.array([0.0, 3.0]), np.array([-1.0, 5.0]))
    calibrated = apply_calibration(test, calibration, {"size": mean})
    assert calibrated["imputed"].tolist() == pytest.approx([mean + expected * -2.0, mean + expected * 4.0])
    assert calibrated["actual"].tolist() == test["actual"].tolist()


def test_a_column_with_few_cells_is_left_almost_as_it_was() -> None:
    """Ten cells are not enough to trust a factor: the pull toward 1 dominates."""
    actual = np.linspace(-1, 1, 10)
    calibration = fit_calibration(_numerical("size", actual, 3 * actual), {"size": 0.0})
    assert calibration.alpha["size"] > 0.8


def test_the_factor_never_amplifies_and_never_flips_a_guess() -> None:
    actual = np.linspace(-1, 1, 1000)
    amplify = fit_calibration(_numerical("a", actual, 0.5 * actual), {"a": 0.0})
    flip = fit_calibration(_numerical("b", actual, -actual), {"b": 0.0})
    assert amplify.alpha["a"] == pytest.approx(1.0)
    assert flip.alpha["b"] == pytest.approx(SHRINK_PRIOR_CELLS / (1000 + SHRINK_PRIOR_CELLS))


def test_a_low_confidence_category_falls_back_to_the_mode_when_that_is_better_on_validation() -> None:
    """The 30 least confident guesses are wrong where the mode is right; the threshold that cuts
    them is chosen, and a column with too few cells keeps every guess."""
    actual = ["red"] * 30 + ["blue"] * 70
    imputed = ["green"] * 30 + ["blue"] * 70
    confidence = [0.1] * 30 + [0.9] * 70
    few = MIN_CATEGORICAL_CELLS - 1
    validation = pd.concat(
        [
            _categorical("colour", actual, imputed, confidence),
            _categorical("shape", ["round"] * few, ["square"] * few, [0.1] * few),
        ],
        ignore_index=True,
    )
    baselines = {"colour": "red", "shape": "round"}

    calibration = fit_calibration(validation, baselines)

    assert calibration.threshold["colour"] is not None
    assert calibration.threshold["shape"] is None
    calibrated = apply_calibration(validation, calibration, baselines)
    colour = calibrated[calibrated["column"] == "colour"]
    assert (colour["imputed"] == colour["actual"]).all()
    shape = calibrated[calibrated["column"] == "shape"]
    assert (shape["imputed"] == "square").all()


def test_a_threshold_is_kept_only_when_it_lowers_the_validation_error() -> None:
    actual = ["blue"] * 100
    calibration = fit_calibration(
        _categorical("colour", actual, ["blue"] * 100, list(np.linspace(0.1, 0.9, 100))), {"colour": "red"}
    )
    assert calibration.threshold["colour"] is None
