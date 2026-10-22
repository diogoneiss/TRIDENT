"""E65: the decode stage's masks weighted towards where a training fold's own gaps fall.

The seam that matters is a silent one: a weighting that leaks a hidden value, moves the default
draw, or changes how many cells a column loses would all still train. So the tests pin the
weights' scale, their direction under MAR, the no-model fallbacks, and that the weighted draw
never masks a gap and keeps the uniform rule's per-column count in expectation.
"""

import numpy as np
import pandas as pd

from src.training.mask_propensity import MIN_GAPS, propensity_weights
from src.utils import draw_dynamic_mask, draw_weighted_mask


def _mar(rows: int = 2000, seed: int = 0) -> pd.DataFrame:
    """``driver`` fully observed; ``target`` missing mostly where ``driver`` is large."""
    stream = np.random.default_rng(seed)
    driver = stream.normal(size=rows)
    frame = pd.DataFrame({
        "driver": driver,
        "target": stream.normal(size=rows),
        "colour": stream.choice(["red", "blue"], size=rows),
    })
    hidden = stream.random(rows) < 1 / (1 + np.exp(-2.5 * driver)) * 0.6
    frame.loc[hidden, "target"] = np.nan
    return frame


def test_weights_lean_towards_rows_like_the_gaps_and_average_one_over_observed_cells() -> None:
    frame = _mar()
    weights = propensity_weights(frame, ["colour"])
    observed = frame["target"].notna().to_numpy()
    w = weights[:, 1]
    np.testing.assert_allclose(w[observed].mean(), 1.0, rtol=1e-9)
    high = frame["driver"].to_numpy() > 1
    assert w[observed & high].mean() > 2 * w[observed & ~high].mean(), "the masks move to rows that look like the gaps"
    assert (w[~observed] == 1).all(), "a gap keeps weight 1; it is never masked"


def test_a_column_without_enough_gaps_keeps_uniform_weights() -> None:
    frame = _mar()
    weights = propensity_weights(frame, ["colour"])
    assert (weights[:, 0] == 1).all() and (weights[:, 2] == 1).all(), "driver and colour have no gaps"
    few = frame.copy()
    few["target"] = few["target"].fillna(0.0)
    few.loc[few.index[: MIN_GAPS - 1], "target"] = np.nan
    assert (propensity_weights(few, ["colour"]) == 1).all()


def test_the_fit_draws_nothing_from_the_global_stream() -> None:
    frame = _mar()
    np.random.seed(3)
    before = np.random.get_state()[1].copy()
    propensity_weights(frame, ["colour"])
    np.testing.assert_array_equal(np.random.get_state()[1], before)


def test_the_weighted_draw_never_masks_a_gap_and_keeps_the_uniform_count_in_expectation() -> None:
    """Before the clip at 1 and the one-cell fallback for an unmasked row, each column loses the
    uniform rule's count in expectation; with enough columns the fallback is rare, so the counts
    stay within a few percent."""
    frame = _mar(rows=4000)
    for k in range(6):
        frame[f"noise{k}"] = np.random.default_rng(k).normal(size=len(frame))
    nulls = frame.isna().to_numpy()
    weights = propensity_weights(frame, ["colour"])
    np.random.seed(0)
    weighted = np.mean([draw_weighted_mask(nulls, 0.3, weights) for _ in range(30)], axis=0)
    np.random.seed(0)
    uniform = np.mean([draw_dynamic_mask(nulls, 0.3) for _ in range(30)], axis=0)
    assert (weighted[nulls] == 0).all()
    np.testing.assert_allclose(weighted.sum(axis=0), uniform.sum(axis=0), rtol=0.05)
    high = frame["driver"].to_numpy() > 1
    observed = ~nulls[:, 1]
    assert weighted[observed & high, 1].mean() > weighted[observed & ~high, 1].mean()


def test_unit_weights_draw_exactly_the_uniform_mask() -> None:
    frame = _mar()
    nulls = frame.isna().to_numpy()
    np.random.seed(11)
    plain = draw_dynamic_mask(nulls, 0.2)
    np.random.seed(11)
    weighted = draw_weighted_mask(nulls, 0.2, np.ones(nulls.shape))
    np.testing.assert_array_equal(plain, weighted)
