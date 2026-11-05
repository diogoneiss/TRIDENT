"""The distributional score read beside ``impute_score`` (imputation-protocol/51)."""

import numpy as np
import pandas as pd
import pytest

from src.training.distribution_metrics import distribution_scores
from src.training.imputation_metrics import score_cells

BASELINES = {"size": 0.0, "colour": "red"}


def _cells(sizes: list[float], colours: list[str]) -> pd.DataFrame:
    truth_sizes = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    truth_colours = ["red", "red", "red", "blue", "blue", "green"]
    rows = [{"row": i, "column": "size", "kind": "numerical", "actual": a, "imputed": g} for i, (a, g) in enumerate(zip(truth_sizes, sizes))]
    rows += [{"row": i, "column": "colour", "kind": "categorical", "actual": a, "imputed": g} for i, (a, g) in enumerate(zip(truth_colours, colours))]
    return pd.DataFrame(rows)


def test_the_naive_fill_scores_one_and_a_perfect_fill_zero() -> None:
    naive = distribution_scores(_cells([0.0] * 6, ["red"] * 6), BASELINES)
    perfect = distribution_scores(_cells([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], ["red", "red", "red", "blue", "blue", "green"]), BASELINES)

    # The truths' mean absolute deviation from the fill, and the share of truths off the mode.
    assert naive["wd_num_z"] == pytest.approx(9 / 6)
    assert naive["tv_cat"] == pytest.approx(0.5)
    assert naive["dist_score"] == pytest.approx(1.0)
    assert (perfect["wd_num_z"], perfect["tv_cat"], perfect["dist_score"]) == (0.0, 0.0, 0.0)


def test_it_reads_the_distribution_and_not_the_rows() -> None:
    """The right values on the wrong rows: every column's distribution is the truth's, so the score
    is perfect where ``impute_score`` is worse than the naive fill. Why it is never read alone."""
    shuffled = _cells([3.0, 2.0, 1.0, 0.0, -1.0, -2.0], ["green", "blue", "blue", "red", "red", "red"])

    assert distribution_scores(shuffled, BASELINES)["dist_score"] == pytest.approx(0.0)
    assert score_cells(shuffled, BASELINES).metrics["impute_score"] > 1.0


def test_a_fill_shrunk_to_the_centre_is_penalised_by_how_much_it_shrinks() -> None:
    shrunk = _cells([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], ["red", "red", "red", "red", "blue", "blue"])
    scores = distribution_scores(shrunk, BASELINES)

    assert scores["wd_num_z"] == pytest.approx(np.mean([1.0, 0.5, 0.0, 0.5, 1.0, 1.5]))
    # red 4/6 against 3/6, blue 2/6 against 2/6, green 0 against 1/6.
    assert scores["tv_cat"] == pytest.approx(0.5 * (1 / 6 + 0 + 1 / 6))


def test_one_kind_alone_and_a_column_the_naive_fill_gets_right() -> None:
    numbers = _cells([0.0] * 6, ["red"] * 6)
    numbers = numbers[numbers["kind"] == "numerical"]
    assert distribution_scores(numbers, BASELINES) == {"wd_num_z": pytest.approx(1.5), "dist_score": pytest.approx(1.0)}

    constant = pd.DataFrame({"row": [0, 1], "column": "size", "kind": "numerical", "actual": [0.0, 0.0], "imputed": [0.5, 0.5]})
    # Nothing for the naive fill to get wrong: parity plus the model's own distance, as ``_ratio``.
    assert distribution_scores(constant, BASELINES)["dist_score"] == pytest.approx(1.5)
    assert distribution_scores(numbers.iloc[0:0], BASELINES) == {}
