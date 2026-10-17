"""The MNAR smoke generator (E45): matched counts, row alignment, a tilt toward large values."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "datasets"))

import generate_mnar  # noqa: E402


def _complete(rows: int = 500) -> pd.DataFrame:
    stream = np.random.default_rng(0)
    return pd.DataFrame({"a": stream.normal(size=rows), "b": stream.exponential(size=rows), "class": stream.integers(0, 2, rows)})


def test_each_column_loses_the_mcar_siblings_count_and_keeps_every_other_cell() -> None:
    complete = _complete()
    variant = generate_mnar.tilted(complete, 0.6)
    assert variant["class"].equals(complete["class"])
    for column in ("a", "b"):
        assert int(variant[column].isna().sum()) == int(round(len(complete) * 0.6))
        observed = variant[column].notna()
        assert (variant.loc[observed, column] == complete.loc[observed, column]).all()


def test_large_values_are_hidden_more_often_and_the_draw_is_fixed() -> None:
    complete = _complete(4000)
    variant = generate_mnar.tilted(complete, 0.6)
    for column in ("a", "b"):
        hidden = variant[column].isna()
        assert complete.loc[hidden, column].mean() > complete.loc[~hidden, column].mean()
    assert variant.equals(generate_mnar.tilted(complete, 0.6))
