"""The MNAR generator (E45, E46): matched counts, row alignment, a tilt toward large values, the dial."""

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


def test_the_default_slope_keeps_e45s_names_and_the_dial_spells_its_slope() -> None:
    assert generate_mnar.variant_name("biodeg", 0.6) == "biodeg_probit_60nan"
    assert generate_mnar.variant_name("kc2", 0.2) == "kc2_probit_20nan"
    assert generate_mnar.variant_name("biodeg", 0.6, 0.25) == "biodeg_probit0p25_60nan"
    assert generate_mnar.variant_name("biodeg", 0.6, 1.0) == "biodeg_probit1p0_60nan"
    assert generate_mnar.variant_name("biodeg", 0.6, 2.0) == "biodeg_probit2p0_60nan"


def test_a_steeper_slope_hides_larger_values_and_the_default_draw_is_unchanged() -> None:
    complete = _complete(4000)
    gaps = [
        complete.loc[generate_mnar.tilted(complete, 0.6, slope=slope)["a"].isna(), "a"].mean()
        for slope in (0.25, 0.5, 1.0, 2.0)
    ]
    assert gaps == sorted(gaps)
    assert generate_mnar.tilted(complete, 0.6, slope=0.5).equals(generate_mnar.tilted(complete, 0.6))
