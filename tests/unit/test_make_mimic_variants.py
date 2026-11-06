"""E149's cutter: the nearest incomplete row, and three masks with one count of gaps per column."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "datasets"))

import make_mimic_variants  # noqa: E402


def test_each_complete_row_is_matched_to_its_nearest_incomplete_row_on_the_columns_without_gaps() -> None:
    stream = np.random.default_rng(0)
    table = pd.DataFrame({"a": stream.normal(size=400), "b": stream.exponential(size=400) * 50,
                          "sparse": stream.normal(size=400), "kind": stream.choice(["x", "y"], size=400)})
    table.loc[stream.random(400) < 0.6, "sparse"] = np.nan
    table.loc[5, "a"] = np.nan  # a gap in a matching column: the row stays in the pool, at the mean

    donors, sources = make_mimic_variants.nearest_incomplete(table)

    incomplete = table.isna().any(axis=1).to_numpy()
    assert donors.tolist() == np.flatnonzero(~incomplete).tolist()
    assert incomplete[sources].all()
    standardised = ((table[["a", "b"]] - table[["a", "b"]].mean()) / table[["a", "b"]].std()).fillna(0.0).to_numpy()
    for donor, source in zip(donors, sources):
        distances = np.linalg.norm(standardised[incomplete] - standardised[donor], axis=1)
        assert np.isclose(np.linalg.norm(standardised[source] - standardised[donor]), distances.min())


def test_the_three_variants_have_one_count_of_gaps_per_column_and_trand_keeps_the_patterns() -> None:
    stream = np.random.default_rng(1)
    tmatch = stream.random((500, 6)) < np.array([0.0, 0.1, 0.3, 0.3, 0.6, 0.0])
    tmatch[:, 3] = tmatch[:, 2]  # a block: two columns missing together

    masks, order = make_mimic_variants.cut(tmatch, seed=42)

    for mask in masks.values():
        assert (mask.sum(axis=0) == tmatch.sum(axis=0)).all()
    assert (masks["trand"] == tmatch[order]).all() and sorted(order) == list(range(500))
    assert (masks["trand"][:, 3] == masks["trand"][:, 2]).all()  # the block survives the permutation
    assert not (masks["mcar"][:, 3] == masks["mcar"][:, 2]).all()  # and not the uniform draw
    again, _ = make_mimic_variants.cut(tmatch, seed=42)
    assert all((again[name] == masks[name]).all() for name in masks)
