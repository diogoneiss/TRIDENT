"""The baselines with a tuning budget (E42): the grids, the early-stopping split and the shrinkage."""

import numpy as np
import pandas as pd
import pytest

from src.training.imputation_baselines import GradientBoostingImputer, KnnImputer
from src.training.lightgbm_baseline import LightGbmImputer
from src.training.missforest import MissForestImputer
from src.training.tuned_baselines import build, by_original_row, grid, shrink


def test_each_grid_is_the_declared_size_and_starts_at_the_current_setting() -> None:
    """A tie on the validation rows keeps the setting met first, so the family's current one,
    the untuned bar, must lead its grid."""
    sizes = {family: len(grid(family)) for family in ("knn", "hgb", "lgbm", "missforest")}
    assert sizes == {"knn": 4, "hgb": 12, "lgbm": 12, "missforest": 4}
    assert grid("knn")[0].values == {"k": 5}
    assert grid("hgb")[0].values == {"learning_rate": 0.1, "max_iter": 100, "max_depth": None}
    assert grid("lgbm")[0].values == {"learning_rate": 0.1, "n_estimators": 100, "num_leaves": 31}
    assert grid("missforest")[0].values == {"half_features": False, "trees": 100}
    for family in ("hgb", "lgbm"):
        labels = [setting.label for setting in grid(family)]
        assert len(set(labels)) == len(labels)


def test_the_early_stopping_split_never_puts_a_row_and_its_hidden_copy_on_both_sides() -> None:
    rows = 1000
    train, held = by_original_row(rows)
    assert len(held) == 2 * 100 and len(train) == 2 * 900
    assert not set(train % rows) & set(held % rows)
    assert sorted(np.concatenate([train, held]).tolist()) == list(range(2 * rows))
    again_train, again_held = by_original_row(rows)
    assert (again_held == held).all() and (again_train == train).all()


def test_shrinkage_pulls_toward_the_mean_and_zero_changes_nothing() -> None:
    guesses, means = np.array([2.0, -1.0]), np.array([0.0, 1.0])
    assert shrink(guesses, means, 0.0).tolist() == [2.0, -1.0]
    assert shrink(guesses, means, 0.5).tolist() == [1.0, 0.0]


def test_the_current_settings_fill_exactly_as_the_live_baselines_do() -> None:
    """The first point of each grid is the bar's own imputer: the tuned bar can only be lower."""
    stream = np.random.default_rng(0)
    frame = pd.DataFrame({"a": stream.normal(size=120), "b": stream.normal(size=120), "kind": stream.choice(["x", "y"], 120)})
    frame["b"] += frame["a"]
    train = frame.mask(stream.random(frame.shape) < 0.2)
    test = frame.iloc[:20].reset_index(drop=True)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[::2, 1] = True
    hidden[1::2, 2] = True
    live = {
        "knn": KnnImputer(["a", "b"], ["kind"], n_neighbors=5),
        "hgb": GradientBoostingImputer(["a", "b"], ["kind"]),
        "lgbm": LightGbmImputer(["a", "b"], ["kind"]),
        "missforest": MissForestImputer(["a", "b"], ["kind"]),
    }
    for family, reference in live.items():
        tuned = build(grid(family)[0], ["a", "b"], ["kind"])
        reference.fit(train)
        tuned.fit(train)
        pd.testing.assert_frame_equal(tuned.impute(test, hidden), reference.impute(test, hidden), obj=family)


@pytest.mark.parametrize("family", ["hgb", "lgbm", "missforest"])
def test_every_setting_of_a_grid_fits_and_fills(family: str) -> None:
    stream = np.random.default_rng(1)
    frame = pd.DataFrame({"a": stream.normal(size=80), "kind": stream.choice(["x", "y", "z"], 80)})
    train = frame.mask(stream.random(frame.shape) < 0.2)
    hidden = np.zeros(frame.shape, dtype=bool)
    hidden[::3, 0] = True
    for setting in grid(family):
        imputer = build(setting, ["a"], ["kind"])
        imputer.fit(train)
        filled = imputer.impute(frame, hidden)
        assert not filled["a"].isna().any(), setting.label
