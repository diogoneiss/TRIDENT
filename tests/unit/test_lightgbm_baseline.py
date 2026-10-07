"""The single-pass LightGBM baseline, ``lgbm``: the hgb recipe with LightGBM (ADR 0017)."""

import random

import numpy as np
import pandas as pd
import pytest
import torch

from src.training import lightgbm_baseline
from src.training.lightgbm_baseline import LightGbmImputer, fit_lightgbm_imputers


def _step_frame(rows: int) -> pd.DataFrame:
    """``level`` jumps from -3 to 3 where ``x`` passes 10, and ``side`` follows the same
    threshold, with an unrelated ``colour`` in between: a rule a per-column model finds
    exactly and an average of neighbours only approaches near the jump."""
    x = [float(index % 21) for index in range(rows)]
    return pd.DataFrame(
        {
            "side": ["low" if value <= 10 else "high" for value in x],
            "x": x,
            "colour": [["red", "blue", "green"][index % 3] for index in range(rows)],
            "level": [-3.0 if value <= 10 else 3.0 for value in x],
        }
    )


def _imputer() -> LightGbmImputer:
    return LightGbmImputer(["x", "level"], ["side", "colour"])


def test_lightgbm_learns_each_column_from_the_others_with_a_neighbour_hidden() -> None:
    """The hgb test, with LightGBM. Worked by hand: at ``x = 2`` the rule says
    ``level = -3`` and ``side = "low"``, at ``x = 18`` it says 3 and ``"high"``. The
    training fold is complete and ``side`` is hidden beside ``level``: a booster that
    never saw ``side`` missing would send the gap down its larger branch and answer -3 at
    ``x = 18``. The recipe's hidden training copy is what teaches it to read ``x``."""
    train = _step_frame(420)
    test = pd.DataFrame(
        {"side": ["high", "low"], "x": [2.0, 18.0], "colour": ["red", "blue"], "level": [99.0, -99.0]}
    )
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[:, test.columns.get_loc("level")] = True
    hidden[:, test.columns.get_loc("side")] = True

    imputer = _imputer()
    imputer.fit(train)
    filled = imputer.impute(test, hidden)

    assert imputer.name == "lgbm"
    assert filled.columns.tolist() == test.columns.tolist()
    assert filled["level"].tolist() == pytest.approx([-3.0, 3.0], abs=0.5)
    assert filled["side"].tolist() == ["low", "high"]
    assert filled["x"].tolist() == [2.0, 18.0]


def test_lightgbm_fills_the_same_way_whatever_the_thread_count(monkeypatch) -> None:
    """The cache keys an entry on the machine, not its cores: two fits of one fold must
    fill alike at any thread count."""
    train = _step_frame(2_100)
    train.loc[train.index % 7 == 0, "level"] = np.nan
    test = _step_frame(63)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[::3, test.columns.get_loc("level")] = True
    hidden[1::4, test.columns.get_loc("side")] = True

    fills = []
    for threads in (1, 8, 8):
        monkeypatch.setattr(lightgbm_baseline, "THREADS", threads)
        imputer = _imputer()
        imputer.fit(train)
        fills.append(imputer.impute(test, hidden))

    pd.testing.assert_frame_equal(fills[0], fills[1])
    pd.testing.assert_frame_equal(fills[1], fills[2])


def test_lightgbm_fills_bitwise_alike_every_time_it_is_asked() -> None:
    """The check missForest's threaded forests failed (ADR 0017): on a noisy target every
    leaf value is inexact, so a prediction summed in a varying order would move."""
    stream = np.random.default_rng(1)
    train = _step_frame(840)
    train["level"] = train["level"] + stream.normal(0, 1, len(train))
    test = _step_frame(420)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[:, test.columns.get_loc("level")] = True

    imputer = _imputer()
    imputer.fit(train)
    fills = [imputer.impute(test, hidden)["level"].to_numpy() for _ in range(5)]

    assert all(np.array_equal(fills[0], other) for other in fills[1:])


def test_lightgbm_leaves_every_global_random_stream_untouched() -> None:
    train = _step_frame(420)
    test = _step_frame(21)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[::2, :] = True

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    imputer = _imputer()
    imputer.fit(train)
    imputer.impute(test, hidden)

    assert (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist()) == expected


def test_lightgbm_falls_back_to_a_constant_where_there_is_nothing_to_learn() -> None:
    train = _step_frame(60)
    train["colour"] = "red"
    train["level"] = np.nan
    test = _step_frame(9)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[:, test.columns.get_loc("colour")] = True
    hidden[:, test.columns.get_loc("level")] = True

    imputer = _imputer()
    imputer.fit(train)
    filled = imputer.impute(test, hidden)

    assert set(filled["colour"]) == {"red"}
    assert filled["level"].tolist() == [0.0] * 9


def test_the_lightgbm_family_is_lgbm_alone() -> None:
    imputers = fit_lightgbm_imputers(_step_frame(84), ["x", "level"], ["side", "colour"])

    assert [imputer.name for imputer in imputers] == ["lgbm"]
