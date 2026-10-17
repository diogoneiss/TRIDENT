"""The linear and low-rank baseline imputers of E41: condexp, inductive SoftImpute and ICE."""

import random

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from src.training.linear_baselines import (
    ConditionalExpectationImputer,
    IceImputer,
    SequentialRoundsImputer,
    SoftImputeImputer,
    fit_linear_imputers,
)

NUMERICAL = ["a", "b", "c", "d"]


def _correlated(rows: int, seed: int = 0) -> pd.DataFrame:
    """Four numbers driven by two hidden factors, plus a category that follows the first one."""
    stream = np.random.default_rng(seed)
    first, second = stream.normal(size=rows), stream.normal(size=rows)
    noise = stream.normal(scale=0.1, size=(rows, 4))
    return pd.DataFrame(
        {
            "a": first + noise[:, 0],
            "kind": np.where(first > 0, "up", "down"),
            "b": 2 * first + noise[:, 1],
            "c": second + noise[:, 2],
            "d": first - second + noise[:, 3],
        }
    )


def _gappy(frame: pd.DataFrame, share: float, seed: int = 3) -> pd.DataFrame:
    return frame.mask(np.random.default_rng(seed).random(frame.shape) < share)


def _hide(frame: pd.DataFrame, columns: list[str], every: int = 3) -> np.ndarray:
    hidden = np.zeros(frame.shape, dtype=bool)
    for offset, column in enumerate(columns):
        hidden[offset::every, frame.columns.get_loc(column)] = True
    return hidden


@pytest.mark.parametrize("make", [ConditionalExpectationImputer, SoftImputeImputer, IceImputer, SequentialRoundsImputer])
def test_no_linear_imputer_reads_the_truth_of_the_cell_it_fills(make) -> None:
    """The scored frame still holds the truth under every hidden cell (ADR 0007's protocol hides
    by a mask, not by NaN). A fill that read it would set an unbeatable bar."""
    imputer = make(NUMERICAL, ["kind"])
    imputer.fit(_gappy(_correlated(300), 0.3))
    test = _correlated(30, seed=1)
    hidden = _hide(test, ["b", "kind", "d"])
    tampered = test.copy()
    for column in ("b", "d"):
        tampered.loc[hidden[:, test.columns.get_loc(column)], column] = 1e6
    tampered.loc[hidden[:, test.columns.get_loc("kind")], "kind"] = "up"

    assert imputer.impute(test, hidden).equals(imputer.impute(tampered, hidden))


@pytest.mark.parametrize("make", [ConditionalExpectationImputer, SoftImputeImputer, IceImputer])
def test_numerical_fills_stay_inside_the_training_range_and_beat_the_mean(make) -> None:
    """Every numerical fill is clipped to the column's training range (the pre-registered clip),
    and on numbers this correlated a linear fill is closer to the truth than the column mean."""
    train = _gappy(_correlated(400), 0.2)
    test = _correlated(60, seed=1)
    test.loc[0, "a"] = 50.0  # an observed outlier pushes an unclipped fill of b far out
    hidden = _hide(test, ["b"], every=1)
    imputer = make(NUMERICAL, ["kind"])
    imputer.fit(train)
    filled = imputer.impute(test, hidden)

    low, high = train["b"].min(), train["b"].max()
    assert filled["b"].between(low, high).all()
    truth = test["b"].to_numpy()[1:]
    error = np.sqrt(np.mean((filled["b"].to_numpy()[1:] - truth) ** 2))
    naive = np.sqrt(np.mean((train["b"].mean() - truth) ** 2))
    assert error < 0.5 * naive
    assert (filled.drop(columns="b") == test.drop(columns="b")).all().all()


def test_condexp_fills_a_row_with_nothing_observed_with_the_training_means() -> None:
    train = _gappy(_correlated(200), 0.1)
    test = _correlated(4, seed=1)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[0, [test.columns.get_loc(column) for column in NUMERICAL]] = True
    imputer = ConditionalExpectationImputer(NUMERICAL, ["kind"])
    imputer.fit(train)
    filled = imputer.impute(test, hidden)
    for column in NUMERICAL:
        assert filled.loc[0, column] == pytest.approx(train[column].mean())


def test_softimpute_may_keep_more_than_half_the_columns() -> None:
    """The probe's p // 2 cap bound on spambase and broke the arm. With three comparable factors
    in four columns the 90% energy rule asks for rank 3, above 4 // 2."""
    stream = np.random.default_rng(0)
    factors = stream.normal(size=(500, 3))
    frame = pd.DataFrame(
        np.column_stack([factors[:, 0], factors[:, 1], factors[:, 2], factors.sum(axis=1)])
        + stream.normal(scale=0.05, size=(500, 4)),
        columns=NUMERICAL,
    )
    imputer = SoftImputeImputer(NUMERICAL, [])
    imputer.fit(frame)
    assert imputer.rank == 3
    assert imputer.energy >= 0.9


def test_ice_fills_as_scikit_learns_iterative_imputer_does() -> None:
    """On a table of numbers, ICE is IterativeImputer(BayesianRidge) with the same round count,
    ascending order and the training range as its bounds. missForest's own fill, which replays the
    kept round's models from the means, would not match: it is the defect this arm avoids."""
    train = _gappy(_correlated(300).drop(columns="kind"), 0.3)
    test = _gappy(_correlated(40, seed=1).drop(columns="kind"), 0.3, seed=5)
    ice = IceImputer(NUMERICAL, [])
    ice.fit(train)
    reference = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=ice.rounds,
        tol=0.0,
        imputation_order="ascending",
        initial_strategy="mean",
        min_value=train.min().to_numpy(),
        max_value=train.max().to_numpy(),
    ).fit(train.to_numpy())

    hidden = np.zeros(test.shape, dtype=bool)
    ours = ice.impute(test, hidden).to_numpy()
    theirs = reference.transform(test.to_numpy())

    assert ice.rounds >= 2
    np.testing.assert_allclose(ours, theirs, atol=1e-6)


def test_ice_learns_a_category_from_one_hot_features() -> None:
    train = _gappy(_correlated(400), 0.2)
    test = _correlated(60, seed=1)
    hidden = _hide(test, ["kind"], every=1)
    imputer = IceImputer(NUMERICAL, ["kind"])
    imputer.fit(train)
    filled = imputer.impute(test, hidden)
    assert set(filled["kind"]) <= {"up", "down"}
    assert (filled["kind"] == test["kind"]).mean() > 0.9


def test_fitting_and_filling_leave_every_global_stream_where_it_was() -> None:
    """ADR 0007 decision 6: a baseline never moves the streams the seeded model draws from."""
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    before = (random.random(), np.random.random(), torch.rand(1).item())
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    train = _gappy(_correlated(200), 0.3)
    imputers = fit_linear_imputers(train, NUMERICAL, ["kind"])
    test = _correlated(20, seed=1)
    for imputer in imputers:
        imputer.impute(test, _hide(test, ["a", "kind"]))
    after = (random.random(), np.random.random(), torch.rand(1).item())

    assert [imputer.name for imputer in imputers] == ["condexp", "softimpute", "ice"]
    assert before == after
