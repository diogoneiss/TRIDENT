"""The missForest baseline imputers, with random forests and with LightGBM (ADR 0016)."""

import random

import numpy as np
import pandas as pd
import pytest
import torch

from src.training import missforest
from src.training.missforest import (
    MissForestImputer,
    RoundChange,
    fit_missforest_imputers,
    still_converging,
)


@pytest.mark.parametrize(
    ("previous", "latest", "expected"),
    [
        # Both kinds changed less than in the round before: keep going.
        (RoundChange(0.20, 0.10), RoundChange(0.05, 0.04), True),
        # One kind still settling is enough to go on.
        (RoundChange(0.20, 0.10), RoundChange(0.25, 0.04), True),
        (RoundChange(0.20, 0.10), RoundChange(0.05, 0.12), True),
        # Both grew for the first time: stop, and the round before is the answer.
        (RoundChange(0.20, 0.10), RoundChange(0.21, 0.11), False),
        # A change no smaller than the last is no improvement.
        (RoundChange(0.20, 0.10), RoundChange(0.20, 0.10), False),
        # A kind with no gap to fill has no change to compare; the other decides alone.
        (RoundChange(0.20, None), RoundChange(0.30, None), False),
        (RoundChange(None, 0.10), RoundChange(None, 0.05), True),
        # Nothing to fill at all: there is nothing left to converge.
        (RoundChange(None, None), RoundChange(None, None), False),
    ],
)
def test_iteration_stops_once_neither_kind_of_column_changes_less_than_before(
    previous: RoundChange, latest: RoundChange, expected: bool
) -> None:
    """Stekhoven and Bühlmann's rule (2012, and ``missForest``'s ``stopCriterion`` in R):
    iterate while the numerical or the categorical change shrank from one round to the
    next, and stop the first time neither did. Stopping only once both grew, or as soon
    as either did, would return a different round and look just as plausible."""
    assert still_converging(previous, latest) is expected


LEARNERS = ["random_forest", "lightgbm"]


def _step_frame(rows: int) -> pd.DataFrame:
    """``level`` jumps from -3 to 3 where ``x`` passes 10, and ``side`` follows the same
    threshold, with an unrelated ``colour`` in between: a rule a per-column model finds
    exactly. The columns interleave the two kinds, so a fill read from the wrong column
    lands on the wrong answer."""
    x = [float(index % 21) for index in range(rows)]
    return pd.DataFrame(
        {
            "side": ["low" if value <= 10 else "high" for value in x],
            "x": x,
            "colour": [["red", "blue", "green"][index % 3] for index in range(rows)],
            "level": [-3.0 if value <= 10 else 3.0 for value in x],
        }
    )


def _imputer(learner: str) -> MissForestImputer:
    return MissForestImputer(["x", "level"], ["side", "colour"], learner=learner)


@pytest.mark.parametrize("learner", LEARNERS)
def test_missforest_learns_each_column_from_the_others(learner: str) -> None:
    """Worked by hand: at ``x = 2`` the rule says ``level = -3``, and where ``level`` is 3
    it says ``side = "high"``. The training fold is complete, as on any ``_00nan``
    variant, so no column has a gap to fill while fitting: canonical missForest would fit
    no model at all, and a test cell the evaluation mask hides would get its column's
    mean or mode (0 and whichever side is commoner)."""
    train = _step_frame(420)
    test = pd.DataFrame(
        {"side": ["low", "low"], "x": [2.0, 18.0], "colour": ["red", "blue"], "level": [99.0, 3.0]}
    )
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[0, test.columns.get_loc("level")] = True
    hidden[1, test.columns.get_loc("side")] = True

    imputer = _imputer(learner)
    imputer.fit(train)
    filled = imputer.impute(test, hidden)

    assert imputer.name == {"random_forest": "missforest", "lightgbm": "missforest_lgbm"}[learner]
    assert filled.columns.tolist() == test.columns.tolist()
    assert filled["level"].tolist() == pytest.approx([-3.0, 3.0], abs=0.1)
    assert filled["side"].tolist() == ["low", "high"]
    assert filled["x"].tolist() == [2.0, 18.0]


def _gappy(frame: pd.DataFrame, share: float, seed: int = 3) -> pd.DataFrame:
    """The frame with about ``share`` of every column's cells missing, as on a ``_40nan``
    variant's training fold."""
    stream = np.random.default_rng(seed)
    return frame.mask(stream.random(frame.shape) < share)


@pytest.mark.parametrize("learner", LEARNERS)
def test_missforest_learns_the_rule_from_a_training_fold_full_of_gaps(learner: str) -> None:
    """Where a third of every training column is missing, the rounds have gaps to fill
    and the models of the round kept are trained on fills rather than on the truth;
    the rule is still the one found."""
    train = _gappy(_step_frame(840), 0.33)
    test = _step_frame(42)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[::2, test.columns.get_loc("level")] = True
    hidden[1::2, test.columns.get_loc("side")] = True

    imputer = _imputer(learner)
    imputer.fit(train)
    filled = imputer.impute(test, hidden)

    assert imputer.rounds >= 2
    assert filled["level"][::2].tolist() == pytest.approx(test["level"][::2].tolist(), abs=0.5)
    assert filled["side"][1::2].tolist() == test["side"][1::2].tolist()


@pytest.mark.parametrize("learner", LEARNERS)
def test_missforest_never_reads_the_truth_of_the_cell_it_fills(learner: str) -> None:
    """The scored frame still holds the truth under every hidden cell. A fill that read
    it would make the bar unbeatable and look like a strong baseline."""
    train = _gappy(_step_frame(420), 0.2)
    test = _step_frame(30)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[::3, test.columns.get_loc("level")] = True
    hidden[1::3, test.columns.get_loc("side")] = True
    tampered = test.copy()
    tampered.loc[hidden[:, test.columns.get_loc("level")], "level"] = 1e6
    tampered.loc[hidden[:, test.columns.get_loc("side")], "side"] = "low"

    imputer = _imputer(learner)
    imputer.fit(train)

    pd.testing.assert_frame_equal(
        imputer.impute(test, hidden).where(hidden), imputer.impute(tampered, hidden).where(hidden)
    )


@pytest.mark.parametrize("learner", LEARNERS)
def test_missforest_fills_the_same_way_whatever_the_thread_count(learner: str, monkeypatch) -> None:
    """The cache keys an entry on the machine, not on its cores, and gorgona8 and the
    Windows PC have different numbers of them: two fits of one fold must fill alike at
    any thread count, or a cached bar would be some other computation's."""
    train = _gappy(_step_frame(2_100), 0.3)
    test = _gappy(_step_frame(63), 0.3, seed=5)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[::4, :] = True

    fills = []
    for threads in (1, 8, 8):
        monkeypatch.setattr(missforest, "THREADS", threads)
        imputer = _imputer(learner)
        imputer.fit(train)
        fills.append(imputer.impute(test, hidden))

    pd.testing.assert_frame_equal(fills[0], fills[1])
    pd.testing.assert_frame_equal(fills[1], fills[2])


@pytest.mark.parametrize("learner", LEARNERS)
def test_missforest_leaves_every_global_random_stream_untouched(learner: str) -> None:
    """Every seeded result depends on the global draw sequences, and the regression
    fixtures pin them; a baseline that consumed from one would move numbers that have
    nothing to do with it."""
    train = _gappy(_step_frame(420), 0.2)
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

    imputer = _imputer(learner)
    imputer.fit(train)
    imputer.impute(test, hidden)

    assert (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist()) == expected


@pytest.mark.parametrize("learner", LEARNERS)
def test_missforest_falls_back_to_a_constant_where_there_is_nothing_to_learn(learner: str) -> None:
    """A column the training fold holds in one category only, or not at all, gives a
    model nothing to fit. The fill is then that category, or the scaled mean of zero,
    never a crash or a NaN that would abort the run."""
    train = _step_frame(60)
    train["colour"] = "red"
    train["level"] = np.nan
    test = _step_frame(9)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[:, test.columns.get_loc("colour")] = True
    hidden[:, test.columns.get_loc("level")] = True

    imputer = _imputer(learner)
    imputer.fit(train)
    filled = imputer.impute(test, hidden)

    assert set(filled["colour"]) == {"red"}
    assert filled["level"].tolist() == [0.0] * 9


def test_an_all_missing_training_categorical_column_is_refused_clearly() -> None:
    train = _step_frame(30)
    train["side"] = np.nan

    with pytest.raises(ValueError, match="'side' has no observed category"):
        _imputer("random_forest").fit(train)


def test_the_missforest_baselines_are_the_forest_and_lightgbm() -> None:
    imputers = fit_missforest_imputers(_step_frame(84), ["x", "level"], ["side", "colour"])

    assert [imputer.name for imputer in imputers] == ["missforest", "missforest_lgbm"]
