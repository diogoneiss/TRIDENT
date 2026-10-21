"""The TabPFN v2 wrapper: column-wise, inductive, hidden cells kept out of the features.

The estimators are a duck-typed fake, so these tests need neither the ``tabpfn`` extra nor its
weights; the fake records what the wrapper hands it.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.training.tabpfn_baseline import TabPFNImputer


class _Fake:
    def __init__(self, kind: str, categorical: list[int], log: list["_Fake"]) -> None:
        self.kind, self.categorical = kind, categorical
        self.fit_x: np.ndarray = np.empty(0)
        self.fit_y: np.ndarray = np.empty(0)
        self.predicted: list[np.ndarray] = []
        log.append(self)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_Fake":
        self.fit_x, self.fit_y = x, y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.predicted.append(x)
        return np.full(len(x), 1 if self.kind == "classifier" else 7.0)


def _factory(log: list[_Fake]) -> Any:
    return lambda kind, categorical: _Fake(kind, categorical, log)


def _mixed(rows: int = 60) -> pd.DataFrame:
    stream = np.random.default_rng(0)
    frame = pd.DataFrame({
        "a": stream.normal(size=rows),
        "b": stream.normal(size=rows),
        "x": stream.choice(["blue", "red"], size=rows),
    })
    frame.loc[[3, 9], "a"] = np.nan
    return frame


def test_fills_only_the_hidden_cells_with_the_estimator_of_their_column() -> None:
    frame, log = _mixed(), []
    imputer = TabPFNImputer(["a", "b"], ["x"], factory=_factory(log))
    before = frame.copy()
    imputer.fit(frame)
    hidden = np.zeros(frame.shape, dtype=bool)
    hidden[[0, 1], 1] = True
    hidden[[2], 2] = True
    filled = imputer.impute(frame, hidden)
    assert frame.equals(before), "the frame is never rewritten"
    assert list(filled.columns) == list(frame.columns)
    assert filled.loc[[0, 1], "b"].tolist() == [7.0, 7.0]
    assert filled.loc[2, "x"] == "red", "class 1 of the sorted training vocabulary"
    mean_a = float(frame["a"].mean())
    np.testing.assert_allclose(filled.loc[[3, 9], "a"].to_numpy(dtype=float), [mean_a, mean_a], err_msg="a gap nobody asked for keeps the column's constant")
    untouched = ~(hidden | frame.isna().to_numpy())
    np.testing.assert_array_equal(filled[["a", "b"]].to_numpy(dtype=float)[untouched[:, :2]], frame[["a", "b"]].to_numpy(dtype=float)[untouched[:, :2]])
    assert [f.kind for f in log] == ["regressor", "regressor", "classifier"]


def test_hidden_cells_enter_the_features_as_gaps_and_categories_as_indices() -> None:
    frame, log = _mixed(), []
    imputer = TabPFNImputer(["a", "b"], ["x"], factory=_factory(log))
    imputer.fit(frame)
    hidden = np.zeros(frame.shape, dtype=bool)
    hidden[0, 0] = True  # row 0's ``a`` is asked for
    hidden[0, 1] = True  # and its ``b`` is hidden too, so ``a``'s estimator must not see it
    imputer.impute(frame, hidden)
    by_target = {"a": log[0], "b": log[1], "x": log[2]}
    asked = by_target["a"].predicted[0]
    # ``a``'s features are b and x, in that order; b is hidden in row 0.
    assert np.isnan(asked[0, 0])
    assert asked[0, 1] == float(["blue", "red"].index(frame.loc[0, "x"]))
    assert by_target["a"].categorical == [1], "x is the second of a's features"
    assert by_target["x"].categorical == [], "x's own features are all numbers"
    assert not np.isnan(by_target["a"].fit_y).any(), "the context holds observed targets only"
    assert len(by_target["a"].fit_y) == 58


def test_the_context_is_capped_by_a_fixed_subsample() -> None:
    frame = _mixed(200)
    sizes = []
    for _ in range(2):
        log: list[_Fake] = []
        imputer = TabPFNImputer(["a", "b"], ["x"], factory=_factory(log), context_rows=50)
        imputer.fit(frame)
        sizes.append([f.fit_x.copy() for f in log])
    assert all(len(x) == 50 for x in sizes[0])
    for first, second in zip(*sizes):
        np.testing.assert_array_equal(first, second)


def test_a_one_class_column_is_filled_with_its_class_and_the_global_stream_is_untouched() -> None:
    frame = _mixed()
    frame["x"] = "blue"
    log: list[_Fake] = []
    imputer = TabPFNImputer(["a", "b"], ["x"], factory=_factory(log))
    np.random.seed(5)
    state = np.random.get_state()[1].copy()
    imputer.fit(frame)
    hidden = np.zeros(frame.shape, dtype=bool)
    hidden[4, 2] = True
    filled = imputer.impute(frame, hidden)
    assert filled.loc[4, "x"] == "blue"
    assert [f.kind for f in log] == ["regressor", "regressor"], "no estimator for a one-class column"
    np.testing.assert_array_equal(np.random.get_state()[1], state)


def test_the_frames_own_gaps_ride_along_so_the_induced_call_needs_no_pass_of_its_own() -> None:
    frame, log = _mixed(), []
    imputer = TabPFNImputer(["a", "b"], ["x"], factory=_factory(log))
    imputer.fit(frame)
    masked = np.zeros(frame.shape, dtype=bool)
    masked[[0, 1], 1] = True
    first = imputer.impute(frame, masked)
    calls = [len(f.predicted) for f in log]
    induced = frame.isna().to_numpy()
    second = imputer.impute(frame, induced)
    assert [len(f.predicted) for f in log] == calls, "the induced view was predicted with the masked one"
    assert calls == [1, 1, 0], "one predict per column that has anything to fill; x has nothing"
    assert second.loc[[3, 9], "a"].tolist() == [7.0, 7.0]
    assert first.loc[[0, 1], "b"].tolist() == [7.0, 7.0]
    # The stacked predict held a's gap rows under the gaps-only view: b observed there.
    asked_a = log[0].predicted[0]
    assert len(asked_a) == 2 and not np.isnan(asked_a[:, 0]).any()
