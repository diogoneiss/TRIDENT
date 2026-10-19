"""The vendored MaCoDE baseline: inductive, any column mix, its own random streams."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.imputation_baselines import score_baselines
from src.training.imputation_metrics import NUMERICAL, mean_mode_baselines
from src.training.macode import MacodeImputer, fit_macode_imputers

CPU = torch.device("cpu")


def _numerical(rows: int = 240) -> pd.DataFrame:
    stream = np.random.default_rng(0)
    a = stream.normal(size=rows)
    return pd.DataFrame({"a": a, "b": a + 0.05 * stream.normal(size=rows), "c": stream.normal(size=rows)})


def _categorical(rows: int = 240) -> pd.DataFrame:
    stream = np.random.default_rng(1)
    x = stream.choice(["red", "blue", "green"], size=rows)
    return pd.DataFrame({"x": x, "y": x, "z": stream.choice(["s", "t"], size=rows)})


def test_a_table_of_one_column_type_fits_and_fills_only_the_gaps() -> None:
    """The released embedding stacked an empty list on a table without both kinds of column."""
    for frame, numerical, categorical in ((_numerical(), ["a", "b", "c"], []), (_categorical(), [], ["x", "y", "z"])):
        imputer = MacodeImputer(numerical, categorical, epochs=2, device=CPU)
        before = frame.copy()
        imputer.fit(frame)
        hidden = np.zeros(frame.shape, dtype=bool)
        hidden[::7, 1] = True
        filled = imputer.impute(frame, hidden)
        assert frame.equals(before), "the frame is never rewritten"
        assert list(filled.columns) == list(frame.columns)
        kept = ~hidden
        if numerical:
            np.testing.assert_array_equal(filled.to_numpy(dtype=float)[kept], frame.to_numpy(dtype=float)[kept])
            assert not np.isnan(filled.to_numpy(dtype=float)).any()
        else:
            assert (filled.to_numpy()[kept] == frame.to_numpy()[kept]).all()
            assert set(filled["y"]) <= {"red", "blue", "green"}


def test_the_bins_reach_the_data_and_each_bin_stands_for_its_training_mean() -> None:
    """The released code cut the data into 10 bins while the model was built for 50."""
    frame = _numerical(1000)
    imputer = MacodeImputer(["a", "b", "c"], [], epochs=1, bins=50, device=CPU)
    imputer.fit(frame)
    codes = imputer._codes(frame, np.zeros(frame.shape, dtype=bool))
    assert codes.min() == 1 and codes.max() == 50
    assert len(np.unique(codes[:, 0])) == 50
    values = frame["a"].to_numpy()
    means = imputer._columns["a"].bin_values
    for b in (1, 25, 50):
        assert means[b - 1] == pytest.approx(values[codes[:, 0] == b].mean())


def test_training_leaves_the_global_streams_alone_and_repeats_itself() -> None:
    frame = _numerical()
    torch.manual_seed(123)
    np.random.seed(123)
    torch_state, numpy_state = torch.random.get_rng_state(), np.random.get_state()
    hidden = np.zeros(frame.shape, dtype=bool)
    hidden[::5, 1] = True
    fills = []
    for _ in range(2):
        imputer = MacodeImputer(["a", "b", "c"], [], epochs=3, device=CPU)
        imputer.fit(frame)
        fills.append(imputer.impute(frame, hidden))
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert np.random.get_state()[1].tolist() == numpy_state[1].tolist()
    assert fills[0].equals(fills[1])


def test_with_a_twin_column_it_beats_the_mean_fill() -> None:
    """b is a with a little noise: after some training, MaCoDE fills b from a far better than the
    column mean, on rows it never saw."""
    train, test = _numerical(600), _numerical(200).sample(frac=1.0, random_state=3).reset_index(drop=True)
    imputers = fit_macode_imputers(train, ["a", "b", "c"], [], epochs=60)
    hidden = np.zeros(test.shape, dtype=bool)
    hidden[:, 1] = True
    cells = pd.DataFrame(
        {"row": np.arange(len(test)), "column": "b", "kind": NUMERICAL, "population": "masked",
         "actual": test["b"].to_numpy(), "imputed": 0.0, "confidence": np.nan}
    )
    score = score_baselines(imputers, test, cells, mean_mode_baselines(train, ["a", "b", "c"], [])).metrics
    assert score["baseline/macode/impute_score"] < 0.5
