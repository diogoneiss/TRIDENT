"""ReMasker as a baseline: inductive, fills only the gaps, numerical only, its own random streams."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.remasker import RemaskerImputer, _remask, fit_remasker_imputers

CPU = torch.device("cpu")


def _frame(rows: int = 200) -> pd.DataFrame:
    stream = np.random.default_rng(0)
    a = stream.normal(size=rows)
    frame = pd.DataFrame({"a": a, "b": 2 * a + 0.05 * stream.normal(size=rows), "c": stream.normal(size=rows)})
    frame = frame.mask(stream.random(frame.shape) < 0.3)
    return frame


def test_fills_every_gap_and_only_the_gaps() -> None:
    frame = _frame()
    imputer = RemaskerImputer(["a", "b", "c"], epochs=2, device=CPU)
    imputer.fit(frame)
    hidden = np.zeros(frame.shape, dtype=bool)
    hidden[:10, 0] = True
    filled = imputer.impute(frame, hidden)
    assert not filled.isna().any().any()
    keep = ~hidden & frame.notna().to_numpy()
    np.testing.assert_array_equal(filled.to_numpy()[keep], frame.to_numpy()[keep])
    assert list(filled.columns) == list(frame.columns)
    assert frame.isna().any().any(), "the frame itself is never rewritten"


def test_leaves_the_global_streams_and_repeats_itself() -> None:
    frame = _frame()
    torch.manual_seed(3)
    expected = torch.rand(4)
    torch.manual_seed(3)
    fills = []
    for _ in range(2):
        imputer = RemaskerImputer(["a", "b", "c"], epochs=1, device=CPU)
        imputer.fit(frame)
        fills.append(imputer.impute(frame, np.zeros(frame.shape, dtype=bool)).to_numpy())
    assert torch.equal(torch.rand(4), expected)
    np.testing.assert_array_equal(fills[0], fills[1])


def test_a_gap_is_never_visible_and_half_the_tokens_at_most_are() -> None:
    observed = torch.rand(500, 8) < 0.4
    visible = _remask(observed)
    assert not (visible & ~observed).any()
    assert int(visible.sum(dim=1).max()) <= 4


def test_refuses_a_categorical_column() -> None:
    with pytest.raises(ValueError, match="numerical columns only"):
        fit_remasker_imputers(_frame(), ["a", "b"], ["c"], epochs=1)


def test_the_paper_rule_remasks_half_of_each_rows_observed_cells() -> None:
    observed = torch.rand(500, 8) < 0.2
    visible = _remask(observed, "paper")
    assert not (visible & ~observed).any()
    count = observed.sum(dim=1)
    assert torch.equal(visible.sum(dim=1), count // 2)
    complete = torch.ones(50, 8, dtype=torch.bool)
    assert torch.equal(_remask(complete, "paper").sum(dim=1), _remask(complete).sum(dim=1))


def test_refuses_an_unknown_rule() -> None:
    with pytest.raises(ValueError, match="remask rule"):
        RemaskerImputer(["a"], remask="half")
