"""C1's arm T (``PRETRAIN_HOLE_TOKEN null``): holes shown as [NULL], the loss record untouched, the gate on the flag."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.embedder import EpochMasker, TabularEmbedder
from src.training.types import Hyperparameters


def _frame(rows: int = 80) -> pd.DataFrame:
    stream = np.random.default_rng(0)
    frame = pd.DataFrame({"c": stream.choice(["x", "y", "z"], rows), "a": stream.normal(size=rows), "b": stream.normal(size=rows)})
    frame.loc[frame.index[::3], "a"] = np.nan
    frame.loc[frame.index[1::4], "c"] = np.nan
    return frame


def test_t_shows_every_hole_as_null_and_keeps_the_hole_record() -> None:
    frame = _frame()
    embedder = TabularEmbedder(frame, ["c"], ["a", "b"])
    np.random.seed(3)
    s = EpochMasker(embedder, frame, torch.device("cpu")).draw(0.3)
    t = embedder.holes_as_null(s)
    assert torch.equal(t.masked_positions, s.masked_positions) and s.masked_positions.any()
    assert not t.mask_flags.any()
    assert torch.equal(t.null_flags, s.null_flags | s.mask_flags)
    assert torch.equal(t.num_values, s.num_values)
    null_id, mask_id = (int(embedder.label_encoders["c"].transform([tok])[0]) for tok in ("[NULL]", "[MASK]"))
    assert (s.cat_indices == mask_id).any() and not (t.cat_indices == mask_id).any()
    assert torch.equal(t.cat_indices == null_id, (s.cat_indices == null_id) | (s.cat_indices == mask_id))


def test_t_draws_the_same_holes_as_s() -> None:
    frame = _frame()
    embedder = TabularEmbedder(frame, ["c"], ["a", "b"])
    np.random.seed(5)
    s = EpochMasker(embedder, frame, torch.device("cpu")).draw(0.3)
    np.random.seed(5)
    t = embedder.holes_as_null(EpochMasker(embedder, frame, torch.device("cpu")).draw(0.3))
    assert torch.equal(s.masked_positions, t.masked_positions)


def test_the_flag_is_gated() -> None:
    assert Hyperparameters.from_mapping({}).pretrain_hole_token == "mask"
    assert Hyperparameters.from_mapping({"PRETRAIN_HOLE_TOKEN": "null"}).pretrain_hole_token == "null"
    with pytest.raises(ValueError):
        Hyperparameters.from_mapping({"PRETRAIN_HOLE_TOKEN": "cls"})
    with pytest.raises(ValueError):
        Hyperparameters.from_mapping({"PRETRAIN_HOLE_TOKEN": "null", "PRETRAIN_GAP_TOKEN": "mask"})
