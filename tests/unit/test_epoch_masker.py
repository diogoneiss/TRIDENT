"""Re-masking an encoded table every epoch without going back through pandas.

The training loops draw a fresh mask every epoch. Drawing it with exactly the calls
``preprocess_table`` makes and applying it to the tensors of the table encoded once must give,
tensor for tensor, what encoding the masked DataFrame gives, and leave the random stream where
the old path left it: every seeded result depends on that stream.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.embedder import EpochMasker, TabularEmbedder
from src.utils import preprocess_table

CATEGORICAL = ["colour", "shape"]
NUMERICAL = ["size", "weight"]


def _frame(rows: int = 60) -> pd.DataFrame:
    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "size": (index % 7).astype(float) - 3.0,
            "colour": np.array(["red", "blue", "green", "red"], dtype=object)[index % 4],
            "weight": (index % 5).astype(float) / 4,
            "shape": np.array(["round", "square", "star"], dtype=object)[index % 3],
        }
    )
    frame.loc[index % 6 == 0, "colour"] = np.nan
    frame.loc[index % 9 == 0, "size"] = np.nan
    frame.loc[index % 11 == 0, "weight"] = np.nan
    frame.loc[5, ["size", "colour", "weight", "shape"]] = np.nan  # a row with nothing to hide
    return frame


def _tensors(table) -> dict[str, torch.Tensor]:
    return {
        "cat_indices": table.cat_indices,
        "num_values": table.num_values,
        "mask_flags": table.mask_flags,
        "null_flags": table.null_flags,
        "masked_positions": table.masked_positions,
    }


@pytest.mark.parametrize("rate", [0.02, 0.3, 0.6])
def test_an_epoch_mask_on_the_tensors_is_the_masked_frame_encoded(rate: float) -> None:
    frame = _frame()
    torch.manual_seed(0)
    embedder = TabularEmbedder(frame, CATEGORICAL, NUMERICAL, dimensao=8, hidden_dim=4)
    masker = EpochMasker(embedder, frame, torch.device("cpu"))

    for seed in range(4):
        np.random.seed(seed)
        expected = embedder.encode(preprocess_table(frame.copy(), p_base=rate, fine_tunning=False), "cpu")
        state_after_frame = np.random.get_state()
        np.random.seed(seed)
        drawn = masker.draw(rate)
        state_after_tensors = np.random.get_state()

        for name, tensor in _tensors(expected).items():
            got = _tensors(drawn)[name]
            assert got.dtype == tensor.dtype and got.shape == tensor.shape, name
            assert torch.equal(got, tensor), (seed, name)
        assert all(np.array_equal(a, b) for a, b in zip(state_after_frame[1:3], state_after_tensors[1:3]))
        assert state_after_frame[3:] == state_after_tensors[3:]


def test_showing_the_gaps_as_mask_is_the_frame_with_its_gaps_masked_encoded() -> None:
    """ADR 0004 shows a real gap to the model as ``[NULL]``; the decode stage may instead show
    it as ``[MASK]``, the token the induced scoring uses (T02 step 2). The helper turns every
    gap of an encoded table into its column's ``[MASK]`` id or mask flag and touches nothing
    else: the cells to reconstruct stay the ones the epoch's mask chose, because a gap has no
    truth to learn from."""
    frame = _frame()
    embedder = TabularEmbedder(frame, CATEGORICAL, NUMERICAL, dimensao=8, hidden_dim=4)
    device = torch.device("cpu")
    np.random.seed(3)
    hidden = EpochMasker(embedder, frame, device).draw(0.3)

    shown = embedder.gaps_as_mask(hidden)

    # The same table, with every gap spelt [MASK] before encoding, is the reference. Its
    # masked_positions would also mark the gaps, which is exactly what the helper must not do.
    as_mask = frame.copy()
    for column in frame.columns:
        as_mask[column] = frame[column].astype(object)
    np.random.seed(3)
    hidden_frame = preprocess_table(frame.copy(), p_base=0.3)
    hidden_frame[frame.isna() & (hidden_frame != "[MASK]")] = "[MASK]"
    reference = embedder.encode(hidden_frame, device)
    for name in ("cat_indices", "mask_flags", "null_flags"):
        assert torch.equal(getattr(shown, name), getattr(reference, name)), name
    assert torch.equal(shown.masked_positions, hidden.masked_positions)
    assert not torch.equal(shown.masked_positions, reference.masked_positions)
    assert not shown.null_flags.any()
    # Numbers the model can see are untouched; the flagged ones are not read.
    visible = ~(shown.mask_flags | hidden.null_flags)
    assert torch.equal(shown.num_values[visible], hidden.num_values[visible])
