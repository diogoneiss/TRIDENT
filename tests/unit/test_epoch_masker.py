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
