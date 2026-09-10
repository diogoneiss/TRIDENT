"""Behaviour of the shared TabularEmbedder, used by both tasks."""

import numpy as np
import pandas as pd
import torch

from src.embedder import TabularEmbedder


def _embedder(frame: pd.DataFrame) -> TabularEmbedder:
    return TabularEmbedder(frame, ["colour"], [], dimensao=4, hidden_dim=2)


def test_a_missing_category_is_one_category_however_it_was_written() -> None:
    """``None`` and ``NaN`` are the same absence, so they share one vocabulary entry.

    Reading a CSV only ever yields ``NaN``, so this is invisible on a real run; it
    matters for frames built in code, where the two would otherwise become two
    different categories with two different embeddings.
    """
    written_as_none = pd.DataFrame({"colour": ["red", "blue", None]})
    written_as_nan = pd.DataFrame({"colour": ["red", "blue", np.nan]})

    none_embedder = _embedder(written_as_none)
    nan_embedder = _embedder(written_as_nan)

    assert list(none_embedder.label_encoders["colour"].classes_) == [
        "[MASK]",
        "[NULL]",
        "blue",
        "nan",
        "red",
    ]
    assert list(nan_embedder.label_encoders["colour"].classes_) == [
        "[MASK]",
        "[NULL]",
        "blue",
        "nan",
        "red",
    ]
    device = torch.device("cpu")
    assert none_embedder.encode(written_as_none, device).cat_indices.tolist() == (
        nan_embedder.encode(written_as_nan, device).cat_indices.tolist()
    )
