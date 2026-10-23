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


def _mixed_frame(rows: int = 40) -> pd.DataFrame:
    stream = np.random.default_rng(0)
    frame = pd.DataFrame({
        "a": stream.choice(["x", "y", "z"], size=rows),
        "b": stream.choice(["p", "q"], size=rows),
        "c": stream.choice(["k1", "k2", "k3", "k4"], size=rows),
        "n": stream.normal(size=rows),
    })
    frame.loc[[2, 7], "a"] = np.nan
    return frame


def test_the_batched_lookup_gives_the_per_column_values_and_gradients_to_rounding() -> None:
    """T06. ``EMBEDDING_LOOKUP batched`` reads the categorical tables in one gather. The forward
    values are the per-column ones exactly, the parameters and state dict are the same, and the
    gradients agree to float rounding; the seeded initialisation does not move."""
    frame = _mixed_frame()
    built = []
    for batched in (False, True):
        torch.manual_seed(0)
        built.append(TabularEmbedder(frame, ["a", "b", "c"], ["n"], dimensao=8, hidden_dim=4, batched_lookup=batched))
    per_column, batched = built
    assert per_column.state_dict().keys() == batched.state_dict().keys()
    for name, value in per_column.state_dict().items():
        assert torch.equal(value, batched.state_dict()[name]), name
    encoded = per_column.encode(frame, torch.device("cpu"))
    outputs = [embedder(encoded) for embedder in built]
    assert torch.equal(outputs[0], outputs[1])
    weights = torch.randn_like(outputs[0])
    for output in outputs:
        (output * weights).sum().backward()
    for (name, a), (_, b) in zip(per_column.named_parameters(), batched.named_parameters()):
        assert a.grad is not None and b.grad is not None, name
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)
