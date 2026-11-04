"""C15's arms: A2's fills and indicator, pre-training holes equal to A1's, the off switch."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.embedder import EpochMasker, TabularEmbedder
from src.training.c15_inputs import GAP_PREFIX, fold_dataset
from src.training.types import FoldSplit, Hyperparameters, PreparedDataset


def _dataset(rows: int = 60) -> PreparedDataset:
    stream = np.random.default_rng(0)
    frame = pd.DataFrame({"a": stream.normal(size=rows), "b": stream.normal(size=rows), "class": stream.integers(0, 2, rows)})
    frame.loc[frame.index[::3], "a"] = np.nan
    frame.loc[frame.index[1::4], "b"] = np.nan
    from pathlib import Path

    return PreparedDataset(frame=frame, label_column="class", categorical_columns=[], numerical_columns=["a", "b"],
                           label_classes=[0, 1], source_path=Path("x"), splits_path=Path("y"))


def _fold(rows: int = 60) -> FoldSplit:
    return FoldSplit(train_indices=np.arange(0, 40), validation_indices=np.arange(40, 50), test_indices=np.arange(50, rows))


def test_off_and_a1_leave_the_dataset_as_it_is() -> None:
    data = _dataset()
    assert fold_dataset(data, _fold(), "off") is data and fold_dataset(data, _fold(), "A1") is data


def test_a2_fills_every_gap_from_the_training_rows_and_adds_the_indicators() -> None:
    data = _dataset()
    filled = fold_dataset(data, _fold(), "A2").frame
    assert not filled[["a", "b"]].isna().any().any()
    for column in ("a", "b"):
        assert (filled[GAP_PREFIX + column].to_numpy() == data.frame[column].isna().to_numpy()).all()
        observed = data.frame[column].notna()
        assert np.allclose(filled.loc[observed, column], data.frame.loc[observed, column])
    assert filled["class"].equals(data.frame["class"])
    assert GAP_PREFIX + "a" not in fold_dataset(data, _fold(), "A3").frame.columns
    assert not fold_dataset(data, _fold(), "A4:1").frame[["a", "b"]].isna().any().any()


def test_a2_pretraining_holes_are_a1s_draw_for_draw_and_never_a_fill() -> None:
    data = _dataset()
    features = data.frame.drop(columns="class")
    a2 = fold_dataset(data, _fold(), "A2").frame.drop(columns="class")
    plain = TabularEmbedder(features, [], ["a", "b"])
    wide = TabularEmbedder(a2, [], ["a", "b"], indicator_channel=True)
    np.random.seed(7)
    holes_a1 = EpochMasker(plain, features, torch.device("cpu")).draw(0.3)
    np.random.seed(7)
    holes_a2 = EpochMasker(wide, a2, torch.device("cpu")).draw(0.3)
    assert torch.equal(holes_a1.masked_positions, holes_a2.masked_positions)
    gaps = torch.tensor(features.isna().to_numpy())
    assert not (holes_a2.masked_positions & gaps).any()
    assert holes_a2.fill_flags is not None and torch.equal(holes_a2.fill_flags, gaps.t())


def test_the_indicator_reaches_the_token() -> None:
    data = _dataset()
    a2 = fold_dataset(data, _fold(), "A2").frame.drop(columns="class")
    wide = TabularEmbedder(a2, [], ["a", "b"], indicator_channel=True)
    encoded = wide.encode(a2)
    flipped = type(encoded)(encoded.cat_indices, encoded.num_values, encoded.mask_flags, encoded.null_flags,
                            encoded.masked_positions, fill_flags=~encoded.fill_flags)  # type: ignore[operator]
    assert not torch.allclose(wide(encoded), wide(flipped))


def test_an_unknown_arm_is_refused() -> None:
    with pytest.raises(ValueError):
        Hyperparameters.from_mapping({"C15_ARM": "A9"})


def test_a5_cannot_read_its_blocked_column() -> None:
    from src.models import TridentModel
    from src.training.c15_inputs import blocked_position
    from src.transformer import TabularTransformerEncoder

    data = _dataset()
    features = data.frame.drop(columns="class")
    torch.manual_seed(0)
    embedder = TabularEmbedder(features, [], ["a", "b"], dimensao=8, hidden_dim=16)
    transformer = TabularTransformerEncoder(d_model=8, nhead=2, num_layers=2, dim_feedforward=16, dropout=0.0)
    transformer.blocked_key = blocked_position("A5:a", [], ["a", "b"])
    model = TridentModel(embedder, transformer, num_labels=2).eval()
    shifted = features.assign(a=features["a"] * 7 + 3)
    from src.utils import preprocess_table

    with torch.no_grad():
        one = model(embedder.encode(preprocess_table(features, p_base=0.0, fine_tunning=True)))
        two = model(embedder.encode(preprocess_table(shifted, p_base=0.0, fine_tunning=True)))
        transformer.blocked_key = None
        three = model(embedder.encode(preprocess_table(shifted, p_base=0.0, fine_tunning=True)))
    assert torch.equal(one, two) and not torch.allclose(one, three)
