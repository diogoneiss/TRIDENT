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


def test_a_frozen_encoder_does_not_move_while_the_head_does() -> None:
    import copy

    from src.models import TridentModel
    from src.training.types import Hyperparameters
    from src.transformer import TabularTransformerEncoder
    from src.utils import preprocess_table

    frame = _frame()
    features = frame.copy()
    torch.manual_seed(0)
    embedder = TabularEmbedder(features, ["c"], ["a", "b"], dimensao=8, hidden_dim=16)
    transformer = TabularTransformerEncoder(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.3)
    model = TridentModel(embedder, transformer, num_labels=2)
    h = Hyperparameters.from_mapping({"FINETUNE_FREEZE_ENCODER": "all"})
    assert h.finetune_freeze_encoder == "all" and Hyperparameters.from_mapping({}).finetune_freeze_encoder == "off"
    with pytest.raises(ValueError):
        Hyperparameters.from_mapping({"FINETUNE_FREEZE_ENCODER": "head"})
    # The same steps finetuning.py takes under the flag.
    before = copy.deepcopy(model.state_dict())
    for p in list(model.embedder.parameters()) + list(model.transformer.parameters()):
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.01)
    model.train()
    model.embedder.eval()
    model.transformer.eval()
    assert not model.transformer.training and model.classifier.training
    encoded = embedder.encode(preprocess_table(features, p_base=0.0, fine_tunning=True))
    _, loss = model(encoded, labels=torch.zeros(len(frame), dtype=torch.long))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    encoder_keys = [k for k in before if k.startswith(("embedder.", "transformer."))]
    assert encoder_keys and all(torch.equal(before[k], after[k]) for k in encoder_keys)
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("classifier."))


def test_the_fix_flags_build_the_optimizer_groups_they_name() -> None:
    from src.models import TridentModel
    from src.training.finetuning import _finetune_parameters
    from src.training.types import Hyperparameters
    from src.transformer import TabularTransformerEncoder

    frame = _frame()
    embedder = TabularEmbedder(frame, ["c"], ["a", "b"], dimensao=8, hidden_dim=16)
    model = TridentModel(embedder, TabularTransformerEncoder(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0), num_labels=2)
    plain = Hyperparameters.from_mapping({})
    assert plain.finetune_freeze_epochs == 0 and plain.finetune_encoder_lr_scale == 1.0 and plain.finetune_head_lr_scale == 1.0
    assert not isinstance(_finetune_parameters(model, plain), list)  # one group, every run before the arms
    scaled = Hyperparameters.from_mapping({"FINETUNE_ENCODER_LR_SCALE": 0.1})
    head, encoder = _finetune_parameters(model, scaled)
    assert head["lr"] == pytest.approx(0.001) and encoder["lr"] == pytest.approx(0.0001)
    assert len(head["params"]) + len(encoder["params"]) == len(list(model.parameters()))
    frozen = Hyperparameters.from_mapping({"FINETUNE_FREEZE_ENCODER": "all", "FINETUNE_HEAD_LR_SCALE": 10})
    (only,) = _finetune_parameters(model, frozen)
    assert only["lr"] == pytest.approx(0.01) and len(only["params"]) == len(list(model.classifier.parameters()))
    assert Hyperparameters.from_mapping({"FINETUNE_FREEZE_EPOCHS": 50}).finetune_freeze_epochs == 50
    with pytest.raises(ValueError):
        Hyperparameters.from_mapping({"FINETUNE_FREEZE_EPOCHS": -1})
    with pytest.raises(ValueError):
        Hyperparameters.from_mapping({"FINETUNE_ENCODER_LR_SCALE": 0})


def test_fine_tuning_may_name_its_own_schedule() -> None:
    from src.training.types import Hyperparameters

    assert Hyperparameters.from_mapping({}).finetune_lr_scheduler == ""
    assert Hyperparameters.from_mapping({"FINETUNE_LR_SCHEDULER": "cosine"}).finetune_lr_scheduler == "cosine"
    with pytest.raises(ValueError):
        Hyperparameters.from_mapping({"FINETUNE_LR_SCHEDULER": "sawtooth"})
