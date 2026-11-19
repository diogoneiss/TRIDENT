"""``--decode_sw_weight`` (imputation-protocol/73): a sliced Wasserstein term in the decode loss, between the batch's
rows as the model fills them and as they truly are, laid out as ``sw_score`` lays them out."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.embedder import EncodedTable, TabularEmbedder
from src.models import TridentDecoder
from src.training.decoding import _sliced_rows
from src.transformer import TabularTransformerEncoder
from src.utils import preprocess_table

CATEGORICAL = ["colour", "shape"]
NUMERICAL = ["size", "weight"]


def _frame(rows: int = 48) -> pd.DataFrame:
    """Column kinds interleaved; ``colour`` holds gaps, so its real categories are not contiguous ids."""
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "size": (index % 7).astype(float),
            "colour": np.array(["red", "blue", "green", np.nan], dtype=object)[index % 4],
            "weight": (index % 3).astype(float) / 2,
            "shape": np.array(["round", "square", "star", "oval", "cube"], dtype=object)[index % 5],
        }
    )


def _parts(frame: pd.DataFrame) -> tuple[TabularEmbedder, TabularTransformerEncoder]:
    embedder = TabularEmbedder(frame, CATEGORICAL, NUMERICAL, dimensao=8, hidden_dim=4)
    transformer = TabularTransformerEncoder(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0)
    return embedder, transformer


def _batch(frame: pd.DataFrame, embedder: TabularEmbedder, p_base: float = 0.4, seed: int = 0) -> tuple[EncodedTable, EncodedTable]:
    device = torch.device("cpu")
    np.random.seed(seed)
    hidden = embedder.encode(preprocess_table(frame.copy(), p_base=p_base, fine_tunning=False), device)
    clean = embedder.encode(preprocess_table(frame.copy(), p_base=0.0, fine_tunning=True), device)
    return hidden, clean


def _width(embedder: TabularEmbedder) -> int:
    return len(NUMERICAL) + sum(len(embedder.label_encoders[c].classes_) + 1 for c in CATEGORICAL)


def _rng_states() -> tuple[torch.Tensor, tuple[object, ...], list[torch.Tensor] | None]:
    return (torch.get_rng_state(), np.random.get_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _same_states(a: tuple[torch.Tensor, tuple[object, ...], list[torch.Tensor] | None],
                 b: tuple[torch.Tensor, tuple[object, ...], list[torch.Tensor] | None]) -> bool:
    numpy_a, numpy_b = a[1], b[1]
    same_numpy = numpy_a[0] == numpy_b[0] and np.array_equal(numpy_a[1], numpy_b[1]) and numpy_a[2:] == numpy_b[2:]
    same_cuda = a[2] is None and b[2] is None or a[2] is not None and b[2] is not None and all(
        torch.equal(x, y) for x, y in zip(a[2], b[2]))
    return torch.equal(a[0], b[0]) and bool(same_numpy) and same_cuda


def test_off_is_the_decoder_as_it_was() -> None:
    """At weight 0 the decoder draws, holds and returns nothing new; at any weight it draws nothing from the global
    streams, so an arm with the term sees the same masks, shuffles and dropout as its control, and its validation loss
    is the control's arithmetic."""
    frame = _frame()
    built = []
    for weight in (0.0, 0.5):
        torch.manual_seed(0)
        np.random.seed(0)
        embedder, transformer = _parts(frame)
        model = TridentDecoder(embedder, transformer, sw_weight=weight)
        built.append((model, embedder, _rng_states()))
    (off, embedder, after_off), (on, _, after_on) = built
    assert _same_states(after_off, after_on)
    assert off.state_dict().keys() == on.state_dict().keys()
    for (name, a), (_, b) in zip(off.state_dict().items(), on.state_dict().items()):
        assert torch.equal(a, b), name

    hidden, clean = _batch(frame, embedder)
    after_forward = []
    for model in (off, on):
        model.train()
        torch.manual_seed(1)
        np.random.seed(1)
        loss, metrics = model(hidden, clean)
        loss.backward()
        after_forward.append((metrics, _rng_states()))
    assert "sliced" not in after_forward[0][0]
    assert after_forward[1][0]["sliced"].item() > 0
    assert _same_states(after_forward[0][1], after_forward[1][1])

    evaluated = []
    for model in (off, on):
        model.eval()
        with torch.no_grad():
            evaluated.append(model(hidden, clean))
    assert torch.equal(evaluated[0][0], evaluated[1][0])
    assert evaluated[0][1].keys() == evaluated[1][1].keys() == {"cross_entropy", "mse"}


def _pair(frame: pd.DataFrame, weight: float = 0.5) -> tuple[TridentDecoder, TridentDecoder, TabularEmbedder]:
    torch.manual_seed(0)
    embedder, transformer = _parts(frame)
    per_column = TridentDecoder(embedder, transformer, sw_weight=weight)
    batched = TridentDecoder(embedder, transformer, sw_weight=weight, batched_heads=True)
    batched.load_state_dict(per_column.state_dict())
    return per_column, batched, embedder


def _without_hidden(hidden: EncodedTable, column: int | None = None, rows: slice = slice(None)) -> EncodedTable:
    """The same view with nothing hidden in ``column`` (every column if None) of ``rows``."""
    positions = hidden.masked_positions.clone()
    positions[rows, slice(None) if column is None else column] = False
    return EncodedTable(cat_indices=hidden.cat_indices, num_values=hidden.num_values, mask_flags=hidden.mask_flags,
                        null_flags=hidden.null_flags, masked_positions=positions)


def test_both_head_paths_give_the_same_term_and_a_fallback_batch_keeps_it() -> None:
    frame = _frame()
    per_column, batched, embedder = _pair(frame)
    hidden, clean = _batch(frame, embedder)
    assert bool(hidden.masked_positions.any(dim=0).all()), "every column must hide a cell, or no batch is batched"

    terms = [model(hidden, clean)[1]["sliced"].item() for model in (per_column, batched)]
    assert terms[0] > 0
    assert terms[1] == pytest.approx(terms[0], abs=1e-5)

    # ``shape`` (second categorical column) hides nothing: the batched model falls back and keeps the term
    fallback = _without_hidden(hidden, CATEGORICAL.index("shape"))
    kept = [model(fallback, clean)[1]["sliced"].item() for model in (per_column, batched)]
    assert kept[1] > 0
    assert kept[1] == pytest.approx(kept[0], abs=1e-5)


def _fixed_fills(model: TridentDecoder, embedder: TabularEmbedder) -> tuple[list[float], list[int]]:
    """Heads that answer constants: each numerical column a number, each categorical column one of its real
    categories with probability 1 (logits a saturated bias). Returns the numbers and the vocabulary ids."""
    numbers = [0.75, -1.5]
    ids = []
    with torch.no_grad():
        for index, key in enumerate(embedder.numerical_keys):
            for parameter in model.numerical_heads[key].parameters():
                parameter.zero_()
            model.numerical_heads[key][2].bias.fill_(numbers[index])
        for index, key in enumerate(embedder.categorical_keys):
            head = model.categorical_heads[key]
            head.weight.zero_()
            head.bias.zero_()
            chosen = 1 + index  # neither the first real category nor the same slot in both columns
            head.bias[chosen] = 1e4
            ids.append(int(getattr(model, f"valid_ids_{key}")[chosen]))
    return numbers, ids


@pytest.mark.parametrize("batched", [False, True])
def test_the_term_is_sw_scores_distance_on_the_batchs_touched_rows(batched: bool) -> None:
    """An independent reference built with the scorer's own row layout and ``default_rng(1)``'s directions: a slot
    shifted by one, or the categorical blocks laid before the numbers, moves the value."""
    frame = _frame()
    per_column, both, embedder = _pair(frame)
    model = both if batched else per_column
    numbers, ids = _fixed_fills(model, embedder)
    hidden, clean = _batch(frame, embedder, p_base=0.3)
    hidden = _without_hidden(hidden, rows=slice(0, 5))  # every masked row hides a cell; these five then hide none
    model.train()

    term = model(hidden, clean)[1]["sliced"].item()

    n_cat = len(CATEGORICAL)
    mask = hidden.masked_positions
    true_numbers = clean.num_values.double()
    true_categories = clean.cat_indices.clone()
    filled_numbers, filled_categories = true_numbers.clone(), true_categories.clone()
    for index in range(len(NUMERICAL)):
        filled_numbers[index, mask[:, n_cat + index]] = numbers[index]
    for index in range(n_cat):
        filled_categories[index, mask[:, index]] = ids[index]
    touched = torch.nonzero(mask.any(dim=1)).squeeze(1)
    assert len(touched) < len(frame), "a row with nothing hidden must stay out of the term"
    sizes = [len(embedder.label_encoders[c].classes_) for c in CATEGORICAL]
    filled = _sliced_rows(filled_numbers, filled_categories, sizes, touched).numpy()
    true = _sliced_rows(true_numbers, true_categories, sizes, touched).numpy()
    width = _width(embedder)
    drawn = np.random.default_rng(1).normal(size=(512, width))
    along = (drawn / np.linalg.norm(drawn, axis=1, keepdims=True)).T
    reference = float(np.mean(np.abs(np.sort(filled @ along, axis=0) - np.sort(true @ along, axis=0))))

    assert reference > 0
    assert term / math.sqrt(width) == pytest.approx(reference, abs=1e-5)


def test_a_hidden_categorys_true_slot_is_cleared() -> None:
    """The helper fed ``colour``'s gaps, whose truth [NULL] is no head class, so no probability lands on it: its true
    one-hot must still be cleared at slot id+1, leaving the block all zero. A clear shifted by one slot moves the value,
    which a hidden head class would hide (its probability overwrites the slot)."""
    frame = _frame()
    model, _, embedder = _pair(frame)
    _, clean = _batch(frame, embedder)
    rows = torch.tensor([3, 7])  # colour is nan here
    column = CATEGORICAL.index("colour")
    null = int(np.flatnonzero(embedder.label_encoders["colour"].classes_ == "[NULL]")[0])
    assert bool((clean.cat_indices[column, rows] == null).all()) and null not in getattr(model, "valid_ids_colour").tolist()
    none = rows.new_zeros(0)
    no_probability = torch.zeros((len(rows), model._class_valid.shape[1]))
    term = model._sliced_term(clean, (none, none, clean.num_values.new_zeros(0)),
                              (rows, torch.full_like(rows, column), no_probability)).item()

    sizes = [len(embedder.label_encoders[c].classes_) for c in CATEGORICAL]
    true = _sliced_rows(clean.num_values.double(), clean.cat_indices, sizes, rows).numpy()
    filled = true.copy()
    start = len(NUMERICAL) + sum(size + 1 for size in sizes[:column])
    filled[:, start:start + sizes[column] + 1] = 0.0
    width = _width(embedder)
    drawn = np.random.default_rng(1).normal(size=(512, width))
    along = (drawn / np.linalg.norm(drawn, axis=1, keepdims=True)).T
    reference = float(np.mean(np.abs(np.sort(filled @ along, axis=0) - np.sort(true @ along, axis=0))))

    assert reference > 0
    assert term / math.sqrt(width) == pytest.approx(reference, abs=1e-5)


@pytest.mark.parametrize("batched", [False, True])
def test_the_term_teaches_both_kinds_of_head(batched: bool) -> None:
    """The term's own gradient, the loss with it less the loss without it, reaches a categorical head through the
    softmax and a numerical head through its value."""
    frame = _frame()
    per_column, both, embedder = _pair(frame, weight=1.0)
    model = both if batched else per_column
    hidden, clean = _batch(frame, embedder)
    grads = []
    for weight in (1.0, 0.0):
        model.sw_weight = weight
        model.zero_grad()
        loss, _ = model(hidden, clean)
        loss.backward()
        grads.append({name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None})
    with_term, without = grads
    for name in ("categorical_heads.colour.weight", "numerical_heads.size.2.weight"):
        assert float((with_term[name] - without[name]).abs().max()) > 1e-4, name


def test_a_batch_with_nothing_hidden_costs_nothing() -> None:
    frame = _frame()
    per_column, batched, embedder = _pair(frame)
    _, clean = _batch(frame, embedder)
    for model in (per_column, batched):
        loss, metrics = model(clean, clean)
        assert loss.item() == 0.0 and "sliced" not in metrics


@pytest.mark.parametrize("batched", [False, True])
def test_the_weight_scales_the_term_in_the_loss(batched: bool) -> None:
    """The loss is the per-cell loss plus weight times the term, so the smoke-fixed weight means what the ticket says."""
    frame = _frame()
    per_column, both, embedder = _pair(frame, weight=0.5)
    model = both if batched else per_column
    hidden, clean = _batch(frame, embedder)
    torch.manual_seed(3)
    with_term, metrics = model(hidden, clean)
    model.sw_weight = 0.0
    torch.manual_seed(3)
    without, _ = model(hidden, clean)
    assert float(with_term - without) == pytest.approx(0.5 * float(metrics["sliced"]), abs=1e-6)
