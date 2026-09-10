"""The decoder that reconstructs actual cell values (ADR 0004, decisions 3 and 4)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.embedder import TabularEmbedder
from src.models import TridentDecoder
from src.transformer import TabularTransformerEncoder
from src.utils import preprocess_table

CATEGORICAL = ["colour"]
NUMERICAL = ["size", "weight"]


def _frame(rows: int = 40) -> pd.DataFrame:
    """A mixed frame whose categorical column has missing cells, as a real variant does."""
    colours = ["red", "blue", "red", np.nan]
    return pd.DataFrame(
        {
            "colour": [colours[index % len(colours)] for index in range(rows)],
            "size": [float(index % 7) for index in range(rows)],
            "weight": [float(index % 3) / 2 for index in range(rows)],
        }
    )


def _decoder(frame: pd.DataFrame) -> tuple[TridentDecoder, TabularEmbedder]:
    torch.manual_seed(0)
    embedder = TabularEmbedder(frame, CATEGORICAL, NUMERICAL, dimensao=8, hidden_dim=4)
    transformer = TabularTransformerEncoder(
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0
    )
    return TridentDecoder(embedder, transformer), embedder


def test_an_imputed_category_is_always_a_real_category() -> None:
    """The decoder cannot answer "missing" or "masked" when asked to fill a cell.

    Those tokens live in the column's vocabulary, so a head over the whole vocabulary
    could emit them. With an untrained model over 40 rows, a head that could reach the
    three special ids would hit one with overwhelming probability.
    """
    frame = _frame()
    decoder, embedder = _decoder(frame)
    encoded = embedder.encode(
        preprocess_table(frame.copy(), p_base=0.0, fine_tunning=True), torch.device("cpu")
    )

    prediction = decoder.predict(encoded)

    imputed = embedder.label_encoders["colour"].inverse_transform(
        prediction.categorical_ids[0].tolist()
    )
    assert set(imputed) <= {"blue", "red"}


def test_only_hidden_cells_teach_the_decoder() -> None:
    """A cell the model can already see teaches it nothing, so it carries no loss."""
    frame = _frame()
    decoder, embedder = _decoder(frame)
    device = torch.device("cpu")
    targets = embedder.encode(
        preprocess_table(frame.copy(), p_base=0.0, fine_tunning=True), device
    )
    np.random.seed(0)
    hidden = embedder.encode(
        preprocess_table(frame.copy(), p_base=0.5, fine_tunning=False), device
    )

    with_hidden_cells, _ = decoder(hidden, targets)
    with_nothing_hidden, _ = decoder(targets, targets)

    assert with_hidden_cells.item() > 0.0
    assert with_nothing_hidden.item() == 0.0


def test_each_kind_of_cell_is_averaged_over_its_own_count() -> None:
    """Loss is the mean categorical surprise plus the weighted mean numerical error.

    Averaging per kind rather than over all hidden cells together is what stops a table
    of twenty numerical columns and two categorical ones from drowning the categorical
    term. Silenced heads make the expected value independent of any weights: every
    category scores alike, so the surprise is ln(2) for a two-category column, and every
    numerical guess is 0.0, so the error of a cell is its true value squared.
    """
    clean = pd.DataFrame({"colour": ["red", "blue"], "size": [1.0, 3.0]})
    hidden = pd.DataFrame({"colour": ["[MASK]", "[MASK]"], "size": ["[MASK]", "[MASK]"]})
    torch.manual_seed(0)
    embedder = TabularEmbedder(clean, ["colour"], ["size"], dimensao=8, hidden_dim=4)
    transformer = TabularTransformerEncoder(
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0
    )
    decoder = TridentDecoder(embedder, transformer, lambda_num=3.0)
    with torch.no_grad():
        for head in list(decoder.categorical_heads.values()) + list(decoder.numerical_heads.values()):
            for parameter in head.parameters():
                parameter.zero_()
    device = torch.device("cpu")

    loss, _ = decoder(embedder.encode(hidden, device), embedder.encode(clean, device))

    # ln(2) + 3.0 * mean(1.0**2, 3.0**2) = 0.6931471805599453 + 15.0
    assert loss.item() == pytest.approx(15.693147180559945)


def test_a_numerical_cell_is_imputed_with_one_number_per_column() -> None:
    """Each numerical column gets its own scalar guess, in the scaled space it lives in."""
    frame = _frame(rows=6)
    decoder, embedder = _decoder(frame)
    with torch.no_grad():
        for head in decoder.numerical_heads.values():
            for parameter in head.parameters():
                parameter.zero_()
    encoded = embedder.encode(
        preprocess_table(frame.copy(), p_base=0.0, fine_tunning=True), torch.device("cpu")
    )

    prediction = decoder.predict(encoded)

    assert prediction.numerical_values.tolist() == [[0.0] * 6, [0.0] * 6]


def test_columns_never_share_head_parameters() -> None:
    """Silencing one column's head leaves every other column's imputation untouched."""
    frame = _frame(rows=6)
    decoder, embedder = _decoder(frame)
    encoded = embedder.encode(
        preprocess_table(frame.copy(), p_base=0.0, fine_tunning=True), torch.device("cpu")
    )
    before = decoder.predict(encoded).numerical_values

    with torch.no_grad():
        for parameter in decoder.numerical_heads["size"].parameters():
            parameter.zero_()
    after = decoder.predict(encoded).numerical_values

    assert after[0].tolist() == [0.0] * 6  # size, silenced
    assert after[1].tolist() == before[1].tolist()  # weight, untouched


def test_attaching_the_decoder_leaves_the_pretrained_encoder_untouched() -> None:
    """Everything pre-training learned survives the decoder being attached.

    Pre-training ends by re-initialising every linear and embedding layer in the whole
    model. A decoder that copied that habit would erase the weights it exists to read.
    """
    frame = _frame()
    torch.manual_seed(0)
    embedder = TabularEmbedder(frame, CATEGORICAL, NUMERICAL, dimensao=8, hidden_dim=4)
    transformer = TabularTransformerEncoder(
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0
    )
    pretrained = {
        f"{owner}.{name}": parameter.detach().clone()
        for owner, module in (("embedder", embedder), ("transformer", transformer))
        for name, parameter in module.named_parameters()
    }

    TridentDecoder(embedder, transformer)

    for key, original in pretrained.items():
        owner, name = key.split(".", 1)
        current = dict((embedder if owner == "embedder" else transformer).named_parameters())[name]
        assert torch.equal(current, original), key
