"""The pre-training stage and the objectives it can regress a masked cell onto (ADR 0011)."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.embedder import TabularEmbedder
from src.models import NormalizedEmbeddingPretrainer, TridentPretrainer
from src.training.pretraining import train_pretrainer
from src.training.tracking import BufferedFoldTracker
from src.training.types import FoldSplit, Hyperparameters, PreparedDataset
from src.transformer import TabularTransformerEncoder
from src.utils import preprocess_table

CATEGORICAL = ["sign"]
NUMERICAL = ["x", "copy_of_x"]


def _learnable_dataset(rows: int = 96) -> PreparedDataset:
    """A table whose every column follows from any other, so a hidden cell is recoverable."""
    stream = np.random.default_rng(0)
    x = stream.normal(size=rows)
    frame = pd.DataFrame(
        {
            "sign": np.where(x > 0, "positive", "negative"),
            "x": x,
            "copy_of_x": x,
            "class": (x > 0).astype(int),
        }
    )
    scaler = StandardScaler()
    frame[NUMERICAL] = scaler.fit_transform(frame[NUMERICAL])
    return PreparedDataset(
        frame=frame,
        raw_numerical=frame[NUMERICAL].copy(),
        label_column="class",
        categorical_columns=CATEGORICAL,
        numerical_columns=NUMERICAL,
        label_classes=np.array([0, 1]),
        source_path=Path("datasets/processed_datasets/toy/toy_00nan.csv"),
        splits_path=Path("datasets/processed_datasets/splits/toy_split.json"),
        scaler=scaler,
    )


def _fold(rows: int = 96) -> FoldSplit:
    every = np.arange(rows)
    return FoldSplit(
        train_indices=every[: rows * 3 // 4],
        validation_indices=every[rows * 3 // 4 :],
        test_indices=every[rows * 3 // 4 :],
    )


def _hyperparameters(**overrides: object) -> Hyperparameters:
    return Hyperparameters.from_mapping(
        {
            "DIM": 16, "HIDDEN_DIM": 8, "HEADS": 2, "LAYERS": 1, "DIM_FEED": 32,
            "DROPOUT": 0.0, "EPOCHS_PRE": 60, "BATCH": 32, "PROB_MASCARA": 0.3,
            "LR_PRE": 0.01, "LR_SCHEDULER": "constant",
            **overrides,
        }
    )


def test_value_pretraining_hands_over_an_encoder_that_recovers_hidden_values() -> None:
    """The point of the objective: at a masked cell the encoder carries the cell's value.

    Its stage model reconstructs through the decoder's own heads, so its reconstruction of
    a hidden ``copy_of_x`` is compared with the column mean, the fill that knows nothing.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    dataset = _learnable_dataset()
    fold = _fold()

    outcome = train_pretrainer(
        dataset,
        fold,
        _hyperparameters(PRETRAIN_OBJECTIVE="value"),
        torch.device("cpu"),
        BufferedFoldTracker(),
    )

    features = dataset.frame.drop(columns=["class"])
    validation = features.iloc[fold.validation_indices].reset_index(drop=True)
    hidden = validation.copy()
    hidden["copy_of_x"] = "[MASK]"
    outcome.model.eval()
    with torch.no_grad():
        predicted = outcome.model.predict(outcome.model.embedder.encode(hidden, "cpu"))
    truth = torch.tensor(validation["copy_of_x"].to_numpy(), dtype=torch.float32)
    column = NUMERICAL.index("copy_of_x")
    model_error = torch.mean((predicted.numerical_values[column] - truth) ** 2).item()
    mean_fill_error = torch.mean((truth - truth.mean()) ** 2).item()

    assert model_error < 0.25 * mean_fill_error
    assert outcome.train_losses[-1] < outcome.train_losses[0]


def test_the_normalised_objective_cannot_be_met_by_rescaling_the_embeddings() -> None:
    """Critique F-13-1: the raw embedding loss moves when the embeddings change scale.

    That lets the loss fall without the encoder predicting anything. With the target
    layer-normalised, a predictor that says nothing scores the same at every scale, so only
    predicting the hidden cell can lower it.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    features = _learnable_dataset().frame.drop(columns=["class"])
    embedder = TabularEmbedder(features, CATEGORICAL, NUMERICAL, dimensao=16, hidden_dim=8)
    transformer = TabularTransformerEncoder(
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0
    )
    normalised = NormalizedEmbeddingPretrainer(embedder, transformer)
    raw = TridentPretrainer(embedder, transformer)
    silent = normalised.predictor[-1]
    assert isinstance(silent, torch.nn.Linear)
    with torch.no_grad():
        silent.weight.zero_()
        silent.bias.zero_()
    masked = preprocess_table(features.copy(), p_base=0.5, fine_tunning=False)

    losses = []
    for scale in (1.0, 4.0):
        with torch.no_grad():
            for parameter in embedder.parameters():
                parameter.mul_(scale)
            losses.append((normalised(masked, features)[0].item(), raw(masked, features)[0].item()))

    (normalised_before, raw_before), (normalised_after, raw_after) = losses
    assert abs(normalised_before - 1.0) < 1e-3
    assert abs(normalised_after - 1.0) < 1e-3
    assert raw_after > 2 * raw_before


def test_each_objective_trains_its_own_stage_model() -> None:
    """A run asking for the normalised objective must not silently train the raw one."""
    dataset = _learnable_dataset()
    kinds = {}
    for objective in ("embedding", "embedding_normalized"):
        outcome = train_pretrainer(
            dataset,
            _fold(),
            _hyperparameters(PRETRAIN_OBJECTIVE=objective, EPOCHS_PRE=1),
            torch.device("cpu"),
            BufferedFoldTracker(),
        )
        kinds[objective] = type(outcome.model)

    assert kinds == {
        "embedding": TridentPretrainer,
        "embedding_normalized": NormalizedEmbeddingPretrainer,
    }
