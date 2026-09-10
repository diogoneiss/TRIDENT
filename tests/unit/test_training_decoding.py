"""The decode stage: training a decoder and scoring what it reconstructs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.embedder import TabularEmbedder
from src.models import TridentPretrainer
from src.training.data import evaluation_mask
from src.training.decoding import train_and_evaluate_decoder
from src.training.tracking import BufferedFoldTracker
from src.training.types import FoldSplit, Hyperparameters, PreparedDataset, PretrainingOutcome
from src.transformer import TabularTransformerEncoder
from src.utils import preprocess_table

CATEGORICAL = ["colour", "shape"]
NUMERICAL = ["size", "weight"]


def _complete_frame(rows: int = 36) -> pd.DataFrame:
    """The table before any cell was taken away, as a _00nan sibling is stored."""
    colours = ["red", "blue", "green", "red"]
    shapes = ["round", "square", "round", "square"]
    return pd.DataFrame(
        {
            "colour": [colours[index % 4] for index in range(rows)],
            "shape": [shapes[index % 4] for index in range(rows)],
            "size": [float(index % 9) for index in range(rows)],
            "weight": [float(index % 5) / 2 for index in range(rows)],
            "class": [index % 2 for index in range(rows)],
        }
    )


def _dataset(rows: int = 36) -> PreparedDataset:
    """That table with cells taken away, then scaled as preparation would leave it."""
    frame = _complete_frame(rows)
    frame.loc[frame.index % 5 == 0, "size"] = np.nan
    frame.loc[frame.index % 7 == 0, "colour"] = np.nan
    scaler = StandardScaler()
    frame[NUMERICAL] = scaler.fit_transform(frame[NUMERICAL])
    frame.attrs["dataset_name"] = "toy_20nan"
    return PreparedDataset(
        frame=frame,
        label_column="class",
        categorical_columns=CATEGORICAL,
        numerical_columns=NUMERICAL,
        label_classes=np.array([0, 1]),
        source_path=Path("datasets/processed_datasets/toy/toy_20nan.csv"),
        splits_path=Path("datasets/processed_datasets/splits/toy_split.json"),
        scaler=scaler,
    )


def _fold(rows: int = 36) -> FoldSplit:
    every = np.arange(rows)
    return FoldSplit(
        train_indices=every[: rows // 2],
        validation_indices=every[rows // 2 : rows // 2 + rows // 4],
        test_indices=every[rows // 2 + rows // 4 :],
    )


def _pretrained(dataset: PreparedDataset) -> PretrainingOutcome:
    """The encoder a real run would hand over, without paying for pre-training."""
    torch.manual_seed(0)
    features = dataset.frame.drop(columns=[dataset.label_column])
    embedder = TabularEmbedder(features, CATEGORICAL, NUMERICAL, dimensao=8, hidden_dim=4)
    transformer = TabularTransformerEncoder(
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0
    )
    return PretrainingOutcome(
        model=TridentPretrainer(embedder, transformer), train_losses=[], validation_losses=[]
    )


def _hyperparameters(**overrides) -> Hyperparameters:
    return Hyperparameters.from_mapping(
        {
            "DIM": 8, "HIDDEN_DIM": 4, "HEADS": 2, "LAYERS": 1, "DIM_FEED": 16,
            "DROPOUT": 0.0, "EPOCHS_PRE": 1, "BATCH": 16, "PROB_MASCARA": 0.5,
            "EPOCHS_DECODE": 3, "LR_DECODE": 0.01, "EVAL_MASK_RATE": 0.3,
            **overrides,
        }
    )


def _run(dataset=None, hyperparameters=None, sibling=None, **kwargs):
    dataset = dataset if dataset is not None else _dataset()
    tracker = BufferedFoldTracker()
    outcome = train_and_evaluate_decoder(
        dataset=dataset,
        fold=_fold(len(dataset.frame)),
        pretraining=_pretrained(dataset),
        hyperparameters=hyperparameters or _hyperparameters(),
        device=torch.device("cpu"),
        tracker=tracker,
        seed=42,
        fold_ordinal=1,
        complete_sibling=sibling,
        **kwargs,
    )
    return outcome, tracker


def test_the_decode_stage_reports_a_loss_for_every_epoch_it_trained() -> None:
    """Its loss curves are what a reader plots, so every epoch has to be on them."""
    outcome, tracker = _run(hyperparameters=_hyperparameters(EPOCHS_DECODE=3))

    assert len(outcome.train_losses) == 3
    assert len(outcome.validation_losses) == 3
    logged = {event.key for event in tracker.metric_events if event.step is not None}
    assert {"decode/train_loss", "decode/val_loss", "decode/learning_rate"} <= logged


def test_the_decode_stage_scores_and_lists_every_cell_it_hid() -> None:
    """Two outputs from one pass: numbers that rank the fold, and the cells behind them.

    The count is checked against the mask drawn independently here, so the table cannot
    quietly agree with metrics computed from the same mistake.
    """
    dataset = _dataset()
    hyperparameters = _hyperparameters()
    outcome, tracker = _run(dataset=dataset, hyperparameters=hyperparameters)

    metrics = outcome.result.metrics
    assert "impute/masked/impute_score" in metrics
    assert "impute/masked/rmse_num_z" in metrics
    assert "impute/masked/acc_cat" in metrics

    features = dataset.frame.drop(columns=[dataset.label_column])
    test_frame = features.iloc[_fold(len(dataset.frame)).test_indices].reset_index(drop=True)
    drawn_here = evaluation_mask(test_frame, hyperparameters.eval_mask_rate, 42, 1)
    hidden_here = int((drawn_here == "[MASK]").to_numpy().sum())

    listed = outcome.scored_cells[outcome.scored_cells["population"] == "masked"]
    assert len(listed) == hidden_here
    assert set(listed.columns) >= {"row", "column", "kind", "population", "actual", "imputed"}


def test_cells_the_dataset_is_missing_are_scored_against_the_complete_table() -> None:
    """The gaps in a variant are the real imputation benchmark: their truth is known.

    They reach the model as [MASK], the token the decoder was trained on, rather than as
    [NULL], which it has never been asked to reconstruct.
    """
    dataset = _dataset()
    with_truth, _ = _run(dataset=dataset, sibling=_complete_frame())
    without_truth, _ = _run(dataset=dataset, sibling=None)

    assert "impute/induced/impute_score" in with_truth.result.metrics
    assert not any(key.startswith("impute/induced/") for key in without_truth.result.metrics)

    features = dataset.frame.drop(columns=[dataset.label_column])
    test_rows = features.iloc[_fold(len(dataset.frame)).test_indices]
    gaps_here = int(test_rows.isna().to_numpy().sum())
    listed = with_truth.scored_cells[with_truth.scored_cells["population"] == "induced"]
    assert len(listed) == gaps_here


def test_the_best_epoch_is_the_one_that_is_kept() -> None:
    """Training runs the whole budget, but the model handed back is the best it ever was.

    This only means anything because the validation mask never changes: every epoch is
    graded on the same hidden cells, so the losses are comparable at all.
    """
    dataset = _dataset()
    hyperparameters = _hyperparameters(EPOCHS_DECODE=6, LR_DECODE=0.2)
    outcome, _ = _run(dataset=dataset, hyperparameters=hyperparameters)

    features = dataset.frame.drop(columns=[dataset.label_column])
    validation = features.iloc[_fold(len(dataset.frame)).validation_indices].reset_index(drop=True)
    embedder = outcome.model.embedder
    device = torch.device("cpu")
    hidden = embedder.encode(
        evaluation_mask(validation, hyperparameters.eval_mask_rate, 42, 1), device
    )
    clean = embedder.encode(preprocess_table(validation.copy(), p_base=0.0, fine_tunning=True), device)
    with torch.no_grad():
        kept_model_scores, _ = outcome.model(hidden, clean)

    assert kept_model_scores.item() == pytest.approx(min(outcome.validation_losses), rel=1e-5)


def test_extra_rates_are_scored_as_diagnostics_beside_the_one_that_ranks() -> None:
    """One model, several difficulties. Only the primary rate ranks the fold.

    Scoring a second rate costs one forward pass; training a second model to get it would
    cost the whole pre-training budget again.
    """
    outcome, _ = _run(
        hyperparameters=_hyperparameters(EVAL_MASK_RATES_EXTRA=[0.1, 0.5])
    )

    metrics = outcome.result.metrics
    assert "impute/masked/impute_score" in metrics
    assert "impute/masked/rate_10/rmse_num_z" in metrics
    assert "impute/masked/rate_50/rmse_num_z" in metrics
    assert metrics["impute/masked/rate_10/n_num_cells"] < metrics["impute/masked/rate_50/n_num_cells"]


def test_the_null_path_diagnostic_asks_the_same_cells_through_the_other_token() -> None:
    """The same gaps, shown as [NULL] instead of [MASK], to test whether the head transfers.

    Off unless asked for, and never allowed near the ranking metric.
    """
    dataset = _dataset()
    asked, _ = _run(dataset=dataset, sibling=_complete_frame(), score_null_path=True)
    unasked, _ = _run(dataset=dataset, sibling=_complete_frame())

    assert "impute/induced/null_token/acc_cat" in asked.result.metrics
    assert not any("null_token" in key for key in unasked.result.metrics)
    through_mask = asked.scored_cells["population"] == "induced"
    through_null = asked.scored_cells["population"] == "induced_null_token"
    assert int(through_null.sum()) == int(through_mask.sum())
