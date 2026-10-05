"""Masked reconstruction pre-training stage."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.embedder import EncodedTable, EpochMasker, TabularEmbedder
from src.models import NormalizedEmbeddingPretrainer, TridentDecoder, TridentPretrainer
from src.transformer import TabularTransformerEncoder

from .schedulers import StageScheduler, batches_per_epoch
from .types import FoldSplit, Hyperparameters, PreparedDataset, PretrainingOutcome, TrainingTracker


def _stage_model(
    embedder: TabularEmbedder,
    transformer: TabularTransformerEncoder,
    hyperparameters: Hyperparameters,
) -> nn.Module:
    """What the stage trains for its objective (ADR 0011). It hands over only the encoder.

    ``value`` trains fresh decoder heads on the hidden values with the decode stage's own
    loss; they are discarded with the stage, since the decode stage builds its own.
    """
    if hyperparameters.pretraining_objective == "value":
        return TridentDecoder(
            embedder,
            transformer,
            lambda_num=hyperparameters.lambda_num,
            batched_heads=hyperparameters.decoder_heads == "batched",
        )
    if hyperparameters.pretraining_objective == "embedding_normalized":
        return NormalizedEmbeddingPretrainer(embedder, transformer)
    return TridentPretrainer(embedder, transformer)


def _as_is(encoded: EncodedTable) -> EncodedTable:
    return encoded


def train_pretrainer(
    dataset: PreparedDataset,
    fold: FoldSplit,
    hyperparameters: Hyperparameters,
    device: torch.device,
    tracker: TrainingTracker,
) -> PretrainingOutcome:
    """Train the masked reconstruction model using the legacy optimization loop."""
    feature_frame = dataset.frame.drop(columns=[dataset.label_column]).copy()
    train_frame = feature_frame.iloc[fold.train_indices].reset_index(drop=True)
    validation_frame = feature_frame.iloc[fold.validation_indices].reset_index(drop=True)

    embedder = TabularEmbedder(
        df=feature_frame,
        categorical_columns=dataset.categorical_columns,
        numerical_columns=dataset.numerical_columns,
        dimensao=hyperparameters.dimension,
        hidden_dim=hyperparameters.hidden_dimension,
    )
    transformer = TabularTransformerEncoder(
        d_model=embedder.dimensao,
        nhead=hyperparameters.heads,
        num_layers=hyperparameters.layers,
        dim_feedforward=hyperparameters.feedforward_dimension,
        dropout=hyperparameters.dropout,
    )
    model = _stage_model(embedder, transformer, hyperparameters).to(device)

    def initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)

    model.apply(initialize_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparameters.pretraining_learning_rate,
        weight_decay=hyperparameters.pretraining_weight_decay,
    )
    scheduler = StageScheduler(
        optimizer,
        hyperparameters.lr_scheduler,
        epochs=hyperparameters.pretraining_epochs,
        batches_per_epoch=batches_per_epoch(len(train_frame), hyperparameters.batch_size),
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []

    print("\n=== Starting Pre-Training (new masks each epoch) ===")
    # The clean reconstruction targets never change, so they are encoded once per fold.
    original_train = model.embedder.encode(train_frame, device)
    original_validation = model.embedder.encode(validation_frame, device)
    # Fresh masks every epoch, drawn exactly as preprocess_table draws them but applied to
    # tensors encoded once per fold, so no epoch goes back through pandas.
    train_masker = EpochMasker(model.embedder, train_frame, device)
    validation_masker = EpochMasker(model.embedder, validation_frame, device)
    # What the model sees at a row's real gap (``PRETRAIN_GAP_TOKEN``): the [NULL] it stores,
    # or [MASK] as the decode stage shows it since ADR 0014. Applied to the corrupted inputs
    # only; the targets and the loss positions do not move, and nothing is drawn.
    embedder = model.embedder
    assert isinstance(embedder, TabularEmbedder)
    shown = embedder.gaps_as_mask if hyperparameters.pretrain_gap_token == "mask" else _as_is

    for epoch in tqdm(range(hyperparameters.pretraining_epochs), desc="Pre train epochs"):
        masked_train = shown(train_masker.draw(hyperparameters.mask_probability))
        masked_validation = shown(validation_masker.draw(hyperparameters.mask_probability))
        model.train()
        # Drawn on the CPU generator, then moved once so batch slicing stays on device.
        indices = torch.randperm(len(train_frame)).to(device)
        # Accumulated on device in float64, so the per-batch losses are summed in
        # the same order and precision as before without a synchronisation each step.
        train_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        train_steps = 0
        for start in range(0, len(train_frame), hyperparameters.batch_size):
            batch_indices = indices[start : start + hyperparameters.batch_size]
            optimizer.zero_grad()
            total_loss, _ = model(masked_train[batch_indices], original_train[batch_indices])
            total_loss.backward()
            optimizer.step()
            scheduler.after_batch()
            train_loss_sum += total_loss.detach().double()
            train_steps += 1
        average_train_loss = (train_loss_sum / train_steps).item()
        train_losses.append(average_train_loss)

        model.eval()
        with torch.no_grad():
            validation_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            validation_steps = 0
            for start in range(0, len(validation_frame), hyperparameters.batch_size):
                batch = slice(start, start + hyperparameters.batch_size)
                total_loss, _ = model(masked_validation[batch], original_validation[batch])
                validation_loss_sum += total_loss.double()
                validation_steps += 1
            average_validation_loss = (validation_loss_sum / validation_steps).item()
            validation_losses.append(average_validation_loss)
        # Logged before the epoch-level schedules advance, so the value is the
        # rate this epoch actually trained at.
        tracker.log_metrics(
            {
                "pretrain/train_loss": float(average_train_loss),
                "pretrain/val_loss": float(average_validation_loss),
                "pretrain/learning_rate": scheduler.learning_rate,
            },
            step=epoch,
        )
        scheduler.after_epoch(average_validation_loss)

    return PretrainingOutcome(
        model=model,
        train_losses=train_losses,
        validation_losses=validation_losses,
    )
