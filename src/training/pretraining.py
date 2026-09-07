"""Masked reconstruction pre-training stage."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.embedder import TabularEmbedder
from src.models import TridentPretrainer
from src.transformer import TabularTransformerEncoder
from src.utils import preprocess_table

from .types import FoldSplit, Hyperparameters, PreparedDataset, PretrainingOutcome, TrainingTracker


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
    model = TridentPretrainer(embedder, transformer).to(device)

    def initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)

    model.apply(initialize_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparameters.pretraining_learning_rate,
        weight_decay=hyperparameters.pretraining_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hyperparameters.pretraining_epochs
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []

    print("\n=== Starting Pre-Training (new masks each epoch) ===")
    for epoch in tqdm(range(hyperparameters.pretraining_epochs), desc="Pre train epochs"):
        masked_train_frame = preprocess_table(
            train_frame.copy(), p_base=hyperparameters.mask_probability, fine_tunning=False
        )
        masked_validation_frame = preprocess_table(
            validation_frame.copy(), p_base=hyperparameters.mask_probability, fine_tunning=False
        )
        model.train()
        indices = torch.randperm(len(masked_train_frame))
        train_loss_sum = 0.0
        train_steps = 0
        # for start in tqdm(
        #     range(0, len(masked_train_frame), hyperparameters.batch_size),
        #     desc="Batch",
        #     unit="batch",
        #     leave=False,
        # ):
        for start in range(0, len(masked_train_frame), hyperparameters.batch_size):

            batch_indices = indices[start : start + hyperparameters.batch_size]
            masked_batch = masked_train_frame.iloc[batch_indices].reset_index(drop=True)
            original_batch = train_frame.iloc[batch_indices].reset_index(drop=True)
            optimizer.zero_grad()
            total_loss, _ = model(masked_batch, original_batch)
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_sum += total_loss.item()
            train_steps += 1
        average_train_loss = train_loss_sum / train_steps
        train_losses.append(average_train_loss)

        model.eval()
        with torch.no_grad():
            validation_loss_sum = 0.0
            validation_steps = 0
            for start in range(0, len(masked_validation_frame), hyperparameters.batch_size):
                masked_batch = masked_validation_frame.iloc[
                    start : start + hyperparameters.batch_size
                ].reset_index(drop=True)
                original_batch = validation_frame.iloc[
                    start : start + hyperparameters.batch_size
                ].reset_index(drop=True)
                total_loss, _ = model(masked_batch, original_batch)
                validation_loss_sum += total_loss.item()
                validation_steps += 1
            average_validation_loss = validation_loss_sum / validation_steps
            validation_losses.append(average_validation_loss)
        tracker.log_metrics(
            {
                "pretrain/train_loss": float(average_train_loss),
                "pretrain/val_loss": float(average_validation_loss),
            },
            step=epoch,
        )

    return PretrainingOutcome(
        model=model,
        train_losses=train_losses,
        validation_losses=validation_losses,
    )
