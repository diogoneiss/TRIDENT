"""Classifier fine-tuning and fold metric calculation."""

import copy
from typing import Sequence

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from src.models import TridentModel
from src.utils import preprocess_table

from .types import FoldResult, FoldSplit, Hyperparameters, PreparedDataset, PretrainingOutcome, TrainingTracker


def build_fold_result(
    fold: int | str,
    dataset_name: str,
    expected_labels: np.ndarray,
    predicted_labels: np.ndarray,
    test_loss: float,
    dataset_label_classes: Sequence[object] | None = None,
) -> FoldResult:
    """Build the legacy per-fold classification metric set."""
    metrics: dict[str, float | int | str] = {
        "accuracy": accuracy_score(expected_labels, predicted_labels),
        "f1_micro": f1_score(expected_labels, predicted_labels, average="micro"),
        "f1_macro": f1_score(expected_labels, predicted_labels, average="macro"),
        "precision_micro": precision_score(expected_labels, predicted_labels, average="micro"),
        "precision_macro": precision_score(expected_labels, predicted_labels, average="macro"),
        "recall_micro": recall_score(expected_labels, predicted_labels, average="micro"),
        "recall_macro": recall_score(expected_labels, predicted_labels, average="macro"),
    }
    labels = (
        dataset_label_classes
        if dataset_label_classes is not None
        else np.unique(np.concatenate([expected_labels, predicted_labels]))
    )
    if len(labels) == 2:
        encoded_labels = (
            np.arange(len(dataset_label_classes)) if dataset_label_classes is not None else None
        )
        matrix = confusion_matrix(expected_labels, predicted_labels, labels=encoded_labels)
        metrics.update(
            {
                "confusion_matrix_tn": matrix[0, 0],
                "confusion_matrix_fp": matrix[0, 1],
                "confusion_matrix_fn": matrix[1, 0],
                "confusion_matrix_tp": matrix[1, 1],
            }
        )
    return FoldResult(fold=fold, dataset_name=dataset_name, metrics=metrics)


def train_and_evaluate_classifier(
    dataset: PreparedDataset,
    fold: FoldSplit,
    pretraining: PretrainingOutcome,
    hyperparameters: Hyperparameters,
    device: torch.device,
    tracker: TrainingTracker,
) -> FoldResult:
    """Fine-tune and evaluate the classifier using the legacy optimization loop."""
    train_frame = dataset.frame.iloc[fold.train_indices].reset_index(drop=True)
    validation_frame = dataset.frame.iloc[fold.validation_indices].reset_index(drop=True)
    test_frame = dataset.frame.iloc[fold.test_indices].reset_index(drop=True)
    train_labels = torch.tensor(train_frame[dataset.label_column].values, dtype=torch.long, device=device)
    validation_labels = torch.tensor(
        validation_frame[dataset.label_column].values, dtype=torch.long, device=device
    )
    test_labels = torch.tensor(test_frame[dataset.label_column].values, dtype=torch.long, device=device)

    processed_train = preprocess_table(
        train_frame.drop(columns=[dataset.label_column]), p_base=0.0, fine_tunning=True
    )
    processed_validation = preprocess_table(
        validation_frame.drop(columns=[dataset.label_column]), p_base=0.0, fine_tunning=True
    )
    processed_test = preprocess_table(
        test_frame.drop(columns=[dataset.label_column]), p_base=0.0, fine_tunning=True
    )

    number_of_labels = len(dataset.label_classes)
    present_classes = np.unique(train_labels.cpu().numpy())
    class_weights_array = np.ones(number_of_labels, dtype=np.float32)
    if len(present_classes) > 0:
        computed_weights = compute_class_weight(
            class_weight="balanced", classes=present_classes, y=train_labels.cpu().numpy()
        )
        for label, weight in zip(present_classes, computed_weights):
            class_weights_array[label] = weight
    class_weights = torch.tensor(class_weights_array, dtype=torch.float32, device=device)
    model = TridentModel(
        embedder=pretraining.model.embedder,
        transformer=pretraining.model.transformer,
        num_labels=number_of_labels,
        class_weights=class_weights,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparameters.finetuning_learning_rate,
        weight_decay=hyperparameters.finetuning_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hyperparameters.finetuning_epochs
    )
    best_validation_loss = float("inf")
    best_model_state = None

    print("\n=== Starting Fine-Tuning (Classification) ===")
    for epoch in tqdm(range(hyperparameters.finetuning_epochs), desc="Fine-tuning epochs"):
        model.train()
        indices = torch.randperm(len(processed_train))
        train_loss_sum = 0.0
        train_batch_count = 0
        for start in tqdm(
            range(0, len(processed_train), hyperparameters.batch_size),
            desc=f"FineTune Epoch {epoch + 1}/{hyperparameters.finetuning_epochs}",
            unit="batch",
            leave=False,
        ):
            batch_indices = indices[start : start + hyperparameters.batch_size]
            logits, loss = model(processed_train.iloc[batch_indices], labels=train_labels[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item()
            train_batch_count += 1
        average_train_loss = train_loss_sum / train_batch_count

        model.eval()
        with torch.no_grad():
            validation_logits, validation_loss = model(
                processed_validation, labels=validation_labels
            )
            validation_predictions = torch.argmax(validation_logits, dim=1)
        validation_loss_value = validation_loss.item()
        validation_expected = validation_labels.cpu().numpy()
        validation_predicted = validation_predictions.cpu().numpy()
        tracker.log_metrics(
            {
                "finetune/train_loss": float(average_train_loss),
                "finetune/val_loss": float(validation_loss_value),
                "finetune/val_f1_micro": float(
                    f1_score(validation_expected, validation_predicted, average="micro")
                ),
                "finetune/val_f1_macro": float(
                    f1_score(validation_expected, validation_predicted, average="macro")
                ),
            },
            step=epoch,
        )
        if validation_loss_value < best_validation_loss:
            best_validation_loss = validation_loss_value
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_logits, test_loss = model(processed_test, labels=test_labels)
        test_predictions = torch.argmax(test_logits, dim=1)
    result = build_fold_result(
        fold="single_split",
        dataset_name=dataset.frame.attrs.get("dataset_name", ""),
        expected_labels=test_labels.cpu().numpy(),
        predicted_labels=test_predictions.cpu().numpy(),
        test_loss=float(test_loss.item()),
        dataset_label_classes=dataset.label_classes,
    )
    tracker.log_metrics(
        {
            **{f"test/{name}": float(value) for name, value in result.metrics.items()},
            "test/loss": float(test_loss.item()),
        }
    )
    return result
