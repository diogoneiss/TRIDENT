"""Filesystem artifacts emitted by a training runtime."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .types import FoldResult, Hyperparameters


class ArtifactWriter:
    def __init__(self, output_dir: Path, metrics_dir: Path, dataset_name: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = output_dir / dataset_name / timestamp
        self.metrics_dir = metrics_dir
        self.dataset_name = dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def write_hyperparameters(self, hyperparameters: Hyperparameters) -> Path:
        values = {
            "DIM": hyperparameters.dimension,
            "HIDDEN_DIM": hyperparameters.hidden_dimension,
            "HEADS": hyperparameters.heads,
            "LAYERS": hyperparameters.layers,
            "DIM_FEED": hyperparameters.feedforward_dimension,
            "DROPOUT": hyperparameters.dropout,
            "EPOCHS_PRE": hyperparameters.pretraining_epochs,
            "BATCH": hyperparameters.batch_size,
            "LR_PRE": hyperparameters.pretraining_learning_rate,
            "WEIGHT_DECAY_PRE": hyperparameters.pretraining_weight_decay,
            "PROB_MASCARA": hyperparameters.mask_probability,
            "EPOCH_FINE": hyperparameters.finetuning_epochs,
            "LR_FINE": hyperparameters.finetuning_learning_rate,
            "WEIGHT_DECAY_FINE": hyperparameters.finetuning_weight_decay,
            "LABELS": hyperparameters.labels,
        }
        path = self.results_dir / "hyperparameters.json"
        path.write_text(json.dumps(values, indent=4))
        print(f"Hyperparameters saved to: {path}")
        return path

    def write_metrics(self, fold_results: list[FoldResult]) -> pd.DataFrame:
        rows = [{"fold": result.fold, "dataset": result.dataset_name, **result.metrics} for result in fold_results]
        frame = pd.DataFrame(rows)
        result_path = self.results_dir / "metrics.csv"
        frame.to_csv(result_path, index=False)
        print(f"Metrics saved to: {result_path}")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        root_path = self.metrics_dir / f"{self.dataset_name}_metrics.csv"
        frame.to_csv(root_path, index=False)
        print(f"Metrics also saved to: {root_path}")
        return frame

    def write_loss_plot(self, name: str, train_losses: list[float], validation_losses: list[float]) -> Path:
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        path = plots_dir / name
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(name.removesuffix(".png").replace("_", " ").title())
        plt.legend()
        plt.savefig(path)
        plt.close()
        return path

    def save_model(self, name: str, model: object) -> Path:
        path = self.results_dir / name
        torch.save(model, path)
        print(f"Final model saved to: {path}")
        return path
