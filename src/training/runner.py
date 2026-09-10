"""Typed orchestration for TRIDENT training."""

import time

import torch

from src.utils import set_global_seed

from .artifacts import ArtifactWriter
from .data import build_folds, prepare_dataset
from .environment import runtime_environment_tags
from .finetuning import train_and_evaluate_classifier
from .pretraining import train_pretrainer
from .summary import compute_cv_summary, stage_timing_metrics, summarize_cross_validation
from .tracking import create_tracker
from .types import FoldResult, TrainingRequest, TrainingResult


def _mlflow_hyperparameters(request: TrainingRequest) -> dict[str, object]:
    hyperparameters = request.hyperparameters
    return {
        "DIM": hyperparameters.dimension, "HIDDEN_DIM": hyperparameters.hidden_dimension,
        "HEADS": hyperparameters.heads, "LAYERS": hyperparameters.layers,
        "DIM_FEED": hyperparameters.feedforward_dimension, "DROPOUT": hyperparameters.dropout,
        "EPOCHS_PRE": hyperparameters.pretraining_epochs, "BATCH": hyperparameters.batch_size,
        "LR_PRE": hyperparameters.pretraining_learning_rate,
        "WEIGHT_DECAY_PRE": hyperparameters.pretraining_weight_decay,
        "PROB_MASCARA": hyperparameters.mask_probability,
        "EPOCH_FINE": hyperparameters.finetuning_epochs,
        "LR_FINE": hyperparameters.finetuning_learning_rate,
        "WEIGHT_DECAY_FINE": hyperparameters.finetuning_weight_decay,
        "LABELS": hyperparameters.labels,
        "LR_SCHEDULER": hyperparameters.lr_scheduler,
    }


def run_training(request: TrainingRequest) -> TrainingResult:
    """Run all folds and return raw folds plus the legacy aggregate metrics."""
    set_global_seed(request.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset = prepare_dataset(request.dataset)
    folds = build_folds(dataset.frame, dataset.label_column, request.cv_folds, request.seed)
    artifacts = ArtifactWriter(request.runtime.output_dir, request.runtime.metrics_dir, request.dataset.dataset_name)
    tracker = create_tracker(request.runtime.tracking_enabled)
    results: list[FoldResult] = []
    records = []

    with tracker.parent_run(
        dataset_name=request.dataset.dataset_name,
        seed=request.seed,
        cv_folds=request.cv_folds,
        hyperparameters=_mlflow_hyperparameters(request),
        lr_scheduler=request.hyperparameters.lr_scheduler,
        environment=runtime_environment_tags(device),
    ) as active_tracker:
        for ordinal, fold in enumerate(folds, start=1):
            if request.cv_folds is not None:
                print(f"\n==================== Running Fold {ordinal}/{request.cv_folds} ====================")
            with active_tracker.fold_run(
                fold=ordinal, cv_folds=request.cv_folds, dataset_name=request.dataset.dataset_name
            ) as fold_tracker:
                pretraining_started = time.perf_counter()
                pretraining = train_pretrainer(
                    dataset, fold, request.hyperparameters, device, fold_tracker
                )
                pretraining_seconds = time.perf_counter() - pretraining_started
                finetuning_started = time.perf_counter()
                finetuning = train_and_evaluate_classifier(
                    dataset,
                    fold,
                    pretraining,
                    request.hyperparameters,
                    device,
                    fold_tracker,
                )
                finetuning_seconds = time.perf_counter() - finetuning_started
                # Both stages end with host-side ``.item()`` reads, so GPU work
                # has drained when the clock is read. Logged before plots and
                # model saving so the timing covers training only.
                fold_tracker.log_metrics(
                    stage_timing_metrics(pretraining_seconds, finetuning_seconds)
                )
                if request.plot_losses:
                    suffix = f"_fold_{ordinal}" if request.cv_folds is not None else ""
                    pretraining_plot = artifacts.write_loss_plot(
                        f"pretrain_losses{suffix}.png",
                        list(pretraining.train_losses),
                        list(pretraining.validation_losses),
                    )
                    finetuning_plot = artifacts.write_loss_plot(
                        f"finetune_losses{suffix}.png",
                        list(finetuning.train_losses),
                        list(finetuning.validation_losses),
                    )
                    fold_tracker.log_artifact(str(pretraining_plot), artifact_path="plots")
                    fold_tracker.log_artifact(str(finetuning_plot), artifact_path="plots")
                if request.save_model:
                    suffix = f"_fold_{ordinal}" if request.cv_folds is not None else ""
                    model_path = artifacts.save_model(f"final_model{suffix}.pt", finetuning.model)
                    fold_tracker.log_artifact(str(model_path), artifact_path="models")
                result = FoldResult(
                    fold=ordinal if request.cv_folds is not None else "single_split",
                    dataset_name=request.dataset.dataset_name,
                    metrics=finetuning.result.metrics,
                )
                results.append(result)
                records.append(fold_tracker.to_record(result))

        hyperparameters_path = artifacts.write_hyperparameters(request.hyperparameters)
        frame = artifacts.write_metrics(results)
        active_tracker.log_artifact(
            str(hyperparameters_path), artifact_path="parameters"
        )
        active_tracker.log_artifact(
            str(artifacts.results_dir / "metrics.csv"), artifact_path="metrics"
        )
        if request.cv_folds is not None:
            tracking_summary = summarize_cross_validation(records)
            artifact_paths = artifacts.write_cv_tracking_artifacts(
                records,
                tracking_summary,
                dataset,
                request.seed,
                request.cv_folds,
            )
            active_tracker.finalize_cross_validation(
                records, tracking_summary, dataset, artifact_paths
            )
        else:
            provenance_path = artifacts.write_tracking_provenance(
                dataset, request.seed, request.cv_folds
            )
            active_tracker.log_prepared_dataset(dataset)
            active_tracker.log_artifact(str(provenance_path), artifact_path="data")
            active_tracker.log_single_split_record(records[0])

    if request.cv_folds is not None and len(results) > 1:
        summary = compute_cv_summary(frame)
        print("\n=== Cross-Validation Results Summary ===")
        for name, statistics in summary.items():
            print(f"{name:<20}: {statistics['mean_std']}")
        print("=========================================\n")
        return TrainingResult(results, {name: float(statistics["mean"]) for name, statistics in summary.items()})
    return TrainingResult(results, {name: float(value) for name, value in results[0].metrics.items() if isinstance(value, (int, float))})
