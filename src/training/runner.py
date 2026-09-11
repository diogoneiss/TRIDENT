"""Typed orchestration for TRIDENT training."""

import time

import pandas as pd
import torch

from src.utils import set_global_seed

from .artifacts import ArtifactWriter
from .config import logged_hyperparameters
from .data import build_folds, load_complete_sibling, prepare_dataset
from .decoding import train_and_evaluate_decoder
from .environment import runtime_environment_tags
from .finetuning import train_and_evaluate_classifier
from .imputation_metrics import mean_mode_baselines, score_cells
from .pretraining import train_pretrainer
from .summary import compute_cv_summary, stage_timing_metrics, summarize_cross_validation
from .tracking import create_tracker
from .types import (
    DecodingOutcome,
    FinetuningOutcome,
    FoldKey,
    FoldResult,
    FoldSplit,
    PreparedDataset,
    TrainingRequest,
    TrainingResult,
    task_spec,
)


def _per_column_scores(
    dataset: PreparedDataset, fold: FoldSplit, scored_cells: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Each population's per-column errors, against this fold's own naive baseline."""
    features = dataset.frame.drop(columns=[dataset.label_column])
    train_frame = features.iloc[fold.train_indices].reset_index(drop=True)
    baselines = mean_mode_baselines(
        train_frame, dataset.numerical_columns, dataset.categorical_columns
    )
    # ``groupby`` keys are pandas scalars, not necessarily ``str``; the population column
    # holds strings, so naming that explicitly costs nothing and states the assumption.
    return {
        str(population): score_cells(group, baselines).per_column
        for population, group in scored_cells.groupby("population", sort=True)
    }


def run_training(request: TrainingRequest) -> TrainingResult:
    """Run all folds and return raw folds plus the legacy aggregate metrics."""
    set_global_seed(request.seed)
    task = task_spec(request.task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset = prepare_dataset(request.dataset)
    # Read once per run, not per fold, and only when there is something to score against.
    complete_sibling = (
        load_complete_sibling(request.dataset) if task.name == "imputation" else None
    )
    folds = build_folds(dataset.frame, dataset.label_column, request.cv_folds, request.seed)
    artifacts = ArtifactWriter(request.runtime.output_dir, request.runtime.metrics_dir, request.dataset.dataset_name)
    tracker = create_tracker(
        request.runtime.tracking_enabled, request.runtime.tracking_run_role
    )
    results: list[FoldResult] = []
    records = []
    # Per-column errors, gathered per fold and written once on the parent.
    # Narrowed from ``object`` now that ``_per_column_scores`` states what it returns; this
    # is what ``write_per_column_imputation`` has always required.
    per_column: dict[FoldKey, dict[str, pd.DataFrame]] = {}

    with tracker.parent_run(
        dataset_name=request.dataset.dataset_name,
        seed=request.seed,
        cv_folds=request.cv_folds,
        hyperparameters=logged_hyperparameters(request),
        lr_scheduler=request.hyperparameters.lr_scheduler,
        environment=runtime_environment_tags(device),
        extra_tags=request.runtime.tracking_tags,
        task=task.name,
        config_source=request.config_source,
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
                # The two tasks produce different outcomes; the union states the contract
                # that already holds at runtime. Exhaustiveness (``assert_never``) waits on
                # the task enum, which cannot narrow a ``str`` comparison.
                finetuning: FinetuningOutcome | DecodingOutcome
                if task.name == "imputation":
                    finetuning = train_and_evaluate_decoder(
                        dataset=dataset,
                        fold=fold,
                        pretraining=pretraining,
                        hyperparameters=request.hyperparameters,
                        device=device,
                        tracker=fold_tracker,
                        seed=request.seed,
                        fold_ordinal=ordinal,
                        complete_sibling=complete_sibling,
                        score_null_path=request.score_null_path,
                        score_search_objective=request.score_search_objective,
                    )
                else:
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
                    # Named after the stage that produced the curves, so an imputation
                    # run's plot does not claim to be fine-tuning's.
                    finetuning_plot = artifacts.write_loss_plot(
                        f"{task.stage}_losses{suffix}.png",
                        list(finetuning.train_losses),
                        list(finetuning.validation_losses),
                    )
                    fold_tracker.log_artifact(str(pretraining_plot), artifact_path="plots")
                    fold_tracker.log_artifact(str(finetuning_plot), artifact_path="plots")
                # Narrowed on the outcome type rather than the task name: ``scored_cells``
                # belongs to the decode stage's outcome, and the decoder is exactly what
                # the imputation branch above produced. Same runs take this path as before,
                # but the attribute access is now checked instead of merely correlated.
                if isinstance(finetuning, DecodingOutcome):
                    suffix = f"_fold_{ordinal}" if request.cv_folds is not None else ""
                    preview_path, ledger_path = artifacts.write_imputation_preview(
                        f"imputation{suffix}",
                        finetuning.scored_cells,
                        seed=request.seed,
                        fold=ordinal,
                        scaler=dataset.scaler,
                        numerical_columns=dataset.numerical_columns,
                    )
                    fold_tracker.log_artifact(str(preview_path), artifact_path="imputation")
                    fold_tracker.log_artifact(str(ledger_path), artifact_path="imputation")
                    per_column[ordinal] = _per_column_scores(
                        dataset, fold, finetuning.scored_cells
                    )
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

        if per_column:
            active_tracker.log_artifact(
                str(artifacts.write_per_column_imputation(per_column)), artifact_path="metrics"
            )
        hyperparameters_path = artifacts.write_hyperparameters(request.hyperparameters, task)
        frame = artifacts.write_metrics(results)
        active_tracker.log_artifact(
            str(hyperparameters_path), artifact_path="parameters"
        )
        active_tracker.log_artifact(
            str(artifacts.results_dir / "metrics.csv"), artifact_path="metrics"
        )
        if request.cv_folds is not None:
            tracking_summary = summarize_cross_validation(records, task)
            artifact_paths = artifacts.write_cv_tracking_artifacts(
                records,
                tracking_summary,
                dataset,
                request.seed,
                request.cv_folds,
                task,
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
