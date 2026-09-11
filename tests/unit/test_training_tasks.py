"""The per-task fold-ranking contract (ADR 0004, decisions 6 and 12)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training.artifacts import ArtifactWriter
from src.training.summary import summarize_cross_validation
from src.training.types import (
    IMPUTATION,
    CrossValidationSummary,
    FoldResult,
    FoldTrackingRecord,
    LoggedMetric,
    MetricSummary,
    PreparedDataset,
    task_spec,
)


def test_each_task_declares_its_search_objective() -> None:
    """A search ranks its trials by its own objective, which need not be the fold-ranking
    metric: imputation scores its search on the validation split so the test split never
    chooses hyperparameters, while classification keeps the test-split macro F1 it always
    used (ADR 0005, decision 3).
    """
    assert task_spec("imputation").search_objective == "validation/impute/masked/impute_score"
    assert task_spec("classification").search_objective == "f1_macro"


def _record(fold: int, metrics: dict[str, float], loss_stage: str = "decode") -> FoldTrackingRecord:
    """One cross-validation fold of an imputation run, logging decode-stage losses."""
    return FoldTrackingRecord(
        result=FoldResult(fold, "credit-g_20nan", metrics),
        metric_events=(
            LoggedMetric("pretrain/train_loss", 1.0 + fold, 0),
            LoggedMetric("pretrain/val_loss", 2.0 + fold, 0),
            LoggedMetric(f"{loss_stage}/train_loss", 3.0 + fold, 0),
            LoggedMetric(f"{loss_stage}/val_loss", 4.0 + fold, 0),
        ),
        artifacts=(),
    )


def test_classification_ranks_folds_by_macro_f1_maximised() -> None:
    spec = task_spec("classification")

    assert (spec.ranking_metric, spec.direction) == ("f1_macro", "maximize")


def test_imputation_ranks_folds_by_impute_score_minimised() -> None:
    spec = task_spec("imputation")

    assert (spec.ranking_metric, spec.direction) == ("impute/masked/impute_score", "minimize")


def test_unknown_task_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="regression"):
        task_spec("regression")


def test_imputation_summary_rejects_folds_missing_the_impute_score() -> None:
    records = [
        _record(1, {"impute/masked/rmse_num_z": 0.5}),
        _record(2, {"impute/masked/rmse_num_z": 0.6}),
    ]

    with pytest.raises(ValueError, match="impute/masked/impute_score"):
        summarize_cross_validation(records, task=IMPUTATION)


def test_imputation_summary_marks_the_lowest_score_as_the_best_fold() -> None:
    records = [
        _record(1, {"impute/masked/impute_score": 0.9}),
        _record(2, {"impute/masked/impute_score": 0.4}),
    ]

    summary = summarize_cross_validation(records, task=IMPUTATION)

    assert summary.diagnostic_roles == {2: "best_fold", 1: "worst_fold"}


def test_imputation_summary_bands_the_decode_stage_losses_instead_of_finetuning() -> None:
    records = [
        _record(1, {"impute/masked/impute_score": 0.9}, loss_stage="decode"),
        _record(2, {"impute/masked/impute_score": 0.4}, loss_stage="decode"),
    ]

    summary = summarize_cross_validation(records, task=IMPUTATION)

    assert set(summary.loss_bands) == {
        "pretrain/train_loss", "pretrain/val_loss", "decode/train_loss", "decode/val_loss",
    }
    assert summary.loss_bands["decode/train_loss"][0].mean == pytest.approx(4.5)


def test_imputation_summary_carries_the_decode_stage_timing() -> None:
    def _timed(fold: int) -> FoldTrackingRecord:
        record = _record(fold, {"impute/masked/impute_score": 0.5}, loss_stage="decode")
        return FoldTrackingRecord(
            result=record.result,
            metric_events=record.metric_events + (
                LoggedMetric("time/pretrain_seconds", 10.0, None),
                LoggedMetric("time/decode_seconds", 4.0, None),
                LoggedMetric("time/total_seconds", 14.0, None),
            ),
            artifacts=(),
        )

    summary = summarize_cross_validation([_timed(1), _timed(2)], task=IMPUTATION)

    assert set(summary.timings) == {"pretrain_seconds", "decode_seconds", "total_seconds"}


def test_imputation_manifest_ranks_folds_under_the_impute_score_name(tmp_path) -> None:
    writer = ArtifactWriter(tmp_path / "results", tmp_path / "metrics", "credit-g_20nan")
    records = [
        _record(1, {"impute/masked/impute_score": 0.9}),
        _record(2, {"impute/masked/impute_score": 0.4}),
    ]
    summary = CrossValidationSummary(
        metrics={
            "impute/masked/impute_score": MetricSummary(
                mean=0.65, ci95_lower=0.0, ci95_upper=1.3, std=0.35, minimum=0.4, maximum=0.9, fold_count=2
            )
        },
        loss_bands={},
        diagnostic_roles={1: "worst_fold", 2: "best_fold"},
    )
    frame = pd.DataFrame({"amount": [0.0, 1.0], "class": [0, 1]})
    frame.attrs["dataset_name"] = "credit-g_20nan"
    dataset = PreparedDataset(
        frame=frame,
        label_column="class",
        categorical_columns=(),
        numerical_columns=("amount",),
        label_classes=np.array(["bad", "good"]),
        source_path=Path("datasets/processed_datasets/credit-g/credit-g_20nan.csv"),
        splits_path=Path("datasets/processed_datasets/splits/credit-g_split.json"),
    )

    paths = writer.write_cv_tracking_artifacts(
        records, summary, dataset, seed=42, cv_folds=2, task=IMPUTATION
    )

    manifest = json.loads(paths.manifest_json.read_text())
    assert manifest["impute_score_ranking"] == {"1": 0.9, "2": 0.4}
    assert manifest["selected_folds"] == [
        {"fold": 1, "role": "worst_fold", "impute_score": 0.9},
        {"fold": 2, "role": "best_fold", "impute_score": 0.4},
    ]
