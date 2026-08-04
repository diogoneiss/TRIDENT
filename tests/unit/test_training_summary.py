from dataclasses import replace
from math import nan, sqrt

import pytest

from src.training.summary import summarize_cross_validation
from src.training.types import FoldResult, FoldTrackingRecord, LoggedMetric


def _record(fold: int, f1_macro: float) -> FoldTrackingRecord:
    return FoldTrackingRecord(
        result=FoldResult(
            fold,
            "vehicle_00nan",
            {
                "accuracy": 0.5 + f1_macro / 2,
                "f1_macro": f1_macro,
            },
        ),
        metric_events=(
            LoggedMetric("pretrain/train_loss", 1.0 + fold, 0),
            LoggedMetric("pretrain/val_loss", 2.0 + fold, 0),
            LoggedMetric("finetune/train_loss", 3.0 + fold, 0),
            LoggedMetric("finetune/val_loss", 4.0 + fold, 0),
        ),
        artifacts=(),
    )


def test_summarize_cross_validation_logs_statistics_and_loss_bands() -> None:
    summary = summarize_cross_validation([_record(1, 0.4), _record(2, 0.8)])

    stats = summary.metrics["f1_macro"]
    assert stats.mean == pytest.approx(0.6)
    assert stats.std == pytest.approx(sqrt(0.08))
    assert stats.fold_count == 2
    assert stats.minimum == pytest.approx(0.4)
    assert stats.maximum == pytest.approx(0.8)
    assert stats.ci95_lower < stats.mean < stats.ci95_upper
    assert summary.diagnostic_roles == {1: "worst_fold", 2: "best_fold"}
    assert summary.loss_bands["finetune/val_loss"][0].mean == pytest.approx(5.5)


def test_summarize_cross_validation_uses_one_role_for_a_tie() -> None:
    summary = summarize_cross_validation([_record(1, 0.5), _record(2, 0.5)])

    assert summary.diagnostic_roles == {1: "best_and_worst"}


def test_summarize_cross_validation_rejects_fewer_than_two_folds() -> None:
    with pytest.raises(ValueError, match="at least two"):
        summarize_cross_validation([_record(1, 0.5)])


def test_summarize_cross_validation_rejects_missing_final_metric() -> None:
    incomplete = replace(_record(2, 0.8), result=FoldResult(2, "vehicle_00nan", {"f1_macro": 0.8}))

    with pytest.raises(ValueError, match="metric"):
        summarize_cross_validation([_record(1, 0.4), incomplete])


def test_summarize_cross_validation_rejects_non_finite_final_metric() -> None:
    non_finite = replace(
        _record(2, 0.8),
        result=FoldResult(2, "vehicle_00nan", {"accuracy": 0.9, "f1_macro": nan}),
    )

    with pytest.raises(ValueError, match="finite"):
        summarize_cross_validation([_record(1, 0.4), non_finite])


def test_summarize_cross_validation_rejects_mismatched_loss_epochs() -> None:
    mismatched = replace(
        _record(2, 0.8),
        metric_events=(
            LoggedMetric("pretrain/train_loss", 3.0, 0),
            LoggedMetric("pretrain/val_loss", 4.0, 0),
            LoggedMetric("finetune/train_loss", 5.0, 0),
            LoggedMetric("finetune/val_loss", 6.0, 1),
        ),
    )

    with pytest.raises(ValueError, match="steps"):
        summarize_cross_validation([_record(1, 0.4), mismatched])
