from dataclasses import replace
from math import nan, sqrt

import pytest

from src.training.summary import (
    fold_timings_for_tracking,
    stage_timing_metrics,
    summarize_cross_validation,
)
from src.training.types import FoldResult, FoldTrackingRecord, LoggedMetric, task_spec


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
            LoggedMetric("test/loss", 0.1 * fold, None),
            LoggedMetric("time/pretrain_seconds", 10.0 * fold, None),
            LoggedMetric("time/finetune_seconds", 5.0 * fold, None),
            LoggedMetric("time/total_seconds", 15.0 * fold, None),
        ),
        artifacts=(),
    )


def _without_timings(record: FoldTrackingRecord) -> FoldTrackingRecord:
    return replace(
        record,
        metric_events=tuple(
            event for event in record.metric_events if not event.key.startswith("time/")
        ),
    )


def test_summarize_cross_validation_logs_statistics_and_loss_bands() -> None:
    summary = summarize_cross_validation([_record(1, 0.4), _record(2, 0.8)])

    stats = summary.metrics["f1_macro"]
    assert stats.mean == pytest.approx(0.6)
    assert stats.std == pytest.approx(sqrt(0.08))
    assert stats.fold_count == 2
    assert stats.minimum == pytest.approx(0.4)
    assert stats.maximum == pytest.approx(0.8)
    assert stats.ci95_lower == pytest.approx(-1.94124094723494)
    assert stats.ci95_upper == pytest.approx(3.14124094723494)
    assert summary.metrics["loss"].mean == pytest.approx(0.15)
    assert "pretrain_seconds" not in summary.metrics
    assert set(summary.timings) == {"pretrain_seconds", "finetune_seconds", "total_seconds"}
    assert summary.timings["total_seconds"].mean == pytest.approx(22.5)
    assert summary.timings["total_seconds"].fold_count == 2
    assert summary.timings["pretrain_seconds"].minimum == pytest.approx(10.0)
    assert summary.timings["finetune_seconds"].maximum == pytest.approx(10.0)
    assert summary.diagnostic_roles == {1: "worst_fold", 2: "best_fold"}
    loss_band = summary.loss_bands["finetune/val_loss"][0]
    assert loss_band.mean == pytest.approx(5.5)
    assert loss_band.ci95_lower == pytest.approx(-0.85310236808735)
    assert loss_band.ci95_upper == pytest.approx(11.85310236808735)


def test_summarize_cross_validation_uses_one_role_for_a_tie() -> None:
    summary = summarize_cross_validation([_record(1, 0.5), _record(2, 0.5)])

    assert summary.diagnostic_roles == {1: "best_and_worst"}


def test_summarize_cross_validation_uses_lowest_fold_for_a_minimum_tie() -> None:
    summary = summarize_cross_validation([_record(1, 0.2), _record(2, 0.2), _record(3, 0.8)])

    assert summary.diagnostic_roles == {1: "worst_fold", 3: "best_fold"}


def test_summarize_cross_validation_uses_lowest_fold_for_a_maximum_tie() -> None:
    summary = summarize_cross_validation([_record(1, 0.2), _record(2, 0.8), _record(3, 0.8)])

    assert summary.diagnostic_roles == {1: "worst_fold", 2: "best_fold"}


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
            LoggedMetric("test/loss", 0.2, None),
        ),
    )

    with pytest.raises(ValueError, match="steps"):
        summarize_cross_validation([_record(1, 0.4), mismatched])


def test_summarize_cross_validation_has_empty_timings_without_timing_events() -> None:
    summary = summarize_cross_validation(
        [_without_timings(_record(1, 0.4)), _without_timings(_record(2, 0.8))]
    )

    assert summary.timings == {}
    assert summary.metrics["f1_macro"].mean == pytest.approx(0.6)


def test_summarize_cross_validation_rejects_timings_missing_on_one_fold() -> None:
    with pytest.raises(ValueError, match="timing keys"):
        summarize_cross_validation([_record(1, 0.4), _without_timings(_record(2, 0.8))])


def test_summarize_cross_validation_rejects_duplicate_timing_events() -> None:
    duplicated = replace(
        _record(2, 0.8),
        metric_events=_record(2, 0.8).metric_events
        + (LoggedMetric("time/total_seconds", 1.0, None),),
    )

    with pytest.raises(ValueError, match="at most one"):
        summarize_cross_validation([_record(1, 0.4), duplicated])


def test_summarize_cross_validation_rejects_negative_timing() -> None:
    negative = replace(
        _without_timings(_record(2, 0.8)),
        metric_events=_without_timings(_record(2, 0.8)).metric_events
        + (
            LoggedMetric("time/pretrain_seconds", -1.0, None),
            LoggedMetric("time/finetune_seconds", 1.0, None),
            LoggedMetric("time/total_seconds", 0.0, None),
        ),
    )

    with pytest.raises(ValueError, match="non-negative"):
        summarize_cross_validation([_record(1, 0.4), negative])


def test_fold_timings_for_tracking_ignores_stepped_timing_events() -> None:
    record = replace(
        _without_timings(_record(1, 0.4)),
        metric_events=_without_timings(_record(1, 0.4)).metric_events
        + (LoggedMetric("time/total_seconds", 9.0, 3),),
    )

    assert fold_timings_for_tracking(record) == {}
    assert fold_timings_for_tracking(_record(2, 0.8)) == {
        "pretrain_seconds": 20.0,
        "finetune_seconds": 10.0,
        "total_seconds": 30.0,
    }


def test_stage_timing_metrics_sums_the_two_stages() -> None:
    assert stage_timing_metrics(2.0, 1.5) == {
        "time/pretrain_seconds": 2.0,
        "time/finetune_seconds": 1.5,
        "time/total_seconds": 3.5,
    }


def _without_pretraining(record: FoldTrackingRecord) -> FoldTrackingRecord:
    return replace(
        record,
        metric_events=tuple(
            event for event in record.metric_events if not event.key.startswith("pretrain/")
        ),
    )


def test_a_stage_no_fold_ran_has_no_loss_band() -> None:
    """A run may skip pre-training (zero epochs) to measure what it contributes. No fold
    then logs a pre-training loss, and the summary must say nothing about that stage
    rather than refuse the whole run; the stage that did train keeps its bands."""
    summary = summarize_cross_validation(
        [_without_pretraining(_record(1, 0.4)), _without_pretraining(_record(2, 0.8))]
    )

    assert "pretrain/train_loss" not in summary.loss_bands
    assert "pretrain/val_loss" not in summary.loss_bands
    assert set(summary.loss_bands) == {"finetune/train_loss", "finetune/val_loss"}


def test_a_stage_only_some_folds_ran_is_still_refused() -> None:
    """Folds that disagree about whether a stage ran are not comparable, as before."""
    with pytest.raises(ValueError, match="pretrain/train_loss"):
        summarize_cross_validation([_without_pretraining(_record(1, 0.4)), _record(2, 0.8)])


def _stopped_at(fold: int, epochs: int, stage: str = "decode") -> FoldTrackingRecord:
    """An imputation fold whose decode stage stopped after ``epochs`` epochs (--decode_patience)."""
    losses = tuple(
        LoggedMetric(key, 10.0 * fold + step, step)
        for step in range(epochs)
        for key in (f"{stage}/train_loss", f"{stage}/val_loss")
    )
    return FoldTrackingRecord(
        result=FoldResult(
            fold,
            "credit-g_20nan",
            {"impute/masked/impute_score": 0.9 + 0.01 * fold, "f1_macro": 0.5, "accuracy": 0.5},
        ),
        metric_events=(
            LoggedMetric("pretrain/train_loss", 1.0 + fold, 0),
            LoggedMetric("pretrain/val_loss", 2.0 + fold, 0),
            *losses,
        ),
        artifacts=(),
    )


def test_folds_that_stopped_at_different_epochs_share_the_bands_they_reached() -> None:
    """With early stopping each fold decodes for its own number of epochs. A band averages
    the folds that reached its step, and a step only one fold reached has no interval."""
    imputation = task_spec("imputation")
    summary = summarize_cross_validation(
        [_stopped_at(1, 3), _stopped_at(2, 5), _stopped_at(3, 5)], imputation
    )

    bands = summary.loss_bands["decode/val_loss"]
    assert [band.step for band in bands] == [0, 1, 2, 3, 4]
    assert bands[0].mean == pytest.approx((10.0 + 20.0 + 30.0) / 3)
    assert bands[3].mean == pytest.approx((23.0 + 33.0) / 2)

    lonely = summarize_cross_validation([_stopped_at(1, 2), _stopped_at(2, 4)], imputation)
    assert [band.step for band in lonely.loss_bands["decode/val_loss"]] == [0, 1]


def test_only_the_decode_stage_may_end_folds_at_different_epochs() -> None:
    """No other stage stops early, so a ragged pre-training or fine-tuning curve is a bug."""
    with pytest.raises(ValueError, match="steps"):
        summarize_cross_validation([_stopped_at(1, 3, "finetune"), _stopped_at(2, 5, "finetune")])
