"""Metric aggregation helpers."""

from math import isfinite, sqrt
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t

from src.training.types import (
    CrossValidationSummary,
    FoldTrackingRecord,
    LoggedMetric,
    LossBand,
    MetricSummary,
)


_LOSS_KEYS = (
    "pretrain/train_loss",
    "pretrain/val_loss",
    "finetune/train_loss",
    "finetune/val_loss",
)


def summarize_cross_validation(records: Sequence[FoldTrackingRecord]) -> CrossValidationSummary:
    """Aggregate completed CV folds into comparable final and loss statistics."""
    if len(records) < 2:
        raise ValueError("Cross-validation summary requires at least two folds.")

    final_metrics = tuple(final_metrics_for_tracking(record) for record in records)
    metric_keys = _validate_final_metrics(records, final_metrics)
    metrics = {
        key: _summarize_metric([float(metrics[key]) for metrics in final_metrics])
        for key in metric_keys
    }
    loss_bands = _summarize_loss_bands(records)
    return CrossValidationSummary(
        metrics=metrics,
        loss_bands=loss_bands,
        diagnostic_roles=_diagnostic_roles(records),
    )


def final_metrics_for_tracking(record: FoldTrackingRecord) -> dict[str, float | int | str]:
    """Return a fold's legacy metrics plus its tracked step-less test loss."""
    metrics = dict(record.result.metrics)
    test_loss_events = [
        event
        for event in record.metric_events
        if event.key == "test/loss" and event.step is None
    ]
    if len(test_loss_events) > 1:
        raise ValueError("Each fold must include at most one step-less test/loss event.")
    if test_loss_events:
        metrics["loss"] = test_loss_events[0].value
    return metrics


def _validate_final_metrics(
    records: Sequence[FoldTrackingRecord],
    final_metrics: Sequence[Mapping[str, float | int | str]],
) -> tuple[str, ...]:
    metric_keys = tuple(final_metrics[0])
    if not metric_keys:
        raise ValueError("Cross-validation folds must include final metrics.")
    if "f1_macro" not in metric_keys:
        raise ValueError("Cross-validation folds must include the f1_macro metric.")

    expected_keys = set(metric_keys)
    for record, metrics in zip(records, final_metrics):
        if not isinstance(record.result.fold, int) or isinstance(record.result.fold, bool):
            raise ValueError("Cross-validation fold identifiers must be integers.")
        if set(metrics) != expected_keys:
            raise ValueError("All folds must have the same final metric keys.")
        for key, value in metrics.items():
            if not _is_finite_number(value):
                raise ValueError(f"Final metric {key!r} must be finite and numeric.")
    return metric_keys


def _summarize_metric(values: Sequence[float]) -> MetricSummary:
    value_array, mean, standard_deviation, margin = _student_t_interval(values)
    return MetricSummary(
        mean=mean,
        ci95_lower=mean - margin,
        ci95_upper=mean + margin,
        std=standard_deviation,
        minimum=float(np.min(value_array)),
        maximum=float(np.max(value_array)),
        fold_count=len(value_array),
    )


def _summarize_loss_bands(records: Sequence[FoldTrackingRecord]) -> Mapping[str, Sequence[LossBand]]:
    fold_events = [_loss_events_by_key(record.metric_events) for record in records]
    reference_steps = {key: set(fold_events[0][key]) for key in _LOSS_KEYS}
    for events_by_key in fold_events:
        for key in _LOSS_KEYS:
            if set(events_by_key[key]) != reference_steps[key]:
                raise ValueError(f"Loss events for {key!r} must have matching steps across folds.")

    return {
        key: tuple(
            _loss_band(step, [events_by_key[key][step] for events_by_key in fold_events])
            for step in sorted(reference_steps[key])
        )
        for key in _LOSS_KEYS
    }


def _loss_events_by_key(metric_events: Sequence[LoggedMetric]) -> dict[str, dict[int, float]]:
    events_by_key: dict[str, dict[int, float]] = {key: {} for key in _LOSS_KEYS}
    for event in metric_events:
        if event.key not in events_by_key:
            continue
        if not isinstance(event.step, int) or isinstance(event.step, bool):
            raise ValueError(f"Loss event {event.key!r} must have an integer step.")
        if not _is_finite_number(event.value):
            raise ValueError(f"Loss event {event.key!r} must have a finite numeric value.")
        if event.step in events_by_key[event.key]:
            raise ValueError(f"Loss event {event.key!r} has duplicate step {event.step}.")
        events_by_key[event.key][event.step] = float(event.value)

    for key, events in events_by_key.items():
        if not events:
            raise ValueError(f"Loss events must include {key!r}.")
    return events_by_key


def _loss_band(step: int, values: Sequence[float]) -> LossBand:
    _, mean, _, margin = _student_t_interval(values)
    return LossBand(step=step, mean=mean, ci95_lower=mean - margin, ci95_upper=mean + margin)


def _student_t_interval(
    values: Sequence[float],
) -> tuple[np.ndarray, float, float, float]:
    value_array = np.asarray(values, dtype=float)
    mean = float(np.mean(value_array))
    standard_deviation = float(np.std(value_array, ddof=1))
    margin = float(
        t.ppf(0.975, df=len(value_array) - 1)
        * standard_deviation
        / sqrt(len(value_array))
    )
    return value_array, mean, standard_deviation, margin


def _diagnostic_roles(records: Sequence[FoldTrackingRecord]) -> Mapping[int, str]:
    worst = min(
        records,
        key=lambda record: (float(record.result.metrics["f1_macro"]), record.result.fold),
    )
    best = min(
        records,
        key=lambda record: (-float(record.result.metrics["f1_macro"]), record.result.fold),
    )
    if best.result.fold == worst.result.fold:
        return {best.result.fold: "best_and_worst"}
    return {worst.result.fold: "worst_fold", best.result.fold: "best_fold"}


def _is_finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value))


def compute_cv_summary(
    df_or_path: pd.DataFrame | str | Path,
) -> dict[str, dict[str, float | str]]:
    """Return the legacy mean, sample standard deviation, and display string."""
    frame = pd.read_csv(df_or_path) if isinstance(df_or_path, (str, Path)) else df_or_path.copy()
    numeric_columns = [column for column in frame.columns if column not in ["fold", "dataset"]]
    mean_series = frame[numeric_columns].mean()
    standard_deviation_series = frame[numeric_columns].std()
    return {
        column: {
            "mean": mean_series[column],
            "std": standard_deviation_series[column],
            "mean_std": f"{mean_series[column]:.4f} ± {standard_deviation_series[column]:.4f}",
        }
        for column in numeric_columns
    }
