"""Metric aggregation helpers."""

import re
from math import isfinite, sqrt
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t

from src.training.types import (
    CLASSIFICATION,
    CrossValidationSummary,
    FoldKey,
    FoldTrackingRecord,
    LoggedMetric,
    LossBand,
    MetricSummary,
    TaskSpec,
)


# The per-epoch loss series a fold must log are task-specific: see
# ``TaskSpec.loss_keys`` (the shared pre-training pair plus the task's second stage).

# Step-less wall-clock events the runner logs once per fold. They are kept out
# of ``FoldResult.metrics`` so the deterministic ``metrics.csv`` and the
# regression fixture never see a non-deterministic number, and they are
# summarized separately from the test metrics so the parent shows them under
# ``cv/time/`` rather than ``cv/test/``.
_TIMING_PREFIX = "time/"
TIMING_METRIC_KEYS = (
    f"{_TIMING_PREFIX}pretrain_seconds",
    f"{_TIMING_PREFIX}finetune_seconds",
    f"{_TIMING_PREFIX}decode_seconds",
    f"{_TIMING_PREFIX}total_seconds",
)


def stage_timing_metrics(pretraining_seconds: float, finetuning_seconds: float) -> dict[str, float]:
    """Build the per-fold timing events from the two measured stage durations."""
    return {
        f"{_TIMING_PREFIX}pretrain_seconds": float(pretraining_seconds),
        f"{_TIMING_PREFIX}finetune_seconds": float(finetuning_seconds),
        f"{_TIMING_PREFIX}total_seconds": float(pretraining_seconds) + float(finetuning_seconds),
    }


def summarize_cross_validation(
    records: Sequence[FoldTrackingRecord], task: TaskSpec = CLASSIFICATION
) -> CrossValidationSummary:
    """Aggregate completed CV folds into comparable final and loss statistics.

    ``task`` names the fold-ranking metric the folds must carry and the direction in
    which a fold is best. It defaults to classification so every existing caller keeps
    today's behaviour.
    """
    if len(records) < 2:
        raise ValueError("Cross-validation summary requires at least two folds.")

    final_metrics = tuple(final_metrics_for_tracking(record) for record in records)
    metric_keys = _validate_final_metrics(records, final_metrics, task)
    metrics = {
        key: _summarize_metric([float(metrics[key]) for metrics in final_metrics])
        for key in metric_keys
    }
    best_baselines = _best_baselines(metrics)
    metrics.update(_best_baseline_copies(metrics, best_baselines))
    loss_bands = _summarize_loss_bands(records, task)
    return CrossValidationSummary(
        metrics=metrics,
        loss_bands=loss_bands,
        diagnostic_roles=_diagnostic_roles(records, task),
        timings=_summarize_timings(records),
        best_baselines=best_baselines,
    )


# The name the best baseline's copied statistics are logged under (ADR 0007).
BEST_BASELINE = "best"
_BASELINE_SCORE = re.compile(r"(.+)/baseline/([^/]+)/impute_score")


def _best_baselines(metrics: Mapping[str, MetricSummary]) -> dict[str, str]:
    """Per population, the baseline whose mean ``impute_score`` over the folds is lowest.

    Chosen once per run, on the cross-validated mean, so every fold's numbers come from
    the same baseline; a tie goes to the name that sorts first.
    """
    scores: dict[str, list[tuple[float, str]]] = {}
    for key, statistics in metrics.items():
        match = _BASELINE_SCORE.fullmatch(key)
        if match is None or match.group(2) == BEST_BASELINE:
            continue
        scores.setdefault(match.group(1), []).append((float(statistics.mean), match.group(2)))
    return {population: min(entries)[1] for population, entries in sorted(scores.items())}


def _best_baseline_copies(
    metrics: Mapping[str, MetricSummary], best_baselines: Mapping[str, str]
) -> dict[str, MetricSummary]:
    """Every metric of each population's best baseline again, under ``baseline/best``.

    The chosen baseline's own numbers, never the best of each metric taken separately,
    so ``best`` reads as one imputer the model can be set beside.
    """
    copies: dict[str, MetricSummary] = {}
    for population, name in best_baselines.items():
        prefix = f"{population}/baseline/{name}/"
        for key, statistics in metrics.items():
            metric = key[len(prefix):]
            if key.startswith(prefix) and "/" not in metric:
                copies[f"{population}/baseline/{BEST_BASELINE}/{metric}"] = statistics
    return copies


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


def fold_timings_for_tracking(record: FoldTrackingRecord) -> dict[str, float]:
    """Return a fold's step-less stage timings keyed without the ``time/`` prefix."""
    timings: dict[str, float] = {}
    for key in TIMING_METRIC_KEYS:
        events = [
            event
            for event in record.metric_events
            if event.key == key and event.step is None
        ]
        if len(events) > 1:
            raise ValueError(f"Each fold must include at most one step-less {key!r} event.")
        if not events:
            continue
        value = events[0].value
        if not _is_finite_number(value) or float(value) < 0:
            raise ValueError(f"Timing {key!r} must be a finite, non-negative number.")
        timings[key[len(_TIMING_PREFIX):]] = float(value)
    return timings


def _summarize_timings(records: Sequence[FoldTrackingRecord]) -> Mapping[str, MetricSummary]:
    fold_timings = [fold_timings_for_tracking(record) for record in records]
    expected_keys = set(fold_timings[0])
    for timings in fold_timings[1:]:
        if set(timings) != expected_keys:
            raise ValueError("All folds must have the same timing keys.")
    return {
        key: _summarize_metric([timings[key] for timings in fold_timings])
        for key in fold_timings[0]
    }


def _validate_final_metrics(
    records: Sequence[FoldTrackingRecord],
    final_metrics: Sequence[Mapping[str, float | int | str]],
    task: TaskSpec,
) -> tuple[str, ...]:
    metric_keys = tuple(final_metrics[0])
    if not metric_keys:
        raise ValueError("Cross-validation folds must include final metrics.")
    if task.ranking_metric not in metric_keys:
        raise ValueError(
            f"Cross-validation folds must include the {task.ranking_metric} metric."
        )

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


def _summarize_loss_bands(
    records: Sequence[FoldTrackingRecord], task: TaskSpec
) -> Mapping[str, Sequence[LossBand]]:
    loss_keys = task.loss_keys
    fold_events = [_loss_events_by_key(record.metric_events, loss_keys) for record in records]
    reference_steps = {key: set(fold_events[0][key]) for key in loss_keys}
    for events_by_key in fold_events:
        for key in loss_keys:
            if set(events_by_key[key]) != reference_steps[key]:
                raise ValueError(f"Loss events for {key!r} must have matching steps across folds.")

    return {
        key: tuple(
            _loss_band(step, [events_by_key[key][step] for events_by_key in fold_events])
            for step in sorted(reference_steps[key])
        )
        for key in loss_keys
    }


def _loss_events_by_key(
    metric_events: Sequence[LoggedMetric], loss_keys: Sequence[str]
) -> dict[str, dict[int, float]]:
    events_by_key: dict[str, dict[int, float]] = {key: {} for key in loss_keys}
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


def _diagnostic_roles(
    records: Sequence[FoldTrackingRecord], task: TaskSpec
) -> Mapping[FoldKey, str]:
    # ``sign`` orients the fold-ranking metric so that a smaller key is always worse;
    # ties still resolve to the lowest fold number for both roles.
    sign = 1.0 if task.direction == "maximize" else -1.0
    worst = min(
        records,
        key=lambda record: (sign * float(record.result.metrics[task.ranking_metric]), record.result.fold),
    )
    best = min(
        records,
        key=lambda record: (-sign * float(record.result.metrics[task.ranking_metric]), record.result.fold),
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
