"""Metric aggregation helpers."""

from pathlib import Path

import pandas as pd


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
