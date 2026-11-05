"""A distributional score for imputed cells, read beside ``impute_score`` (imputation-protocol/51).

``impute_score`` asks how far each fill is from its truth. This asks whether a column's fills are
distributed as its truths are, which a fill shrunk to the centre fails while scoring well cell by
cell (ADR 0025 decision 4). Per column: the 1-D Wasserstein distance between fills and truths, in
scaled units, for numbers; the total variation between their category frequencies for categories.
Both are the offline panel's (``scripts/experiments/metric_panel.py``: ``wasserstein``, ``tv``).
``dist_score`` puts each kind as a ratio to the mean/mode fill's own distance and weights the two by
their cells, as ``impute_score`` does: the naive fill scores 1, a perfect one 0.

It is marginal, one column at a time: it does not see a row's joint distribution, and a fill drawn at
random from the right distribution scores well. So it is read beside ``impute_score``, never alone.
``score_cells`` is untouched (ADR 0025): only the model's cells are scored here, when a run asks.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .imputation_metrics import CATEGORICAL, NUMERICAL, _ratio


def naive_distances(cells: pd.DataFrame, baselines: Mapping[str, float | str]) -> tuple[float, float]:
    """The mean/mode fill's own distances, averaged over the columns of each kind: a constant fill is
    a point mass, so its Wasserstein distance is the truths' mean absolute deviation from it and its
    total variation the share of truths that are not the mode."""
    numerical = cells[cells["kind"] == NUMERICAL]
    categorical = cells[cells["kind"] == CATEGORICAL]
    numbers = [
        float(np.mean(np.abs(group["actual"].to_numpy(dtype=float) - float(baselines[str(column)]))))
        for column, group in numerical.groupby("column", sort=True)
    ]
    categories = [
        float(np.mean(group["actual"].astype(str).to_numpy() != str(baselines[str(column)])))
        for column, group in categorical.groupby("column", sort=True)
    ]
    return (float(np.mean(numbers)) if numbers else 0.0, float(np.mean(categories)) if categories else 0.0)


def distribution_scores(cells: pd.DataFrame, baselines: Mapping[str, float | str]) -> dict[str, float]:
    """``wd_num_z``, ``tv_cat`` and ``dist_score`` of scored cells carrying ``actual`` and ``imputed``."""
    numerical = cells[cells["kind"] == NUMERICAL]
    categorical = cells[cells["kind"] == CATEGORICAL]
    total = len(numerical) + len(categorical)
    if not total:
        return {}
    naive_wd, naive_tv = naive_distances(cells, baselines)
    metrics: dict[str, float] = {}
    score = 0.0
    if len(numerical):
        metrics["wd_num_z"] = float(
            np.mean(
                [
                    wasserstein_distance(group["imputed"].to_numpy(dtype=float), group["actual"].to_numpy(dtype=float))
                    for _, group in numerical.groupby("column", sort=True)
                ]
            )
        )
        score += (len(numerical) / total) * _ratio(metrics["wd_num_z"], naive_wd)
    if len(categorical):
        variations = []
        for _, group in categorical.groupby("column", sort=True):
            actual = group["actual"].astype(str).to_numpy()
            imputed = group["imputed"].astype(str).to_numpy()
            variations.append(
                0.5 * float(sum(abs(np.mean(imputed == label) - np.mean(actual == label)) for label in np.union1d(actual, imputed)))
            )
        metrics["tv_cat"] = float(np.mean(variations))
        score += (len(categorical) / total) * _ratio(metrics["tv_cat"], naive_tv)
    metrics["dist_score"] = score
    return metrics
