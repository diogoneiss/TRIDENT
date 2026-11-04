"""C15's arms (docs/plans/2026-11-03-c15-mnar-prediction.md): what a classification fold sees at a gap.

``C15_ARM`` is ``off`` by default, and then nothing here runs. Every other value turns on the
readouts (per-row probabilities, unweighted log loss and the kept model, written per fold) and picks
how a gap is shown:

- ``A1``: the shipped classifier, each gap its column's ``[NULL]``;
- ``A2``: missForest's fill plus a gap indicator the column's value MLP reads, (fill, 1) at a gap and
  (value, 0) elsewhere; pre-training never hides a gap, so its holes are A1's;
- ``A3``: missForest's fill, no indicator;
- ``A4:<k>``: the k-th MICE draw (``IterativeImputer(sample_posterior=True)``, seed k), no indicator;
- ``A5:<column>``: A1 with that column's token blocked as a key in every layer (a null model that cannot read it).

The imputer is fitted on the fold's training rows' features, never the label, and fills every row of
the fold. Numerical tables only, as C15's are.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd

from .missforest import MissForestImputer
from .types import FoldSplit, PreparedDataset

GAP_PREFIX = "__gap__"
ARM_PATTERN = re.compile(r"^(off|A1|A2|A3|A4:\d+|A5:[\w.]+)$")


def has_indicator(arm: str) -> bool:
    return arm == "A2"


def blocked_position(arm: str, categorical: list[str], numerical: list[str]) -> int | None:
    """A5's blocked sequence position: 1 for [CLS] plus the column's token index; None for every other arm."""
    if not arm.startswith("A5:"):
        return None
    return 1 + (list(categorical) + list(numerical)).index(arm.split(":", 1)[1])


def fold_dataset(dataset: PreparedDataset, fold: FoldSplit, arm: str) -> PreparedDataset:
    """``dataset`` as this fold's arm shows it: A1 unchanged, the others filled from the training rows."""
    if arm in ("off", "A1") or arm.startswith("A5:"):
        return dataset
    if dataset.categorical_columns:
        raise ValueError(f"C15_ARM {arm} takes numerical tables only")
    columns = list(dataset.numerical_columns)
    features = dataset.frame[columns]
    train = features.iloc[np.asarray(fold.train_indices)]
    if arm.startswith("A4:"):
        from sklearn.experimental import enable_iterative_imputer  # type: ignore[import-untyped]  # noqa: F401
        from sklearn.impute import IterativeImputer  # type: ignore[attr-defined]

        imputer = IterativeImputer(sample_posterior=True, random_state=int(arm.split(":")[1]))
        imputer.fit(train.to_numpy(dtype=float))
        filled = pd.DataFrame(imputer.transform(features.to_numpy(dtype=float)), columns=columns, index=features.index)
    else:
        forest = MissForestImputer(columns, [])
        forest.fit(train)
        filled = forest.impute(features, np.zeros(features.shape, dtype=bool))
        filled.index = features.index
    frame = dataset.frame.copy()
    frame[columns] = filled[columns].to_numpy(dtype=float)
    if has_indicator(arm):
        gaps = features.isna()
        for column in columns:
            frame[GAP_PREFIX + column] = gaps[column].to_numpy(dtype=float)
    frame.attrs = dict(dataset.frame.attrs)
    return dataclasses.replace(dataset, frame=frame)
