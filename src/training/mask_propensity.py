"""Where a training fold's own gaps fall, so the decode stage's masks can follow it (E65).

With ``DECODE_MASK_WEIGHTING propensity``, the decode stage hides a training cell with the uniform
rule's probability times that cell's weight from here, instead of uniformly. Under MCAR every
weight is about 1. Under MAR, where a column's gaps sit in rows that its other columns pick out,
the masks move towards those rows, so the decoder trains on cells like the ones it is asked to
fill. Nothing here sees a hidden cell's value: the weights come from which cells of the training
fold are missing.

- **Per feature column** with at least ``MIN_GAPS`` gaps and as many observed cells, a logistic
  regression predicts whether the cell is missing from the row's other columns:
  - a numerical column enters standardised on its observed cells, a gap at 0, plus its missing
    indicator;
  - a categorical column enters as one indicator per category among its ``MAX_LEVELS`` most
    frequent, plus its missing indicator.
- **A cell's weight** is its fitted propensity over the mean propensity of the column's observed
  cells. So the expected count of masked cells in each column stays the uniform rule's, before the
  clip at a probability of 1.
- **A column with too few gaps or observed cells** keeps weight 1, and so does every gap. Gaps are
  never masked anyway.

The fit is deterministic (lbfgs) and draws nothing from the global random streams, so the decode
stage's draw sequence is the uniform rule's.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MIN_GAPS = 20
MAX_LEVELS = 10
# sklearn's C. Chosen before any score, on the spread of the weights over observed cells of one
# training fold (seed 404, fold 1): at 0.1 the MAR variants of E64's biodeg and kr-vs-kp keep a
# median spread of 0.44 and 0.41, while their MCAR controls, whose ideal spread is 0, fall to
# 0.22 and 0.11 (at 1.0: 0.50 and 0.47 against 0.27 and 0.12).
REGULARISATION = 0.1
MAX_ITERATIONS = 500


def _blocks(frame: pd.DataFrame, categorical: Sequence[str]) -> list[np.ndarray]:
    """Each column's features for the other columns' models, in frame order."""
    blocks = []
    for column in frame.columns:
        values = frame[column]
        missing = values.isna().to_numpy(dtype=float)[:, None]
        if column in categorical:
            levels = values.dropna().astype(str).value_counts().index[:MAX_LEVELS]
            as_text = values.astype(str).to_numpy()
            indicators = np.column_stack([(as_text == level) & (missing[:, 0] == 0) for level in levels]).astype(float) if len(levels) else np.zeros((len(frame), 0))
            blocks.append(np.hstack([indicators, missing]))
        else:
            numbers = values.to_numpy(dtype=float)
            observed = ~np.isnan(numbers)
            spread = numbers[observed].std() if observed.any() else 0.0
            centred = (numbers - numbers[observed].mean()) / spread if spread > 0 else np.zeros_like(numbers)
            blocks.append(np.column_stack([np.where(observed, centred, 0.0), missing[:, 0]]))
    return blocks


def propensity_weights(frame: pd.DataFrame, categorical: Sequence[str]) -> np.ndarray:
    """(rows, columns) weights in ``frame``'s column order; 1 where a column has no model."""
    gaps = frame.isna().to_numpy()
    weights = np.ones(frame.shape, dtype=float)
    blocks = _blocks(frame, categorical)
    for j in range(frame.shape[1]):
        target = gaps[:, j]
        if target.sum() < MIN_GAPS or (~target).sum() < MIN_GAPS:
            continue
        features = np.hstack([block for k, block in enumerate(blocks) if k != j])
        model = LogisticRegression(C=REGULARISATION, max_iter=MAX_ITERATIONS)
        propensity = model.fit(features, target.astype(int)).predict_proba(features)[:, 1]
        weights[~target, j] = propensity[~target] / propensity[~target].mean()
    return weights
