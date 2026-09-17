from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Average = Literal["micro", "macro", "samples", "weighted", "binary"]
_ZeroDivision = Literal["warn"] | float

def accuracy_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    normalize: bool = ...,
    sample_weight: ArrayLike | None = ...,
) -> float: ...
def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    sample_weight: ArrayLike | None = ...,
    normalize: Literal["true", "pred", "all"] | None = ...,
) -> NDArray[Any]: ...
@overload
def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: _Average = ...,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> float: ...
@overload
def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: None,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> NDArray[np.float64]: ...
@overload
def precision_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: _Average = ...,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> float: ...
@overload
def precision_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: None,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> NDArray[np.float64]: ...
@overload
def recall_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: _Average = ...,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> float: ...
@overload
def recall_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: ArrayLike | None = ...,
    pos_label: Any = ...,
    average: None,
    sample_weight: ArrayLike | None = ...,
    zero_division: _ZeroDivision = ...,
) -> NDArray[np.float64]: ...
