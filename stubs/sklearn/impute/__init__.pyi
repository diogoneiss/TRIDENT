from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

class KNNImputer:
    n_features_in_: int
    def __init__(
        self,
        *,
        missing_values: Any = ...,
        n_neighbors: int = ...,
        weights: Literal["uniform", "distance"] = ...,
        metric: str = ...,
        copy: bool = ...,
        add_indicator: bool = ...,
        keep_empty_features: bool = ...,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: Any = ...) -> Self: ...
    def transform(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def fit_transform(self, X: ArrayLike, y: Any = ..., **fit_params: Any) -> NDArray[np.float64]: ...
