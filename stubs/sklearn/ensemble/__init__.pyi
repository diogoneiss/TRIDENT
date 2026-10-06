from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

class HistGradientBoostingRegressor:
    n_iter_: int
    def __init__(
        self,
        *,
        loss: str = ...,
        learning_rate: float = ...,
        max_iter: int = ...,
        max_leaf_nodes: int | None = ...,
        max_depth: int | None = ...,
        min_samples_leaf: int = ...,
        l2_regularization: float = ...,
        max_features: float = ...,
        max_bins: int = ...,
        categorical_features: ArrayLike | Literal["from_dtype"] | None = ...,
        early_stopping: bool | Literal["auto"] = ...,
        validation_fraction: float | None = ...,
        n_iter_no_change: int = ...,
        tol: float = ...,
        verbose: int = ...,
        random_state: int | None = ...,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = ...) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...

class HistGradientBoostingClassifier:
    classes_: NDArray[Any]
    n_iter_: int
    def __init__(
        self,
        *,
        loss: str = ...,
        learning_rate: float = ...,
        max_iter: int = ...,
        max_leaf_nodes: int | None = ...,
        max_depth: int | None = ...,
        min_samples_leaf: int = ...,
        l2_regularization: float = ...,
        max_features: float = ...,
        max_bins: int = ...,
        categorical_features: ArrayLike | Literal["from_dtype"] | None = ...,
        early_stopping: bool | Literal["auto"] = ...,
        validation_fraction: float | None = ...,
        n_iter_no_change: int = ...,
        tol: float = ...,
        verbose: int = ...,
        random_state: int | None = ...,
        class_weight: dict[Any, float] | Literal["balanced"] | None = ...,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = ...) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[Any]: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]: ...

class RandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = ...,
        *,
        criterion: str = ...,
        max_depth: int | None = ...,
        min_samples_split: int | float = ...,
        min_samples_leaf: int | float = ...,
        max_features: int | float | Literal["sqrt", "log2"] | None = ...,
        bootstrap: bool = ...,
        n_jobs: int | None = ...,
        random_state: int | None = ...,
        verbose: int = ...,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = ...) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...

class RandomForestClassifier:
    classes_: NDArray[Any]
    def __init__(
        self,
        n_estimators: int = ...,
        *,
        criterion: str = ...,
        max_depth: int | None = ...,
        min_samples_split: int | float = ...,
        min_samples_leaf: int | float = ...,
        max_features: int | float | Literal["sqrt", "log2"] | None = ...,
        bootstrap: bool = ...,
        n_jobs: int | None = ...,
        random_state: int | None = ...,
        verbose: int = ...,
        class_weight: dict[Any, float] | Literal["balanced", "balanced_subsample"] | None = ...,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = ...) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[Any]: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]: ...
