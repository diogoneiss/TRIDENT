from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

class BayesianRidge:
    def __init__(
        self,
        *,
        max_iter: int = ...,
        tol: float = ...,
        alpha_1: float = ...,
        alpha_2: float = ...,
        lambda_1: float = ...,
        lambda_2: float = ...,
        compute_score: bool = ...,
        fit_intercept: bool = ...,
        copy_X: bool = ...,
        verbose: bool = ...,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = ...) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...

class LogisticRegression:
    classes_: NDArray[Any]
    def __init__(
        self,
        *,
        penalty: str | None = ...,
        C: float = ...,
        fit_intercept: bool = ...,
        solver: str = ...,
        max_iter: int = ...,
        tol: float = ...,
        random_state: int | None = ...,
        n_jobs: int | None = ...,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = ...) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[Any]: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]: ...
