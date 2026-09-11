from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Indices = NDArray[np.int64]
_Folds = Iterator[tuple[_Indices, _Indices]]

class BaseCrossValidator(metaclass=ABCMeta):
    def split(
        self, X: ArrayLike, y: ArrayLike | None = ..., groups: ArrayLike | None = ...
    ) -> _Folds: ...
    @abstractmethod
    def get_n_splits(self, X: Any = ..., y: Any = ..., groups: Any = ...) -> int: ...

class KFold(BaseCrossValidator):
    def __init__(
        self, n_splits: int = ..., *, shuffle: bool = ..., random_state: int | None = ...
    ) -> None: ...
    def get_n_splits(self, X: Any = ..., y: Any = ..., groups: Any = ...) -> int: ...

class StratifiedKFold(KFold):
    # sklearn narrows `y` to required here; mirroring that is an LSP break it also has.
    def split(  # type: ignore[override]
        self, X: ArrayLike, y: ArrayLike, groups: ArrayLike | None = ...
    ) -> _Folds: ...

class BaseShuffleSplit(BaseCrossValidator):
    def __init__(
        self,
        n_splits: int = ...,
        *,
        test_size: float | int | None = ...,
        train_size: float | int | None = ...,
        random_state: int | None = ...,
    ) -> None: ...
    def get_n_splits(self, X: Any = ..., y: Any = ..., groups: Any = ...) -> int: ...

class ShuffleSplit(BaseShuffleSplit): ...

class StratifiedShuffleSplit(BaseShuffleSplit):
    def split(  # type: ignore[override]
        self, X: ArrayLike, y: ArrayLike, groups: ArrayLike | None = ...
    ) -> _Folds: ...

def train_test_split(
    *arrays: Any,
    test_size: float | int | None = ...,
    train_size: float | int | None = ...,
    random_state: int | None = ...,
    shuffle: bool = ...,
    stratify: ArrayLike | None = ...,
) -> list[Any]: ...
