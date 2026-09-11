from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

def compute_class_weight(
    class_weight: Mapping[int, float] | str | None,
    *,
    classes: NDArray[np.int64],
    y: ArrayLike,
    sample_weight: ArrayLike | None = ...,
) -> NDArray[np.float64]: ...
