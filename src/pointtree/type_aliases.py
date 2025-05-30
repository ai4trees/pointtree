"""Type aliases."""

__all__ = ["BoolArray", "FloatArray", "LongArray"]

from typing import Union

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[Union[np.float32, np.float64]]
LongArray = npt.NDArray[np.int64]
