__all__ = ["distance_transform_edt", "gaussian_filter"]

from typing import Optional, Sequence, Tuple, Union

import numpy as np


def gaussian_filter(input: np.ndarray,
                    sigma: Union[float, Sequence[float]],
                    order: Union[int, Sequence[int]] = ...,
                    output: Optional[Union[np.ndarray, np.dtype]] = ...,
                    mode: Union[str, Sequence] = ...,
                    cval: float = ...,
                    truncate: float = ...,
                    *,
                    radius: Optional[Union[int, Sequence[int]]] = ...,
                    axes: Optional[Tuple[int, ...]] = ...) -> np.ndarray:
    ...


def distance_transform_edt(input: np.ndarray,
                           sampling: Optional[Union[float, Sequence[float]]] = ...,
                           return_distances: bool = ...,
                           return_indices: bool = ...,
                           distances: Optional[np.ndarray] = ...,
                           indices: Optional[np.ndarray] = ...) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    ...
