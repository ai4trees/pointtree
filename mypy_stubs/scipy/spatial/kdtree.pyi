__all__ = ["KDTree"]

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

class KDTree:
    def __init__(
        self,
        data: ArrayLike,
        leafsize: int = ...,
        compact_nodes: bool = ...,
        copy_data: bool = ...,
        balanced_tree: bool = ...,
        boxsize: Optional[Union[ArrayLike, float]] = ...,
    ): ...

    def query(
        self,
        x: ArrayLike,
        k: Optional[int] = ...,
        eps: float = ...,
        p: int = ...,
        distance_upper_bound: float = ...,
        workers: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    def query_ball_point(
        self,
        x: ArrayLike,
        r: float,
        p: float = ...,
        eps: int = ...,
        workers: int = ...,
        return_sorted: bool = ...,
        return_length: bool = ...,
    ): ...
