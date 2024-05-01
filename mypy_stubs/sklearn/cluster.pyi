__all__ = ["DBSCAN"]

from typing import Any, Callable, Dict, Literal, Optional, Union

import numpy as np


class DBSCAN:
    def __init__(self,
                 eps: float = ...,
                 *,
                 min_samples: int = ...,
                 metric: Union[str, Callable] = ...,
                 metric_params: Optional[Dict] = ...,
                 algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = ...,
                 leaf_size: int = ...,
                 p: Optional[float] = ...,
                 n_jobs: Optional[int] = ...):
        self.labels_: np.ndarray

    def fit(self, X: np.ndarray, y: Any = ..., sample_weight: Optional[np.ndarray] = ...) -> DBSCAN:
        ...
