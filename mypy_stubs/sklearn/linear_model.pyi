__all__ = ["LinearRegression"]

from typing import Optional, Union

import numpy.typing as npt

class LinearRegression:
    def __init__(
        self, *, fit_intercept: bool = ..., copy_X: bool = ..., n_jobs: Optional[int] = ..., positive: bool = ...
    ):
        self.coef_: npt.NDArray
        self.rank_: int
        self.singular_: npt.NDArray
        self.intercept_: Union[float, npt.NDArray]
        self.n_features_in_: int
        self.feature_names_in_: npt.NDArray
    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike, sample_weight: Optional[npt.ArrayLike] = None
    ) -> LinearRegression: ...
    def predict(self, X: npt.ArrayLike) -> npt.NDArray: ...
