__all__ = ["PCA"]

from typing import Any, Literal

import numpy as np
import numpy.typing as npt


class PCA:
    feature_names_in_: npt.NDArray = ...
    n_features_in_: int = ...
    noise_variance_: float = ...
    n_samples_: int = ...
    n_components_: int = ...
    mean_: npt.NDArray = ...
    singular_values_: npt.NDArray = ...
    explained_variance_ratio_: npt.NDArray = ...
    explained_variance_: npt.NDArray = ...
    components_: npt.NDArray = ...

    def __init__(
        self,
        n_components: float | None | str | int = None,
        *,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto",
        tol: float = 0.0,
        iterated_power: Literal["auto"] | int = "auto",
        n_oversamples: int = 10,
        power_iteration_normalizer: Literal["auto", "QR", "LU", "none"] = "auto",
        random_state: np.random.RandomState | None | int = None,
    ) -> None: ...

    def fit(self, X: npt.NDArray, y: Any = None) -> "PCA": ...
    def fit_transform(self, X: npt.NDArray, y: Any = None) -> npt.NDArray: ...
    def score_samples(self, X: npt.NDArray) -> npt.NDArray: ...
    def score(self, X: npt.NDArray, y: Any = None) -> float: ...
