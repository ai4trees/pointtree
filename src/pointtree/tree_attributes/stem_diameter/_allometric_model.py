"""Fitting of allometric models for stem diameter estimation."""

__all__ = ["StemDiameterAllometricModel"]

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt


@dataclass
class StemDiameterAllometricModel:
    r"""
    Allometric model that predicts stem diameter from tree height and crown width, following the approach of
    `Jucker, Tommaso, et al. "Allometric equations for integrating remote sensing imagery into forest monitoring \
    programmes." Global Change Biology 23.1 (2017): 177-190. <https://doi.org/10.1111/gcb.13388>`__.

    The model has the form:

    .. math::
        \hat{D} = CF \cdot a \cdot (H \cdot CD)^b

    where :math:`H` is the tree height and :math:`CD` is the crown width. The parameters :math:`a` and :math:`b`
    are learned by :code:`fit` and :math:`CF` is a correction factor that removes the bias introduced by
    back-transforming the log-log regression used to fit :math:`a` and :math:`b` to linear space (see :code:`fit`
    for details).

    Args:
        num_bins: Number of stem diameter bins of equal width in log-space that :code:`fit` uses to compute the
            mean stem diameter and mean :math:`H \cdot CD` values to which the model is fitted.
    """

    num_bins: int = 50
    """
    Number of stem diameter bins of equal width in log-space that :code:`fit` uses to compute the mean stem
    diameter and mean :math:`H \\cdot CD` values to which the model is fitted.
    """

    scale: Optional[float] = None
    """Scale parameter :math:`a` of the power-law model. Set by :code:`fit`."""

    exponent: Optional[float] = None
    """Exponent :math:`b` of the power-law model. Set by :code:`fit`."""

    correction_factor: Optional[float] = None
    """
    Multiplicative correction factor that removes the bias introduced by back-transforming the log-log
    regression to linear space (see :code:`fit`). Set by :code:`fit`.
    """

    log_residual_variance: Optional[float] = None
    """Residual variance of the log-log regression from which :code:`correction_factor` is derived. Set by
    :code:`fit`."""

    def fit(
        self, tree_heights: npt.NDArray, crown_widths: npt.NDArray, stem_diameters: npt.NDArray
    ) -> "StemDiameterAllometricModel":
        r"""
        Fits the model to the given tree height, crown width, and stem diameter measurements.

        The parameters :math:`a` and :math:`b` are obtained by fitting an ordinary least-squares regression of
        :math:`\ln D` on :math:`\ln (H \cdot CD)`. To reduce the influence of skewed distributions of stem diameters
        on the fitted regression, :code:`stem_diameters`, :code:`tree_heights`, and :code:`crown_widths` are not used
        as such but are first aggregated: the trees are grouped into :code:`self.num_bins` bins of equal width in
        log-space according to their stem diameter, and the regression is fitted to the mean stem diameter and the
        mean of :math:`H \cdot CD` within each non-empty bin. Since fitting the regression in log-space and then
        transforming the predictions back to linear space via exponentiation introduces a systematic bias,
        :code:`correction_factor` is set to :math:`\exp(\sigma^2 / 2)`, where :math:`\sigma^2` is the residual variance
        of the log-log regression (Baskerville correction).

        Args:
            tree_heights: Height of each tree.
            crown_widths: Crown width of each tree, in the same unit as :code:`tree_heights`.
            stem_diameters: Stem diameter of each tree. The unit of the fitted model's predictions matches the
                unit of this input.

        Returns:
            This instance, with :code:`scale`, :code:`exponent`, :code:`correction_factor`, and
            :code:`log_residual_variance` set to the fitted values.

        Raises:
            ValueError: If :code:`tree_heights`, :code:`crown_widths`, and :code:`stem_diameters` do not have the
                same length, if any of them is empty, if any of their values is not positive, if
                :code:`self.num_bins` is not a positive integer, or if fewer than three non-empty stem diameter
                bins remain to fit the regression.

        Shape:
            - :code:`tree_heights`: :math:`(N)`
            - :code:`crown_widths`: :math:`(N)`
            - :code:`stem_diameters`: :math:`(N)`

            | where
            |
            | :math:`N` = number of trees
        """

        tree_heights = np.asarray(tree_heights, dtype=np.float64)
        crown_widths = np.asarray(crown_widths, dtype=np.float64)
        stem_diameters = np.asarray(stem_diameters, dtype=np.float64)

        if not len(tree_heights) == len(crown_widths) == len(stem_diameters):
            raise ValueError("tree_heights, crown_widths, and stem_diameters must have the same length.")

        if len(stem_diameters) == 0:
            raise ValueError("tree_heights, crown_widths, and stem_diameters must not be empty.")

        if (tree_heights <= 0).any() or (crown_widths <= 0).any() or (stem_diameters <= 0).any():
            raise ValueError("tree_heights, crown_widths, and stem_diameters must be positive.")

        if self.num_bins < 1:
            raise ValueError("num_bins must be a positive integer.")

        x, y = self._bin_means_by_log_diameter(tree_heights, crown_widths, stem_diameters, self.num_bins)

        if len(x) < 3:
            raise ValueError(
                "At least three non-empty stem diameter bins are required to fit the allometric model. Consider "
                "decreasing num_bins or providing more data."
            )

        exponent, log_scale = np.polyfit(x, y, deg=1)

        log_residuals = y - (log_scale + exponent * x)
        log_residual_variance = float((log_residuals**2).sum() / (len(x) - 2))

        self.scale = float(np.exp(log_scale))
        self.exponent = float(exponent)
        self.log_residual_variance = log_residual_variance
        self.correction_factor = float(np.exp(log_residual_variance / 2))

        return self

    def predict(
        self, tree_heights: npt.NDArray, crown_widths: npt.NDArray, apply_correction_factor: bool = True
    ) -> npt.NDArray:
        r"""
        Predicts the stem diameter from tree height and crown width using the fitted model:

        .. math::
            \hat{D} = CF \cdot a \cdot (H \cdot CD)^b

        Args:
            tree_heights: Height of each tree, in the same unit that was used to fit the model.
            crown_widths: Crown width of each tree, in the same unit that was used to fit the model.
            apply_correction_factor: Whether to multiply the predictions by :code:`correction_factor` to correct
                for the bias introduced by back-transforming the log-log regression to linear space.

        Returns:
            Predicted stem diameter for each tree, in the same unit that was used to fit the model.

        Raises:
            RuntimeError: If the model has not been fitted yet (see :code:`fit`).

        Shape:
            - :code:`tree_heights`: :math:`(N)`
            - :code:`crown_widths`: :math:`(N)`
            - Output: :math:`(N)`

            | where
            |
            | :math:`N` = number of trees
        """

        if self.scale is None or self.exponent is None or self.correction_factor is None:
            raise RuntimeError("The model has not been fitted yet. Call `fit` before calling `predict`.")

        tree_heights = np.asarray(tree_heights, dtype=np.float64)
        crown_widths = np.asarray(crown_widths, dtype=np.float64)

        prediction = self.scale * np.power(tree_heights * crown_widths, self.exponent)

        if apply_correction_factor:
            prediction = prediction * self.correction_factor

        return prediction

    @staticmethod
    def _bin_means_by_log_diameter(
        tree_heights: npt.NDArray, crown_widths: npt.NDArray, stem_diameters: npt.NDArray, num_bins: int
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        r"""
        Groups trees into :code:`num_bins` bins of equal width in log-space according to their stem diameter and
        computes, for each non-empty bin, the log of the mean of height times crown width and the log of the mean
        stem diameter.

        Args:
            tree_heights: Height of each tree.
            crown_widths: Crown width of each tree.
            stem_diameters: Stem diameter of each tree.
            num_bins: Number of stem diameter bins of equal width in log-space.

        Returns:
            :Tuple of two elements:
                - Log of the mean of height times crown width for each non-empty bin.
                - Log of the mean stem diameter for each non-empty bin.

        Shape:
            - :code:`tree_heights`: :math:`(N)`
            - :code:`crown_widths`: :math:`(N)`
            - :code:`stem_diameters`: :math:`(N)`
            - Output: :math:`(M)`, :math:`(M)`

            | where
            |
            | :math:`N` = number of trees
            | :math:`M` = number of non-empty bins (:math:`M \leq` :code:`num_bins`)
        """

        log_diameter = np.log(stem_diameters)
        bin_edges = np.linspace(log_diameter.min(), log_diameter.max(), num_bins + 1)
        bin_indices = np.clip(np.searchsorted(bin_edges, log_diameter, side="right") - 1, 0, num_bins - 1)

        bin_counts = np.bincount(bin_indices, minlength=num_bins)
        bin_predictor_sums = np.bincount(bin_indices, weights=tree_heights * crown_widths, minlength=num_bins)
        bin_diameter_sums = np.bincount(bin_indices, weights=stem_diameters, minlength=num_bins)

        non_empty_bins = bin_counts > 0
        bin_log_predictor = np.log(bin_predictor_sums[non_empty_bins] / bin_counts[non_empty_bins])
        bin_log_diameter = np.log(bin_diameter_sums[non_empty_bins] / bin_counts[non_empty_bins])

        return bin_log_predictor, bin_log_diameter
