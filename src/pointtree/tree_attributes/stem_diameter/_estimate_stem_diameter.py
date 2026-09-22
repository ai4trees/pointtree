"""Estimation of the stem diameter at breast height from circles / ellipses fitted to multiple stem layers."""

__all__ = ["estimate_stem_diameter"]

from typing import List, Optional, Tuple

import numpy as np

from pointtree.operations._estimate_with_linear_model import estimate_with_linear_model
from pointtree.type_aliases import FloatArray, LongArray

from ._estimate_stem_diameter_gam import estimate_stem_diameter_gam


def estimate_stem_diameter(  # pylint: disable=too-many-locals
    layer_xy: FloatArray,
    batch_lengths: LongArray,
    centers: FloatArray,
    fallback_diameters: FloatArray,
    layer_heights: FloatArray,
    target_heights: FloatArray,
    gam_max_radius_diff: Optional[float],
    random_generator: np.random.Generator,
) -> Tuple[FloatArray, FloatArray, List[Optional[float]], List[Optional[FloatArray]]]:
    r"""
    Estimates the stem diameter at one or several given heights from circles or ellipses fitted to multiple
    horizontal stem layers. For each layer, the stem diameter is refined by fitting a generalized additive model
    (GAM) to the points of that layer, falling back to the corresponding entry of :code:`fallback_diameters` if the
    GAM fit is invalid or the layer contains no points. A linear model is then fitted to the resulting per-layer
    diameters to predict the stem diameter as a function of the height above the ground, and the predictions of this
    model for :code:`target_heights` are returned.

    Args:
        layer_xy: X- and y-coordinates of the points of all layers. Points belonging to the same layer must be
            stored consecutively, with the number of points belonging to each layer given by :code:`batch_lengths`.
        batch_lengths: Number of points belonging to each layer.
        centers: Center of the circle or ellipse that was fitted to each layer, used to normalize the points before
            fitting the GAM.
        fallback_diameters: Diameter of the circle or ellipse that was fitted to each layer, used as the estimated
            diameter for a layer if the GAM fit is invalid or the layer contains no points.
        layer_heights: Height above the ground of the midpoint of each layer.
        target_heights: Heights above the ground at which the stem diameter is to be estimated.
        gam_max_radius_diff: If the difference between the minimum and the maximum of the radii predicted by the GAM
            for a layer is greater than this value, the GAM fit for that layer is considered invalid and
            :code:`fallback_diameters` is used instead.
        random_generator: Random number generator used by :code:`estimate_stem_diameter_gam`.

    Returns:
        :Tuple of four elements:
            - Estimated stem diameters at :code:`target_heights`.
            - Diameter used for each layer (from the GAM if valid, otherwise the corresponding entry of
              :code:`fallback_diameters`).
            - Diameter predicted by the GAM for each layer, or :code:`None` for layers where the GAM fit is invalid
              or the layer contains no points.
            - Boundary polygon predicted by the GAM for each layer, or :code:`None` for layers that contain no
              points.

    Shape:
        - :code:`layer_xy`: :math:`(N, 2)`
        - :code:`batch_lengths`: :math:`(L)`
        - :code:`centers`: :math:`(L, 2)`
        - :code:`fallback_diameters`: :math:`(L)`
        - :code:`layer_heights`: :math:`(L)`
        - :code:`target_heights`: :math:`(T)`
        - Output: :math:`(T)`, :math:`(L)`, list of length :math:`L`, list of length :math:`L`

        | where
        |
        | :math:`N` = number of points
        | :math:`L` = number of layers
        | :math:`T` = number of target heights
    """

    num_layers = len(batch_lengths)
    layer_diameters = np.empty(num_layers, dtype=layer_xy.dtype)
    gam_diameters: List[Optional[float]] = []
    polygon_vertices_per_layer: List[Optional[FloatArray]] = []

    batch_starts = np.cumsum(np.concatenate((np.array([0], dtype=np.int64), batch_lengths)))[:-1]

    for layer_idx in range(num_layers):
        start_idx = batch_starts[layer_idx]
        end_idx = start_idx + batch_lengths[layer_idx]

        diameter_gam: Optional[float] = None
        polygon_vertices: Optional[FloatArray] = None
        if start_idx < end_idx:
            diameter_gam, polygon_vertices = estimate_stem_diameter_gam(
                layer_xy[start_idx:end_idx], centers[layer_idx], gam_max_radius_diff, random_generator
            )

        gam_diameters.append(diameter_gam)
        polygon_vertices_per_layer.append(polygon_vertices)
        layer_diameters[layer_idx] = diameter_gam if diameter_gam is not None else fallback_diameters[layer_idx]

    prediction, _ = estimate_with_linear_model(
        layer_heights, layer_diameters, np.asarray(target_heights, dtype=layer_xy.dtype)
    )

    return prediction, layer_diameters, gam_diameters, polygon_vertices_per_layer
