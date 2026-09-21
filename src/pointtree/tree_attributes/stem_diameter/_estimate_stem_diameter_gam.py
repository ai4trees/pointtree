"""Diameter estimation for a horizontal stem layer using a generalized additive model (GAM)."""

__all__ = ["estimate_stem_diameter_gam"]

from typing import Optional, Tuple

import numpy as np
from pygam import LinearGAM, s

from pointtree.operations import polygon_area
from pointtree.type_aliases import FloatArray


def estimate_stem_diameter_gam(  # pylint: disable=too-many-locals
    points: FloatArray,
    center: FloatArray,
    max_radius_diff: Optional[float],
    random_generator: np.random.Generator,
) -> Tuple[Optional[float], FloatArray]:
    r"""
    Estimates the diameter of a tree stem at a certain height using a generalized additive model (GAM). It is
    assumed that a circle or an ellipse has already been fitted to the points of the tree stem in a layer around the
    respective height. To create the GAM, the points are converted into polar coordinates, using the center of the
    previously fitted circle or ellipse as the coordinate origin. The GAM is then fitted to predict the radius of
    the points based on the angles. The fitted GAM is then used to predict the stem radii in one-degree intervals.
    From these predictions, the stem's boundary polygon is constructed and the stem diameter is computed from the
    area of the boundary polygon. Assuming the boundary polygon is approximately circular, the stem diameter is
    calculated using the formula for a circle's diameter.

    .. math::

        d = 2 \cdot \sqrt{\frac{A_{polygon}}{\pi}}

    If any of the predicted radii is negative or the difference between the minimum and maximum of the predicted radii
    is greater than :code:`max_radius_diff`, the fitted GAM is considered invalid, and :code:`None` is returned for the
    stem diameter. In this case, the diameter of the previously fitted circle or ellipse can be used as a more robust
    estimate of the stem diameter.

    Args:
        points: Points belonging to the stem layer for which to estimate the diameter.
        center: Center of the circle or ellipse that has been fitted to the stem layer.
        max_radius_diff: If the difference between the minimum and the maximum of the predicted radii is greater
            than this value, the fitted GAM is considered invalid.
        random_generator: Random number generator used for the small random offset added to the point radii before
            fitting the GAM (to avoid perfect separation).

    Returns:
        :Tuple with two elements:
            - Estimated stem diameter. The estimated stem diameter may be :code:`None` if the fitted GAM is invalid.
            - Array containing the sorted vertices of the stem's boundary polygon predicted by the GAM as cartesian
              coordinates.

    Shape:
        - :code:`points`: :math:`(N, 2)` or :math:`(N, 3)`
        - :code:`center`: :math:`(2)`
        - Output: :math:`(360, 2)`

        | where
        |
        | :math:`N = number of points`
    """

    points_centered = points[:, :2] - center.reshape((-1, 2))

    # calculate polar coordinates
    polar_radius = np.linalg.norm(points_centered[:, :2], axis=-1)

    # add small random offset to avoid perfect separation
    polar_radius = polar_radius + random_generator.normal(0, 1e-8, len(points))

    polar_angle = np.arctan2(points_centered[:, 1], points_centered[:, 0])

    # fit GAM
    polar_xy = np.column_stack((polar_angle, polar_radius))
    gam = LinearGAM(s(0, basis="cp", edge_knots=[-np.pi, np.pi])).fit(polar_xy[:, 0], polar_xy[:, 1])
    del polar_xy

    # predict stem outline using fitted GAM
    polar_angles = np.asarray([-np.pi + 2 * np.pi * k / 360 for k in range(360)])
    polar_radii = gam.predict(polar_angles)

    cartesian_coords_x = polar_radii * np.cos(polar_angles)
    cartesian_coords_y = polar_radii * np.sin(polar_angles)

    cartesian_coords = np.column_stack((cartesian_coords_x, cartesian_coords_y))
    cartesian_coords = cartesian_coords + center.reshape((-1, 2))

    radius_diff = polar_radii.max() - polar_radii.min()

    if (max_radius_diff is not None and radius_diff > max_radius_diff) or (polar_radii < 0).any():
        return None, cartesian_coords

    stem_area = polygon_area(cartesian_coords_x, cartesian_coords_y)
    diameter_gam = 2 * np.sqrt(stem_area / np.pi)

    return diameter_gam, cartesian_coords
