""" Testing whether 2D points are within an ellipse. """

__all__ = ["points_in_ellipse"]

import numpy as np
import numpy.typing as npt

from pointtree._operations_cpp import points_in_ellipse as points_in_ellipse_cpp  # type: ignore[import-not-found] # pylint: disable=import-error, no-name-in-module


def points_in_ellipse(xy: npt.NDArray[np.float64], ellipse: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    r"""
    Tests whether 2D points are within the boundaries of an ellipse.

    Args:
        xy: Coordinates of the points to test.
        ellipse: Parameters of the ellipse in the following order: X- and y-coordinates of its center, radius along the
            semi-major and along the semi-minor axis, and the counterclockwise angle of rotation from the x-axis to the
            semi-major axis of the ellipse.

    Returns:
        Boolean array that indicates for each point whether it lies within the ellipse.

    Raises:
        ValueError: If the shape of :code:`xy` or :code:`ellipse` is invalid.

    Shape:
        - :code:`xy`: :math:`(N, 2)`
        - :code:`ellipse`: :math:`(5)`
        - :code:`batch_indices_query_points`: :math:`(N')`
        - Output: :math:`(N)`

          | where
          |
          | :math:`N = \text{ number of points}`
    """

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be an array of 2D coordinates.")

    if ellipse.shape != (5,):
        raise ValueError("ellipse must contain five parameters.")

    return points_in_ellipse_cpp(xy, ellipse)
