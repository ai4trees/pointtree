""" Testing whether 2D points are within an ellipse. """

__all__ = ["points_in_ellipse"]

import numpy as np
import numpy.typing as npt


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

    center_x, center_y, radius_major, radius_minor, theta = ellipse

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    a = (cos_theta * (xy[:, 0] - center_x) + sin_theta * (xy[:, 1] - center_y)) ** 2
    b = (sin_theta * (xy[:, 0] - center_x) - cos_theta * (xy[:, 1] - center_y)) ** 2
    ellipse_equation = (a / radius_major**2) + (b / radius_minor**2)

    return ellipse_equation <= 1
