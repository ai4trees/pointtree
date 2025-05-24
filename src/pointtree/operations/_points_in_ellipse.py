"""Testing whether 2D points are within an ellipse."""

__all__ = ["points_in_ellipse"]

import numpy as np
import numpy.typing as npt

from pointtree._operations_cpp import points_in_ellipse as points_in_ellipse_cpp  # type: ignore[import-untyped] # pylint: disable=import-error, no-name-in-module


def points_in_ellipse(xy: npt.NDArray, ellipse: npt.NDArray) -> npt.NDArray[np.bool_]:
    r"""
    Tests whether 2D points are within the boundaries of an ellipse.

    If the input arrays have a row-major storage layout
    (`numpy's <https://numpy.org/doc/stable/dev/internals.html>`__ default), a copy of the input arrays is created. To
    pass them by reference, they must be in column-major format.

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
        - Output: :math:`(N)`

          | where
          |
          | :math:`N = \text{ number of points}`
    """

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be an array of 2D coordinates.")

    if ellipse.ndim != 1 or ellipse.shape[0] != 5:
        raise ValueError("ellipse must contain five parameters.")

    if not xy.flags.f_contiguous:
        xy = xy.copy(order="F")  # ensure that the input array is in column-major

    ellipse = ellipse.astype(xy.dtype)
    if not ellipse.flags.f_contiguous:
        ellipse = ellipse.copy(order="F")  # ensure that the input array is in column-major

    return points_in_ellipse_cpp(xy, ellipse)
