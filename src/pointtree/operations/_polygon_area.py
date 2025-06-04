"""Computation of a polygon's area."""

__all__ = ["polygon_area"]

import numpy as np

from pointtree.type_aliases import FloatArray


def polygon_area(x: FloatArray, y: FloatArray) -> float:
    r"""
    Computes the area of a convex polygon using the
    `Shoelace formula <https://en.wikipedia.org/wiki/Shoelace_formula>`__. It is expected that the input vertices are
    already sorted so that adjacent vertices are neighbors in the input vertex array.

    Args:
        x: X-coordinates of the polygon vertices.
        y: Y-coordinates of the polygon vertices.

    Returns:
        Polygon area.

    Shape:
        - :code:`x`: :math:`(N)`
        - :code:`y`: :math:`(N)`

        | where
        |
        | :math:`n = \text{ number of points}`
    """

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
