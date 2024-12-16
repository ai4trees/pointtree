""" Tests for pointtree.operations.polygon_area. """

import numpy as np

from pointtree.operations import polygon_area


class TestPolygonArea:
    """Tests for pointtree.operations.polygon_area."""

    def test_triangle(self):
        x = np.array([-1, -1, 1], dtype=np.float64)
        y = np.array([-1, 1, 1], dtype=np.float64)

        expected_area = 2

        area = polygon_area(x, y)

        assert expected_area == area

    def test_square(self):
        x = np.array([-1, -1, 1, 1], dtype=np.float64)
        y = np.array([-1, 1, 1, -1], dtype=np.float64)

        expected_area = 4

        area = polygon_area(x, y)

        assert expected_area == area

    def test_zero_area(self):
        x = np.array([0, 0, 0], dtype=np.float64)
        y = np.array([0, 0, 0], dtype=np.float64)

        expected_area = 0

        area = polygon_area(x, y)

        assert expected_area == area
