"""Tests for pointtree.operations.points_in_ellipse"""

import pytest
import numpy as np

from pointtree.operations import points_in_ellipse


class TestPointsInEllipse:
    """Tests for pointtree.operations.points_in_ellipse"""

    @pytest.mark.parametrize("storage_format", ["C", "F"])
    @pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
    def test_points_in_ellipse(self, storage_format: str, scalar_type: np.dtype):
        ellipse = np.array([1, 1, np.sqrt(18), np.sqrt(8), np.pi / 4], scalar_type, order=storage_format)
        xy = np.array([[1, 1], [3.9, 4], [4.1, 4]], dtype=scalar_type, order=storage_format)

        in_ellipse = points_in_ellipse(xy, ellipse)
        expected_in_ellipse = np.array([True, True, False], dtype=scalar_type)

        assert in_ellipse.flags.owndata is False

        np.testing.assert_array_equal(expected_in_ellipse, in_ellipse)

    def test_invalid_ellipse(self):
        with pytest.raises(ValueError):
            ellipse = np.array([1, 1], dtype=np.float64)
            xy = np.array([[1, 1], [3.9, 4], [4.1, 4]], dtype=np.float64)

            points_in_ellipse(xy, ellipse)

    def test_invalid_xy(self):
        with pytest.raises(ValueError):
            ellipse = np.array([1, 1, 1, 1, 1], dtype=np.float64)
            xy = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)

            points_in_ellipse(xy, ellipse)
