""" Tests for the normalize_height method in pointtree.operations. """

import numpy as np
import pytest

from pointtree.operations import normalize_height


class TestNormalizeHeight:
    """Tests for the normalize_height method in pointtree.operations."""

    @pytest.mark.parametrize("inplace", [True, False])
    def test_normalize_height(self, inplace: bool):
        coords = np.array([[0, 0, 2], [1, 1, 4], [3, 4, 5]], dtype=np.float64)
        original_coords = coords.copy()
        dtm_offset = np.array([-2, -2], dtype=np.float64)
        dtm = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 2, 2], [0, 0, 2, 4]])
        dtm_resolution = 2

        normalized_coords = normalize_height(coords, dtm, dtm_offset, dtm_resolution, inplace=inplace)

        expected_normalized_coords = np.array([[0, 0, 2], [1, 1, 3], [3, 4, 2]])

        np.testing.assert_array_equal(expected_normalized_coords, normalized_coords)

        if inplace:
            np.testing.assert_array_equal(coords, expected_normalized_coords)
        else:
            np.testing.assert_array_equal(original_coords, coords)

    def test_allow_outside_points(self):
        coords = np.zeros((10, 3), dtype=np.float64)
        coords[:, 2] = 2
        dtm = np.ones((1, 1), dtype=np.float64)
        dtm_offset = np.zeros((1, 1), dtype=np.float64)
        dtm_resolution = 1

        normalized_coords = normalize_height(coords, dtm, dtm_offset, dtm_resolution, allow_outside_points=True)

        expected_normalized_coords = coords.copy()
        expected_normalized_coords[:, 2] = 1

        np.testing.assert_array_equal(expected_normalized_coords, normalized_coords)

    def test_too_small_dtm(self):
        coords = np.zeros((10, 3), dtype=np.float64)
        coords[:, 0] = np.arange(len(coords))
        dtm = np.zeros((1, 1), dtype=np.float64)
        dtm_offset = np.zeros((1, 1), dtype=np.float64)
        dtm_resolution = 1

        with pytest.raises(ValueError):
            normalize_height(coords, dtm, dtm_offset, dtm_resolution, allow_outside_points=False)
