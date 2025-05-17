"""Tests for the distance_to_dtm method in pointtree.operations."""

import numpy as np
import pytest

from pointtree.operations import distance_to_dtm


class TestDistanceToDtm:
    """Tests for the distance_to_dtm method in pointtree.operations."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_distance_to_dtm(self, dtype: np.dtype):
        xyz = np.array([[0, 0, 2], [1, 1, 4], [3, 4, 5]], dtype=dtype)
        dtm_offset = np.array([-2, -2], dtype=dtype)
        dtm = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 2, 2], [0, 0, 2, 4]], dtype=dtype)
        dtm_resolution = 2

        dists = distance_to_dtm(xyz, dtm, dtm_offset, dtm_resolution)

        expected_dists = np.array([2, 3, 2], dtype=dtype)

        np.testing.assert_array_equal(expected_dists, dists)

    def test_allow_outside_points(self):
        coords = np.empty((10, 3), dtype=np.float64)
        coords[:, 0] = 5
        coords[:, 1:] = 2
        dtm = np.ones((3, 2), dtype=np.float64)
        dtm_offset = np.zeros((2, ), dtype=np.float64)
        dtm_resolution = 1

        dists = distance_to_dtm(coords, dtm, dtm_offset, dtm_resolution, allow_outside_points=True)

        expected_dists = np.ones(len(coords), dtype=np.float64)

        np.testing.assert_array_equal(expected_dists, dists)

    def test_too_small_dtm(self):
        xyz = np.zeros((10, 3), dtype=np.float64)
        xyz[:, 0] = np.arange(len(xyz))
        dtm = np.zeros((1, 1), dtype=np.float64)
        dtm_offset = np.zeros((2,), dtype=np.float64)
        dtm_resolution = 1

        with pytest.raises(ValueError):
            distance_to_dtm(xyz, dtm, dtm_offset, dtm_resolution, allow_outside_points=False)
