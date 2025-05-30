"""Tests for the create_digital_terrain_model method in pointtree.operations."""

from typing import Optional

import numpy as np
import pytest

from pointtree.operations import create_digital_terrain_model


class TestCreateDigitalTerrainModel:  # pylint: disable=too-few-public-methods
    """Tests for the create_digital_terrain_model method in pointtree.operations."""

    @pytest.mark.parametrize("num_workers", [1, -1])
    @pytest.mark.parametrize("voxel_size", [None, 0.5])
    def test_create_digital_terrain_model(self, voxel_size: Optional[float], num_workers: int):
        terrain_coords = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 2], [1, 0, 0], [1, 1, 3]], dtype=float)

        dtm, dtm_grid_offset = create_digital_terrain_model(
            terrain_coords, grid_resolution=1, k=2, p=1, voxel_size=voxel_size, num_workers=num_workers
        )

        expected_dtm = np.array([[1, 0], [2, 3]])
        expected_dtm_grid_offset = np.array([0, 0])

        np.testing.assert_array_equal(expected_dtm_grid_offset, dtm_grid_offset)
        np.testing.assert_almost_equal(expected_dtm, dtm, decimal=5)
