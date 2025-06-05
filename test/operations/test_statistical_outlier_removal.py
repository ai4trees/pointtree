"""Tests for pointtree.operations.statistical_outlier_removal"""

import numpy as np
from pointtree.operations import statistical_outlier_removal
import pytest


class TestStatisticalOutlierRemoval:
    """Tests for pointtree.operations.statistical_outlier_removal"""

    @pytest.mark.parametrize("k", [5, 10])
    def test_outlier_removal(self, k: int):
        xyz = np.zeros((10, 3), dtype=np.float64)

        # add a bit noise to all points
        xyz += np.clip(np.random.randn(len(xyz), 3), -0.01, 0.01)

        # outlier points
        xyz[8:, :] += 4

        expected_filtered_indices = np.arange(8, dtype=np.int64)

        filtered_xyz, filtered_indices = statistical_outlier_removal(xyz, k=k, std_multiplier=1.5)

        sorting_indices = np.argsort(filtered_indices)
        filtered_indices = filtered_indices[sorting_indices]
        filtered_xyz = filtered_xyz[sorting_indices]

        np.testing.assert_array_equal(expected_filtered_indices, filtered_indices)
        np.testing.assert_array_equal(xyz[expected_filtered_indices], filtered_xyz)

    def test_small_input(self):
        xyz = np.zeros((2, 3), dtype=np.float64)

        expected_filtered_indices = np.arange(len(xyz), dtype=np.int64)

        filtered_xyz, filtered_indices = statistical_outlier_removal(xyz, k=5, std_multiplier=1.5)

        sorting_indices = np.argsort(filtered_indices)
        filtered_indices = filtered_indices[sorting_indices]
        filtered_xyz = filtered_xyz[sorting_indices]

        np.testing.assert_array_equal(expected_filtered_indices, filtered_indices)
        np.testing.assert_array_equal(xyz[expected_filtered_indices], filtered_xyz)
