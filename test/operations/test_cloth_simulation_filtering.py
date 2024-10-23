""" Tests for the cloth_simulation_filtering method in pointtree.operations. """

import numpy as np
import pytest

from pointtree.operations import cloth_simulation_filtering


class TestClothSimulationFiltering:  # pylint: disable=too-few-public-methods
    """Tests for the cloth_simulation_filtering method in pointtree.operations."""

    @pytest.mark.parametrize("rigidness", [1])
    @pytest.mark.parametrize(
        "classification_threshold, expected_classification",
        [(0.5, np.zeros(100 * 100 + 2, dtype=int)), (0.1, np.array([0] * 100 * 100 + [1] * 2))],
    )
    def test_cloth_simulation_filtering(
        self, rigidness: int, classification_threshold: float, expected_classification: np.ndarray
    ):
        x, y = np.meshgrid(np.arange(100, dtype=float), np.arange(100, dtype=float))
        terrain_coords = np.column_stack([x.flatten(), y.flatten(), np.zeros(100 * 100)])

        above_terrain_coords = np.array([[10, 10, 0.4], [40, 40, 0.2]], dtype=float)

        coords = np.row_stack([terrain_coords, above_terrain_coords])

        resolution = 2
        classification = cloth_simulation_filtering(
            coords, classification_threshold=classification_threshold, resolution=resolution, rigidness=rigidness
        )

        np.testing.assert_array_equal(expected_classification, classification)
