"""Tests for pointtree.tree_attributes.stem_diameter.select_best_stem_layer_combination."""

import numpy as np
import pytest

from pointtree.tree_attributes.stem_diameter import select_best_stem_layer_combination


class TestSelectBestStemLayerCombination:
    """Tests for pointtree.tree_attributes.stem_diameter.select_best_stem_layer_combination."""

    def test_too_few_existing_layers(self):
        existing_layers = np.array([0, 1], dtype=np.int64)
        diameters = np.array([1.0, 1.0, -1.0])

        selected_combination, diameter_std = select_best_stem_layer_combination(
            existing_layers, diameters, 3, max_std_diameter=1.0
        )

        assert selected_combination is None
        assert np.isnan(diameter_std)

    def test_selects_combination_with_lowest_diameter_std(self):
        diameters = np.array([1.0, 1.0, 5.0, 1.0, 5.0])
        existing_layers = np.array([0, 1, 2, 3, 4], dtype=np.int64)

        selected_combination, diameter_std = select_best_stem_layer_combination(
            existing_layers, diameters, 3, max_std_diameter=10.0
        )

        assert selected_combination is not None
        assert selected_combination.dtype == np.int64
        np.testing.assert_array_equal(selected_combination, [0, 1, 3])
        assert diameter_std == pytest.approx(0.0)

    def test_no_combination_below_max_std_diameter(self):
        diameters = np.array([1.0, 2.0, 3.0, 4.0])
        existing_layers = np.array([0, 1, 2, 3], dtype=np.int64)

        selected_combination, diameter_std = select_best_stem_layer_combination(
            existing_layers, diameters, 2, max_std_diameter=0.01
        )

        assert selected_combination is None
        assert np.isnan(diameter_std)

    def test_filters_by_max_std_position(self):
        diameters = np.array([1.0, 1.0, 1.0, 1.2])
        positions = np.array([[0.0, 0.0], [100.0, 100.0], [0.0, 0.0], [0.0, 0.0]])
        existing_layers = np.array([0, 1, 2, 3], dtype=np.int64)

        combination, diameter_std = select_best_stem_layer_combination(
            existing_layers, diameters, 2, max_std_diameter=0.5, positions=positions, max_std_position=10.0
        )

        assert combination is not None
        np.testing.assert_array_equal(combination, [0, 2])
        assert diameter_std == pytest.approx(0.0)

    def test_no_combination_below_max_std_position(self):
        diameters = np.array([1.0, 1.0, 1.0])
        positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        existing_layers = np.array([0, 1, 2], dtype=np.int64)

        selected_combination, diameter_std = select_best_stem_layer_combination(
            existing_layers, diameters, 2, max_std_diameter=1.0, positions=positions, max_std_position=0.01
        )

        assert selected_combination is None
        assert np.isnan(diameter_std)

    def test_returns_std_of_selected_combination_not_the_lowest_possible(self):
        # the combination with the globally lowest std ([0, 2], std 0) is filtered out by max_std_position
        diameters = np.array([1.0, 1.2, 1.0])
        positions = np.array([[0.0, 0.0], [0.0, 0.0], [100.0, 100.0]])
        existing_layers = np.array([0, 1, 2], dtype=np.int64)

        combination, diameter_std = select_best_stem_layer_combination(
            existing_layers, diameters, 2, max_std_diameter=0.5, positions=positions, max_std_position=10.0
        )

        assert combination is not None
        np.testing.assert_array_equal(combination, [0, 1])
        assert diameter_std == pytest.approx(0.1)
