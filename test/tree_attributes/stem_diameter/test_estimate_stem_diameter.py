"""Tests for pointtree.tree_attributes.stem_diameter.estimate_stem_diameter."""

import numpy as np
import pytest

from pointtree.tree_attributes.stem_diameter import estimate_stem_diameter

from ...utils import generate_circle_points


class TestEstimateStemDiameter:
    """Tests for pointtree.tree_attributes.stem_diameter.estimate_stem_diameter."""

    def test_predicts_diameter_from_linearly_tapering_layers(self):
        layer_heights = np.array([1.0, 2.0, 3.0])
        true_diameters = 0.2 + 0.1 * layer_heights
        centers = np.zeros((3, 2))

        layers = [
            generate_circle_points(np.array([[0.0, 0.0, diameter / 2]]), min_points=80, max_points=80, seed=1)
            for diameter in true_diameters
        ]
        layer_xy = np.concatenate(layers).astype(np.float64)
        batch_lengths = np.array([len(layer) for layer in layers], dtype=np.int64)

        target_heights = np.array([1.0, 1.5, 3.0])
        predicted_diameters, layer_diameters, gam_diameters, polygon_vertices_per_layer = estimate_stem_diameter(
            layer_xy,
            batch_lengths,
            centers,
            true_diameters,
            layer_heights,
            target_heights,
            0.3,
            np.random.default_rng(seed=0),
        )

        assert predicted_diameters.shape == (3,)
        np.testing.assert_array_almost_equal(0.2 + 0.1 * target_heights, predicted_diameters, decimal=2)
        np.testing.assert_array_almost_equal(true_diameters, layer_diameters, decimal=2)
        assert all(diameter is not None for diameter in gam_diameters)
        assert all(vertices.shape == (360, 2) for vertices in polygon_vertices_per_layer)

    def test_falls_back_to_fallback_diameter_when_gam_is_invalid(self):
        layer_heights = np.array([1.0, 2.0])
        centers = np.zeros((2, 2))

        valid_layer = generate_circle_points(np.array([[0.0, 0.0, 0.3]]), min_points=80, max_points=80, seed=1)
        # only 5 points concentrated in a small arc make the GAM extrapolate an implausible outline
        invalid_layer = generate_circle_points(np.array([[0.0, 0.0, 1.0]]), min_points=50, max_points=50, seed=0)[:5]

        layer_xy = np.concatenate([valid_layer, invalid_layer]).astype(np.float64)
        batch_lengths = np.array([len(valid_layer), len(invalid_layer)], dtype=np.int64)
        fallback_diameters = np.array([0.3, 0.77])

        predicted_diameters, layer_diameters, gam_diameters, polygon_vertices_per_layer = estimate_stem_diameter(
            layer_xy,
            batch_lengths,
            centers,
            fallback_diameters,
            layer_heights,
            np.array([1.5]),
            None,
            np.random.default_rng(seed=0),
        )

        assert gam_diameters[0] is not None
        assert gam_diameters[1] is None
        assert layer_diameters[1] == pytest.approx(fallback_diameters[1])
        # the boundary polygon is still returned even though the diameter estimate is invalid
        assert polygon_vertices_per_layer[1] is not None
        assert polygon_vertices_per_layer[1].shape == (360, 2)
        assert predicted_diameters[0] == pytest.approx((layer_diameters[0] + layer_diameters[1]) / 2)

    def test_uses_fallback_diameter_for_empty_layer(self):
        layer_heights = np.array([1.0, 2.0])
        centers = np.zeros((2, 2))

        valid_layer = generate_circle_points(np.array([[0.0, 0.0, 0.3]]), min_points=80, max_points=80, seed=1)

        layer_xy = valid_layer.astype(np.float64)
        batch_lengths = np.array([len(valid_layer), 0], dtype=np.int64)
        fallback_diameters = np.array([0.3, 0.77])

        predicted_diameters, layer_diameters, gam_diameters, polygon_vertices_per_layer = estimate_stem_diameter(
            layer_xy,
            batch_lengths,
            centers,
            fallback_diameters,
            layer_heights,
            np.array([1.5]),
            0.3,
            np.random.default_rng(seed=0),
        )

        assert gam_diameters[1] is None
        assert layer_diameters[1] == pytest.approx(fallback_diameters[1])
        assert polygon_vertices_per_layer[1] is None
        assert predicted_diameters[0] == pytest.approx((layer_diameters[0] + layer_diameters[1]) / 2)
