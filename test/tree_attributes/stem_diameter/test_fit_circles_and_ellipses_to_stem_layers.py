"""Tests for pointtree.tree_attributes.stem_diameter.fit_circles_and_ellipses_to_stem_layers."""

import numpy as np
import pytest

from pointtree.tree_attributes.stem_diameter import fit_circles_and_ellipses_to_stem_layers

from ...utils import generate_circle_points, generate_ellipse_points


class TestFitCirclesAndEllipsesToStemLayers:
    """Tests for pointtree.tree_attributes.stem_diameter.fit_circles_and_ellipses_to_stem_layers."""

    @pytest.mark.parametrize("storage_format", ["C", "F"])
    @pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
    def test_fits_circles_to_multiple_layers(self, scalar_type: np.dtype, storage_format: str):
        circles = np.array([[0.0, 0.0, 0.3], [5.0, 5.0, 0.5]])
        points_layer_0 = generate_circle_points(circles[:1], min_points=80, max_points=80, seed=1)
        points_layer_1 = generate_circle_points(circles[1:], min_points=90, max_points=90, seed=2)
        layer_xy = np.concatenate([points_layer_0, points_layer_1]).astype(scalar_type).copy(order=storage_format)
        batch_lengths = np.array([len(points_layer_0), len(points_layer_1)], dtype=np.int64)

        layer_circles, layer_ellipses = fit_circles_and_ellipses_to_stem_layers(
            layer_xy, batch_lengths, min_completeness_idx=None, seed=0
        )

        assert layer_circles.shape == (2, 3)
        assert layer_circles.dtype == scalar_type
        decimal = 3 if scalar_type == np.float32 else 6
        np.testing.assert_array_almost_equal(layer_circles, circles.astype(scalar_type), decimal=decimal)

        # ellipse fitting is disabled by default
        assert layer_ellipses.shape == (2, 5)
        np.testing.assert_array_equal(layer_ellipses, -1)

    def test_empty_layer_is_skipped(self):
        circles = np.array([[0.0, 0.0, 0.3], [5.0, 5.0, 0.5]])
        points_layer_0 = generate_circle_points(circles[:1], min_points=80, max_points=80, seed=1)
        points_layer_2 = generate_circle_points(circles[1:], min_points=80, max_points=80, seed=2)
        layer_xy = np.concatenate([points_layer_0, points_layer_2]).astype(np.float64)
        # the middle layer has no points
        batch_lengths = np.array([len(points_layer_0), 0, len(points_layer_2)], dtype=np.int64)

        layer_circles, layer_ellipses = fit_circles_and_ellipses_to_stem_layers(
            layer_xy, batch_lengths, min_completeness_idx=None, seed=0
        )

        assert layer_circles.shape == (3, 3)
        np.testing.assert_array_equal(layer_circles[1], [-1, -1, -1])
        np.testing.assert_array_almost_equal(layer_circles[0], circles[0], decimal=3)
        np.testing.assert_array_almost_equal(layer_circles[2], circles[1], decimal=3)
        np.testing.assert_array_equal(layer_ellipses[1], [-1, -1, -1, -1, -1])

    def test_min_completeness_idx(self):
        # only 30 % of the circle's outline is covered by points
        circles = np.array([[0.0, 0.0, 0.3]])
        points = generate_circle_points(circles, min_points=200, max_points=200, seed=3)[:60]
        layer_xy = points.astype(np.float64)
        batch_lengths = np.array([len(points)], dtype=np.int64)

        layer_circles_low_threshold, _ = fit_circles_and_ellipses_to_stem_layers(
            layer_xy, batch_lengths, min_completeness_idx=0.3, seed=0
        )
        layer_circles_high_threshold, _ = fit_circles_and_ellipses_to_stem_layers(
            layer_xy, batch_lengths, min_completeness_idx=0.9, seed=0
        )

        np.testing.assert_array_almost_equal(layer_circles_low_threshold[0], circles[0], decimal=3)
        np.testing.assert_array_equal(layer_circles_high_threshold[0], [-1, -1, -1])

    def test_stem_diameter_bounds(self):
        circles = np.array([[0.0, 0.0, 0.05]])

        # the circle's diameter (0.1) is below the min_stem_diameter used in the restricted fit
        points = generate_circle_points(circles, min_points=80, max_points=80, seed=1)
        layer_xy = points.astype(np.float64)
        batch_lengths = np.array([len(points)], dtype=np.int64)

        layer_circles_default_bounds, _ = fit_circles_and_ellipses_to_stem_layers(
            layer_xy, batch_lengths, min_completeness_idx=None, seed=0
        )
        layer_circles_restricted_bounds, _ = fit_circles_and_ellipses_to_stem_layers(
            layer_xy, batch_lengths, min_completeness_idx=None, min_stem_diameter=0.2, max_stem_diameter=1.0, seed=0
        )

        np.testing.assert_array_almost_equal(layer_circles_default_bounds[0], circles[0], decimal=3)
        np.testing.assert_array_equal(layer_circles_restricted_bounds[0], [-1, -1, -1])

    def test_ellipse_fitting_filters_by_axis_ratio(self):
        # axis ratio 0.9 / 1.0 = 0.9 is above the default threshold and should be kept
        near_circular_ellipse = np.array([0.0, 0.0, 1.0, 0.9, 0.0])
        near_circular_ellipse_points = generate_ellipse_points(
            near_circular_ellipse.reshape((1, -1)), min_points=100, max_points=100, seed=1
        )
        # axis ratio 0.3 / 1.0 = 0.3 is below the default threshold and should be filtered out
        elongated_ellipse_points = generate_ellipse_points(
            np.array([[5.0, 5.0, 1.0, 0.3, 0.2]]), min_points=100, max_points=100, seed=2
        )
        layer_xy = np.concatenate([near_circular_ellipse_points, elongated_ellipse_points]).astype(np.float64)
        batch_lengths = np.array([len(near_circular_ellipse_points), len(elongated_ellipse_points)], dtype=np.int64)

        _, layer_ellipses = fit_circles_and_ellipses_to_stem_layers(
            layer_xy, batch_lengths, fit_ellipses=True, min_completeness_idx=None, seed=0
        )

        np.testing.assert_array_almost_equal(layer_ellipses[0], near_circular_ellipse, decimal=3)
        np.testing.assert_array_equal(layer_ellipses[1], [-1, -1, -1, -1, -1])
