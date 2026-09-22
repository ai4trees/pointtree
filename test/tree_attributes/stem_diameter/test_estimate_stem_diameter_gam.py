"""Tests for pointtree.operations.estimate_stem_diameter_gam."""

import numpy as np
import pytest

from pointtree.tree_attributes.stem_diameter import estimate_stem_diameter_gam

from ...utils import generate_circle_points, generate_ellipse_points


class TestEstimateStemDiameterGam:
    """Tests for pointtree.operations.estimate_stem_diameter_gam."""

    @pytest.mark.parametrize("completeness", [1.0, 0.6], ids=["full_circle", "missing_part"])
    def test_circle(self, completeness: float):
        circles = np.array([[1.0, 1.0, 1.0]])
        points = generate_circle_points(circles, min_points=50, max_points=50)
        points = points[: round(len(points) * completeness)]

        diameter, polygon_vertices = estimate_stem_diameter_gam(
            points, circles[0, :2], max_radius_diff=0.3, random_generator=np.random.default_rng(seed=0)
        )

        assert circles[0, 2] * 2 == pytest.approx(diameter, abs=0.001)
        assert polygon_vertices.shape == (360, 2)
        np.testing.assert_array_almost_equal(circles[0, :2], polygon_vertices.mean(axis=0))

    def test_accepts_3d_points(self):
        circles = np.array([[1.0, 1.0, 1.0]])
        points = generate_circle_points(circles, min_points=50, max_points=50)
        points_3d = np.column_stack([points, np.full(len(points), fill_value=5.0)])

        diameter_2d, polygon_vertices_2d = estimate_stem_diameter_gam(
            points, circles[0, :2], 0.3, np.random.default_rng(seed=0)
        )
        diameter_3d, polygon_vertices_3d = estimate_stem_diameter_gam(
            points_3d, circles[0, :2], 0.3, np.random.default_rng(seed=0)
        )

        assert diameter_2d == pytest.approx(diameter_3d)
        np.testing.assert_allclose(polygon_vertices_2d, polygon_vertices_3d)

    def test_valid_ellipse(self):
        ellipses = np.array([[1.0, 1.0, 1.2, 0.9, 0.0]])
        points = generate_ellipse_points(ellipses, min_points=50, max_points=50)

        diameter, polygon_vertices = estimate_stem_diameter_gam(
            points, ellipses[0, :2], 0.4, np.random.default_rng(seed=0)
        )

        # the diameter of the circle with the same area as the fitted ellipse
        expected_diameter = 2 * np.sqrt(ellipses[0, 2] * ellipses[0, 3])

        assert diameter is not None
        assert expected_diameter == pytest.approx(diameter, abs=0.001)
        assert polygon_vertices.shape == (360, 2)
        assert (points.min(axis=0) < polygon_vertices.mean(axis=0)).all()
        assert (points.max(axis=0) > polygon_vertices.mean(axis=0)).all()

    def test_invalid_due_to_radius_diff_exceeding_threshold(self):
        # the fitted ellipse's radii vary more than a circle's, so a strict max_radius_diff rejects the fit
        ellipses = np.array([[1.0, 1.0, 1.2, 0.9, 0.0]])
        points = generate_ellipse_points(ellipses, min_points=50, max_points=50)

        diameter, polygon_vertices = estimate_stem_diameter_gam(
            points, ellipses[0, :2], 0.1, np.random.default_rng(seed=0)
        )

        assert diameter is None
        assert polygon_vertices.shape == (360, 2)

    def test_no_max_radius_diff(self):
        ellipses = np.array([[1.0, 1.0, 1.2, 0.9, 0.0]])
        points = generate_ellipse_points(ellipses, min_points=50, max_points=50)

        diameter_unrestricted, _ = estimate_stem_diameter_gam(
            points, ellipses[0, :2], None, np.random.default_rng(seed=0)
        )
        diameter_with_threshold, _ = estimate_stem_diameter_gam(
            points, ellipses[0, :2], 0.4, np.random.default_rng(seed=0)
        )

        assert diameter_unrestricted is not None
        assert diameter_unrestricted == pytest.approx(diameter_with_threshold)

    def test_invalid_due_to_sparse_points(self):
        circles = np.array([[0.0, 0.0, 1.0]])
        points = generate_circle_points(circles, min_points=50, max_points=50)[:5]

        diameter, polygon_vertices = estimate_stem_diameter_gam(
            points, circles[0, :2], None, np.random.default_rng(seed=0)
        )

        assert diameter is None
        assert polygon_vertices.shape == (360, 2)
