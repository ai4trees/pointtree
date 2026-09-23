"""Tests for pointtree.tree_attributes."""

import numpy as np
import numpy.typing as npt
import pytest

from pointtree.tree_attributes import (
    crown_base_height,
    crown_volume,
    crown_width,
    stem_diameter,
    stem_direction,
    tree_attributes,
    tree_height,
    tree_position,
    under_branch_height,
)
from pointtree.tree_attributes.stem_diameter import StemDiameterAllometricModel

from ..utils import generate_circle_points, generate_ellipse_points


def generate_cylinder_points(
    radius: float,
    min_z: float,
    max_z: float,
    layer_spacing: float = 0.05,
    points_per_layer: int = 30,
    angular_fraction: float = 1.0,
) -> npt.NDArray[np.float64]:
    """
    Generates points sampled around the outline of a vertical cylinder of the given radius.

    Args:
        radius: Radius of the cylinder.
        min_z: Z-coordinate of the lowest layer of points.
        max_z: Z-coordinate up to which layers of points are generated.
        layer_spacing: Vertical spacing between consecutive layers of points.
        points_per_layer: Number of points sampled around the full outline of each layer, before
            :code:`angular_fraction` is applied.
        angular_fraction: Fraction of each layer's circular outline that is covered by points. For example, a value
            of 0.3 means that only the first 30 % of the angular range is covered by points.

    Returns:
        X-, y-, and z-coordinates of the generated points.
    """

    layers = []
    z = min_z
    layer_idx = 0
    while z <= max_z:
        circle_points = generate_circle_points(
            np.array([[0.0, 0.0, radius]]), min_points=points_per_layer, max_points=points_per_layer, seed=layer_idx
        )
        circle_points = circle_points[: round(len(circle_points) * angular_fraction)]
        layers.append(np.column_stack([circle_points, np.full(len(circle_points), fill_value=z)]))
        z += layer_spacing
        layer_idx += 1

    return np.concatenate(layers).astype(np.float64)


def generate_elliptical_stem_points(
    major_radius: float,
    minor_radius: float,
    min_z: float,
    max_z: float,
    layer_spacing: float = 0.05,
    points_per_layer: int = 200,
    variance: float = 0.03,
) -> npt.NDArray[np.float64]:
    """
    Generates points sampled around the outline of a vertical stem with a constant, elliptical cross-section.

    Args:
        major_radius: Radius along the semi-major axis of the elliptical cross-section.
        minor_radius: Radius along the semi-minor axis of the elliptical cross-section.
        min_z: Z-coordinate of the lowest layer of points.
        max_z: Z-coordinate up to which layers of points are generated.
        layer_spacing: Vertical spacing between consecutive layers of points.
        points_per_layer: Number of points sampled around the outline of each layer.
        variance: Variance of the distance of the sampled points to the ellipse outline.

    Returns:
        X-, y-, and z-coordinates of the generated points.
    """

    layers = []
    z = min_z
    layer_idx = 0
    while z <= max_z:
        ellipse_points = generate_ellipse_points(
            np.array([[0.0, 0.0, major_radius, minor_radius, 0.0]]),
            min_points=points_per_layer,
            max_points=points_per_layer,
            seed=layer_idx,
            variance=variance,
        )
        layers.append(np.column_stack([ellipse_points, np.full(len(ellipse_points), fill_value=z)]))
        z += layer_spacing
        layer_idx += 1

    return np.concatenate(layers).astype(np.float64)


class TestCrownBaseHeight:
    """Tests for pointtree.tree_attributes.crown_base_height."""

    def test_valid(self):
        crown_xyz = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 6.0]], dtype=np.float64)

        height = crown_base_height(crown_xyz, ground_height=0.0)

        assert height == pytest.approx(4.0)

    def test_clamped_to_zero(self):
        crown_xyz = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)

        height = crown_base_height(crown_xyz, ground_height=0.0)

        assert height == pytest.approx(0.0)

    def test_no_crown_points(self):
        assert np.isnan(crown_base_height(np.empty((0, 3), dtype=np.float64), ground_height=0.0))


class TestCrownVolume:
    """Tests for pointtree.tree_attributes.crown_volume."""

    def test_empty_crown(self):
        assert crown_volume(np.empty((0, 3), dtype=np.float64)) == 0.0

    def test_single_voxel_column(self):
        xyz = np.array([[0.1, 0.1, 0.0], [0.4, 0.4, 2.0]], dtype=np.float64)

        voxel_volume = 0.5 * 0.5

        assert crown_volume(xyz, voxel_size=0.5) == pytest.approx(voxel_volume * 2.0)

    def test_multiple_voxel_columns(self):
        xyz = np.array(
            [
                [0.1, 0.1, 0.0],
                [0.1, 0.1, 1.0],
                [10.1, 10.1, 0.0],
                [10.1, 10.1, 2.0],
            ],
            dtype=np.float64,
        )

        voxel_volume = 1.0

        assert crown_volume(xyz, voxel_size=1.0) == pytest.approx(voxel_volume * (1.0 + 2.0))


class TestCrownWidth:
    """Tests for pointtree.tree_attributes.crown_width."""

    def test_empty_crown(self):
        assert crown_width(np.empty((0, 3), dtype=np.float64)) == 0.0

    def test_width_along_x_axis(self):
        xyz = np.array([[-1.0, 0.0, 0.0], [2.0, 0.5, 5.0]], dtype=np.float64)

        assert crown_width(xyz) == pytest.approx(3.0)

    def test_width_along_y_axis(self):
        xyz = np.array([[0.0, -2.0, 0.0], [0.5, 3.0, 5.0]], dtype=np.float64)

        assert crown_width(xyz) == pytest.approx(5.0)


class TestStemDirection:
    """Tests for pointtree.tree_attributes.stem_direction."""

    def test_empty_stem(self):
        direction = stem_direction(np.empty((0, 3), dtype=np.float64))

        assert np.isnan(direction).all()

    def test_single_point(self):
        direction = stem_direction(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))

        assert np.isnan(direction).all()

    def test_vertical_stem(self):
        z = np.linspace(0.0, 2.0, 50)
        xyz = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])

        direction = stem_direction(xyz)

        np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-6)

    def test_tilted_stem(self):
        t = np.linspace(0.0, 1.0, 100)
        xyz = np.column_stack([t, np.zeros_like(t), t])

        direction = stem_direction(xyz)
        expected_direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)

        np.testing.assert_allclose(direction, expected_direction, atol=1e-6)

    def test_direction_is_oriented_upwards(self):
        # sklearn's PCA sign convention orients the component so that its largest-magnitude entry (here x) is
        # positive, which leaves z negative for this configuration before the manual sign correction is applied
        t = np.linspace(0.0, 1.0, 100)
        xyz = np.column_stack([2 * t, np.zeros_like(t), -t])

        direction = stem_direction(xyz)
        expected_direction = np.array([-2.0, 0.0, 1.0]) / np.sqrt(5)

        np.testing.assert_array_almost_equal(direction, expected_direction)


class TestStemDiameter:
    """Tests for pointtree.tree_attributes.stem_diameter."""

    def test_too_few_points(self):
        stem_xyz = np.random.rand(5, 3).astype(np.float64)

        diameters, completeness_indices, layer_diameter_std = stem_diameter(stem_xyz, min_points=15, std_num_layers=6)

        assert diameters.shape == (1,)
        assert np.isnan(diameters).all()
        assert completeness_indices.shape == (6,)
        assert np.isnan(completeness_indices).all()
        assert np.isnan(layer_diameter_std)

    def test_no_points_within_any_layer(self):
        # all points are below the default layer_start of 1.0 m, so no horizontal layer contains enough points
        stem_xyz = generate_cylinder_points(radius=0.15, min_z=0.0, max_z=0.5)

        diameters, completeness_indices, layer_diameter_std = stem_diameter(stem_xyz)

        assert np.isnan(diameters).all()
        assert np.isnan(completeness_indices).all()
        assert np.isnan(layer_diameter_std)

    def test_multiple_target_heights(self):
        # the cylinder has a constant radius, so the diameter estimate should be the same at every target height
        stem_xyz = generate_cylinder_points(radius=0.15, min_z=0.0, max_z=4.3, points_per_layer=100)

        diameters, completeness_indices, layer_diameter_std = stem_diameter(
            stem_xyz, target_heights=np.array([1.3, 2.0, 3.0]), std_num_layers=6, random_seed=42
        )

        assert diameters.shape == (3,)
        np.testing.assert_allclose(diameters, 0.3, atol=0.01)
        np.testing.assert_array_equal(completeness_indices, np.ones_like(completeness_indices))
        assert layer_diameter_std == pytest.approx(0.0, abs=1e-3)

    def test_layer_completeness_reflects_angular_coverage(self):
        kwargs = {
            "std_num_layers": 4,
            "num_layers": 4,
            "layer_start": 0.2,
            "layer_height": 0.3,
            "layer_overlap": 0.0,
            "min_completeness_idx": None,
            "random_seed": 42,
        }

        full_stem_xyz = generate_cylinder_points(radius=0.15, min_z=0.0, max_z=1.5, points_per_layer=200)
        _, full_completeness_indices, _ = stem_diameter(full_stem_xyz, **kwargs)

        angular_fraction = 0.3
        partial_stem_xyz = generate_cylinder_points(
            radius=0.15, min_z=0.0, max_z=1.5, points_per_layer=200, angular_fraction=angular_fraction
        )
        _, partial_completeness, _ = stem_diameter(partial_stem_xyz, **kwargs)

        np.testing.assert_array_equal(full_completeness_indices, np.ones_like(full_completeness_indices))
        np.testing.assert_array_almost_equal(
            partial_completeness, np.full_like(partial_completeness, fill_value=angular_fraction), decimal=1
        )

    def test_falls_back_to_ellipses_when_no_valid_circle_combination_exists(self):
        major_radius = 0.3
        minor_radius = 0.2
        stem_xyz = generate_elliptical_stem_points(
            major_radius=major_radius, minor_radius=minor_radius, min_z=0.2, max_z=1.7
        )
        kwargs = {
            "std_num_layers": 5,
            "num_layers": 5,
            "layer_start": 0.2,
            "layer_height": 0.3,
            "layer_overlap": 0.0,
            "max_std_diameter": 0.02,
        }

        # without the ellipse fallback, no valid combination of circles can be found
        diameters, completeness_indices, layer_diameter_std = stem_diameter(stem_xyz, fit_ellipses=False, **kwargs)

        assert np.isnan(diameters).all()
        assert np.isnan(completeness_indices).all()
        assert np.isnan(layer_diameter_std)

        # with the ellipse fallback enabled, a valid combination of ellipses is found instead
        diameters, completeness_indices, layer_diameter_std = stem_diameter(stem_xyz, fit_ellipses=True, **kwargs)

        # the expected diameter is the diameter of the circle with the same area as the elliptical cross-section
        assert diameters[0] == pytest.approx(2 * np.sqrt(major_radius * minor_radius), abs=0.05)
        # no circumferential completeness index is available when the ellipse fallback is used
        assert np.isnan(completeness_indices).all()
        np.testing.assert_array_almost_equal(layer_diameter_std, np.zeros_like(layer_diameter_std), decimal=3)


class TestTreeHeight:
    """Tests for pointtree.tree_attributes.tree_height."""

    def test_empty_tree(self):
        assert tree_height(np.empty((0, 3), dtype=np.float64)) == 0.0

    def test_without_ground_height(self):
        xyz = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 5.5]], dtype=np.float64)

        assert tree_height(xyz) == pytest.approx(4.0)

    def test_with_ground_height(self):
        xyz = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 5.5]], dtype=np.float64)

        assert tree_height(xyz, ground_height=1.0) == pytest.approx(4.5)


class TestTreePosition:
    """Tests for pointtree.tree_attributes.tree_position."""

    def test_empty_stem_and_tree(self):
        empty = np.empty((0, 3), dtype=np.float64)

        position = tree_position(empty, empty)

        assert np.isnan(position).all()

    def test_mean_of_stem_points(self):
        stem_xyz = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]], dtype=np.float64)
        tree_xyz = np.array([[10.0, 20.0, 0.0], [30.0, 40.0, 1.0]], dtype=np.float64)

        assert tree_position(stem_xyz, tree_xyz) == pytest.approx((2.0, 3.0))

    def test_falls_back_to_tree_points_when_stem_is_empty(self):
        empty_stem_xyz = np.empty((0, 3), dtype=np.float64)
        tree_xyz = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]], dtype=np.float64)

        assert tree_position(empty_stem_xyz, tree_xyz) == pytest.approx((2.0, 3.0))


class TestUnderBranchHeight:
    """Tests for pointtree.tree_attributes.under_branch_height."""

    def test_valid(self):
        branch_xyz = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.5]], dtype=np.float64)

        height = under_branch_height(branch_xyz, ground_height=0.0)

        assert height == pytest.approx(3.0)

    def test_clamped_to_zero(self):
        branch_xyz = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)

        height = under_branch_height(branch_xyz, ground_height=0.0)

        assert height == pytest.approx(0.0)

    def test_nan_when_no_branches(self):
        height = under_branch_height(np.empty((0, 3), dtype=np.float64), ground_height=0.0)

        assert np.isnan(height)


class TestTreeAttributes:
    """Tests for pointtree.tree_attributes.tree_attributes."""

    def make_tree(self, angular_fraction: float = 1.0):
        crown_xyz = np.array([[-1.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, -1.0, 6.0], [0.0, 1.0, 7.0]], dtype=np.float64)
        branch_xyz = np.array([[0.0, 0.0, 4.5], [0.0, 0.0, 4.8]], dtype=np.float64)
        stem_xyz = generate_cylinder_points(radius=0.15, min_z=0.0, max_z=4.3, angular_fraction=angular_fraction)
        tree_xyz = np.concatenate([crown_xyz, branch_xyz, stem_xyz])
        classification = np.concatenate(
            [
                np.full(len(crown_xyz), fill_value=1),
                np.full(len(branch_xyz), fill_value=2),
                np.full(len(stem_xyz), fill_value=0),
            ]
        )
        return tree_xyz, classification

    def test_computes_all_attributes_by_default(self):
        tree_xyz, classification = self.make_tree()

        attributes = tree_attributes(
            tree_xyz,
            classification=classification,
            stem_class_ids=[0],
            branch_class_ids=[2],
            leaf_class_ids=[1],
        )

        assert set(attributes.keys()) == {
            "crown_volume",
            "crown_width",
            "tree_height",
            "tree_position",
            "crown_base_height",
            "under_branch_height",
            "stem_diameter_1.3",
            "stem_diameter_layer_completeness",
            "stem_diameter_layer_std",
            "stem_direction",
            "tree_points",
            "stem_points",
            "branch_points",
            "crown_points",
        }
        assert attributes["tree_height"] == pytest.approx(tree_xyz[:, 2].max() - tree_xyz[:, 2].min())
        assert attributes["crown_width"] == pytest.approx(2.0)
        assert attributes["tree_position"] == pytest.approx((0.0, 0.0), abs=0.01)
        assert attributes["crown_base_height"] == pytest.approx(5.0)
        assert attributes["under_branch_height"] == pytest.approx(4.5)
        assert attributes["stem_diameter_1.3"] == pytest.approx(0.3, abs=0.01)
        assert attributes["stem_diameter_layer_completeness"].shape == (6,)
        assert not np.isnan(attributes["stem_diameter_layer_completeness"]).any()
        assert attributes["stem_diameter_layer_std"] == pytest.approx(0.0, abs=1e-3)
        np.testing.assert_allclose(attributes["stem_direction"], [0.0, 0.0, 1.0], atol=1e-6)

    def test_point_counts(self):
        tree_xyz, classification = self.make_tree()

        attributes = tree_attributes(
            tree_xyz,
            attributes=["tree_points", "stem_points", "branch_points", "crown_points"],
            classification=classification,
            stem_class_ids=[0],
            branch_class_ids=[2],
            leaf_class_ids=[1],
        )

        assert attributes == {
            "tree_points": len(tree_xyz),
            "stem_points": int((classification == 0).sum()),
            "branch_points": 2,
            "crown_points": 4,
        }

    def test_point_counts_without_classification(self):
        tree_xyz, _ = self.make_tree()

        attributes = tree_attributes(
            tree_xyz, attributes=["tree_points", "stem_points", "branch_points", "crown_points"]
        )

        assert attributes == {
            "tree_points": len(tree_xyz),
            "stem_points": len(tree_xyz),
            "branch_points": len(tree_xyz),
            "crown_points": len(tree_xyz),
        }

    def test_stem_diameter_for_multiple_target_heights(self):
        tree_xyz, classification = self.make_tree()

        attributes = tree_attributes(
            tree_xyz,
            attributes=["stem_diameter"],
            classification=classification,
            stem_class_ids=[0],
            leaf_class_ids=[1],
            attribute_kwargs={"stem_diameter": {"target_heights": np.array([1.3, 2.0, 3.0])}},
        )

        assert set(attributes.keys()) == {
            "stem_diameter_1.3",
            "stem_diameter_2.0",
            "stem_diameter_3.0",
            "stem_diameter_layer_completeness",
            "stem_diameter_layer_std",
        }
        for key in ["stem_diameter_1.3", "stem_diameter_2.0", "stem_diameter_3.0"]:
            assert attributes[key] == pytest.approx(0.3, abs=0.01)

    @pytest.mark.parametrize("use_allometric_model", (True, False))
    def test_stem_diameter_falls_back_to_allometric_model(self, use_allometric_model: bool):
        # an angular fraction below the default min_completeness_idx of 0.3 causes every fitted circle to be
        # rejected, so the circle / ellipse fitting approach cannot estimate the stem diameter
        tree_xyz, classification = self.make_tree(angular_fraction=0.2)
        # tree_height = 7.0 (crown top) - 0.0 (ground height, i.e., minimum z of the tree), crown_width = 2.0
        allometric_model = StemDiameterAllometricModel(scale=1.0, exponent=1.0, correction_factor=1.0)

        attributes = tree_attributes(
            tree_xyz,
            attributes=["stem_diameter"],
            classification=classification,
            stem_class_ids=[0],
            leaf_class_ids=[1],
            allometric_model=allometric_model if use_allometric_model else None,
            attribute_kwargs={"stem_diameter": {"target_heights": np.array([1.3, 2.0])}},
        )

        if use_allometric_model:
            assert attributes["stem_diameter_1.3"] == pytest.approx(14.0)
        else:
            assert np.isnan(attributes["stem_diameter_1.3"])

        assert np.isnan(attributes["stem_diameter_2.0"])

    def test_allometric_model_is_not_used_when_normal_estimation_succeeds(self):
        tree_xyz, classification = self.make_tree()
        # if used, this model's prediction (1.0 * (7.0 * 2.0) ** 1.0) would be far off the actual diameter of 0.3
        allometric_model = StemDiameterAllometricModel(scale=1.0, exponent=1.0, correction_factor=1.0)

        attributes = tree_attributes(
            tree_xyz,
            attributes=["stem_diameter"],
            classification=classification,
            stem_class_ids=[0],
            leaf_class_ids=[1],
            allometric_model=allometric_model,
        )

        assert attributes["stem_diameter_1.3"] == pytest.approx(0.3, abs=0.01)

    def test_attribute_kwargs_are_passed_to_the_respective_attribute_function(self):
        tree_xyz, classification = self.make_tree()

        attributes_default_voxel_size = tree_attributes(
            tree_xyz,
            attributes=["crown_volume"],
            classification=classification,
            stem_class_ids=[0],
            leaf_class_ids=[1],
        )
        attributes_custom_voxel_size = tree_attributes(
            tree_xyz,
            attributes=["crown_volume"],
            classification=classification,
            stem_class_ids=[0],
            leaf_class_ids=[1],
            attribute_kwargs={"crown_volume": {"voxel_size": 5.0}},
        )

        assert attributes_default_voxel_size["crown_volume"] != attributes_custom_voxel_size["crown_volume"]

    def test_computes_only_requested_attributes(self):
        tree_xyz, classification = self.make_tree()

        attributes = tree_attributes(
            tree_xyz,
            attributes=["tree_height"],
            classification=classification,
            stem_class_ids=[0],
            leaf_class_ids=[1],
        )

        assert set(attributes.keys()) == {"tree_height"}

    def test_without_classification_uses_all_points_for_every_attribute(self):
        tree_xyz, _ = self.make_tree()

        attributes_without_classification = tree_attributes(tree_xyz, attributes=["crown_width"])
        attributes_with_full_tree_as_crown = tree_attributes(
            tree_xyz,
            attributes=["crown_width"],
            classification=np.zeros(len(tree_xyz), dtype=np.int64),
            leaf_class_ids=[0],
        )

        assert attributes_without_classification["crown_width"] == pytest.approx(
            attributes_with_full_tree_as_crown["crown_width"]
        )
