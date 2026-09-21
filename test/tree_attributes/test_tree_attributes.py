"""Tests for pointtree.tree_attributes."""

import numpy as np
import numpy.typing as npt
import pytest

from pointtree.tree_attributes import (
    crown_volume,
    crown_width,
    stem_diameter,
    stem_direction,
    tree_attributes,
    tree_height,
)

from ..utils import generate_circle_points


def generate_cylinder_points(
    radius: float, min_z: float, max_z: float, layer_spacing: float = 0.05, points_per_layer: int = 30
) -> npt.NDArray[np.float64]:
    """Generates points sampled around the outline of a vertical cylinder of the given radius."""

    layers = []
    z = min_z
    layer_idx = 0
    while z <= max_z:
        circle_points = generate_circle_points(
            np.array([[0.0, 0.0, radius]]), min_points=points_per_layer, max_points=points_per_layer, seed=layer_idx
        )
        layers.append(np.column_stack([circle_points, np.full(len(circle_points), fill_value=z)]))
        z += layer_spacing
        layer_idx += 1

    return np.concatenate(layers).astype(np.float64)


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

    def test_direction_is_oriented_upwards(self):
        # points are ordered from top to bottom, so the naive PCA direction could point downwards
        z = np.linspace(2.0, 0.0, 50)
        xyz = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])

        direction = stem_direction(xyz)

        assert direction[2] >= 0
        np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-6)

    def test_tilted_stem(self):
        t = np.linspace(0.0, 1.0, 100)
        xyz = np.column_stack([t, np.zeros_like(t), t])

        direction = stem_direction(xyz)
        expected_direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)

        np.testing.assert_allclose(direction, expected_direction, atol=1e-6)


class TestStemDiameter:
    """Tests for pointtree.tree_attributes.stem_diameter."""

    def test_too_few_points(self):
        stem_xyz = np.random.rand(5, 3).astype(np.float64)

        diameters = stem_diameter(stem_xyz, min_points=15)

        assert diameters.shape == (1,)
        assert np.isnan(diameters).all()

    def test_no_points_within_any_layer(self):
        # all points are below the default layer_start of 1.0 m, so no horizontal layer contains enough points
        stem_xyz = generate_cylinder_points(radius=0.15, min_z=0.0, max_z=0.5)

        diameters = stem_diameter(stem_xyz)

        assert np.isnan(diameters).all()

    def test_multiple_target_heights(self):
        # the cylinder has a constant radius, so the diameter estimate should be the same at every target height
        stem_xyz = generate_cylinder_points(radius=0.15, min_z=0.0, max_z=4.3)

        diameters = stem_diameter(stem_xyz, target_heights=np.array([1.3, 2.0, 3.0]), random_seed=42)

        assert diameters.shape == (3,)
        np.testing.assert_allclose(diameters, 0.3, atol=0.01)


class TestTreeAttributes:
    """Tests for pointtree.tree_attributes.tree_attributes."""

    def make_tree(self):
        crown_xyz = np.array([[-1.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, -1.0, 6.0], [0.0, 1.0, 7.0]], dtype=np.float64)
        stem_xyz = generate_cylinder_points(radius=0.15, min_z=0.0, max_z=4.3)
        tree_xyz = np.concatenate([crown_xyz, stem_xyz])
        classification = np.concatenate([np.full(len(crown_xyz), fill_value=1), np.full(len(stem_xyz), fill_value=0)])
        return tree_xyz, classification

    def test_computes_all_attributes_by_default(self):
        tree_xyz, classification = self.make_tree()

        attributes = tree_attributes(
            tree_xyz,
            classification=classification,
            stem_class_ids=[0],
            leaf_class_ids=[1],
        )

        assert set(attributes.keys()) == {
            "crown_volume",
            "crown_width",
            "tree_height",
            "stem_diameter",
            "stem_direction",
        }
        assert attributes["tree_height"] == pytest.approx(tree_xyz[:, 2].max() - tree_xyz[:, 2].min())
        assert attributes["crown_width"] == pytest.approx(2.0)
        assert set(attributes["stem_diameter"].keys()) == {1.3}
        assert attributes["stem_diameter"][1.3] == pytest.approx(0.3, abs=0.01)
        np.testing.assert_allclose(attributes["stem_direction"], [0.0, 0.0, 1.0], atol=1e-6)

    def test_stem_diameter_for_multiple_target_heights(self):
        tree_xyz, classification = self.make_tree()

        attributes = tree_attributes(
            tree_xyz,
            attributes=["stem_diameter"],
            classification=classification,
            stem_class_ids=[0],
            leaf_class_ids=[1],
            stem_diameter_target_heights=np.array([1.3, 2.0, 3.0]),
        )

        assert set(attributes["stem_diameter"].keys()) == {1.3, 2.0, 3.0}
        for diameter in attributes["stem_diameter"].values():
            assert diameter == pytest.approx(0.3, abs=0.01)

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
