"""Utilities for generating test point cloud with trees."""

__all__ = ["generate_tree_point_cloud"]

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from ._generate_points import generate_circle_points, generate_grid_points


def generate_tree_point_cloud(  # pylint: disable=too-many-locals
    scalar_type: np.dtype, storage_layout: str, generate_intensities: bool
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Generates a test point cloud with two trees.

    Args:
        scalar_type: Data type to be used for the generated data.
        storage_layout: Storage layout to be used for the generated data. Must be either :code:`"C"` (C style) or
            :code:`"F"` (Fortran style).
        generate_intensities: Whether intensity values should be generated.

    Returns: Tuple of five arrays:
        - Coordinates of the generated points
        - Intensity values of the generated points
        - Trunk positions of the generated trees
        - Trunk diameters of the generated trees
        - Heights of the generated trees
    """
    point_spacing = 0.1
    intensity = 5000
    ground_plane = generate_grid_points((100, 100), point_spacing)
    ground_plane = np.column_stack([ground_plane, np.zeros(len(ground_plane), dtype=np.float64)])
    intensities_ground_plane = np.zeros(len(ground_plane), dtype=scalar_type)

    points_per_layer = 200

    num_layers_trunk_1 = 30
    trunk_1: List[float] = []
    for layer in range(num_layers_trunk_1):
        layer_height = layer * point_spacing
        layer_points = generate_circle_points(
            np.array([[2.05, 2.05, 0.15]]),
            min_points=points_per_layer,
            max_points=points_per_layer,
            variance=0.01,
            seed=layer,
        )
        layer_points = np.column_stack(
            (layer_points, np.full(len(layer_points), fill_value=layer_height, dtype=np.float64))
        )
        trunk_1.extend(layer_points)
    intensities_trunk_1 = np.full(len(trunk_1), fill_value=intensity - 1, dtype=scalar_type)

    num_layers_trunk_2 = 35
    trunk_2: List[float] = []
    for layer in range(num_layers_trunk_2):
        layer_height = layer * point_spacing
        layer_points = generate_circle_points(
            np.array([[5.65, 5.65, 0.25]]),
            min_points=points_per_layer,
            max_points=points_per_layer,
            variance=0.01,
            seed=layer,
        )
        layer_points = np.column_stack(
            (layer_points, np.full(len(layer_points), fill_value=layer_height, dtype=np.float64))
        )
        trunk_2.extend(layer_points)
    intensities_trunk_2 = np.full(len(trunk_2), fill_value=intensity + 1, dtype=scalar_type)

    crown_1 = generate_grid_points((20, 20, 20), point_spacing)
    crown_1[:, :2] += 1.05
    crown_1[:, 2] += 3.1
    intensities_crown_1 = np.full(len(crown_1), fill_value=intensity - 1, dtype=scalar_type)
    crown_2 = generate_grid_points((30, 30, 30), point_spacing)
    crown_2[:, :2] += 4.15
    crown_2[:, 2] += 3.6
    intensities_crown_2 = np.full(len(crown_2), fill_value=intensity + 1, dtype=scalar_type)

    trunk_positions = np.array([[2.05, 2.05], [5.65, 5.65]], dtype=np.float64)
    trunk_diameters = np.array([0.3, 0.5], dtype=np.float64)
    tree_heights = np.array([5.0, 6.5], dtype=np.float64)

    xyz = np.concatenate([ground_plane, crown_1, np.array(trunk_1), crown_2, np.array(trunk_2)]).astype(
        scalar_type, order=storage_layout
    )
    if generate_intensities:
        intensities = np.concatenate(
            [
                intensities_ground_plane,
                intensities_crown_1,
                intensities_trunk_1,
                intensities_crown_2,
                intensities_trunk_2,
            ]
        )
    else:
        intensities = None

    return xyz, intensities, trunk_positions, trunk_diameters, tree_heights
