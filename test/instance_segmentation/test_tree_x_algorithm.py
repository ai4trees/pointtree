""" Tests for pointtree.instance_segmentation.TreeXAlgorithm. """

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import shutil

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from pointtree.instance_segmentation import TreeXAlgorithm
from pointtree._tree_x_algorithm_cpp import (  # type: ignore[import-untyped] # pylint: disable=import-error, no-name-in-module
    collect_inputs_trunk_layers_preliminary_fitting as collect_inputs_trunk_layers_preliminary_fitting_cpp,
    collect_inputs_trunk_layers_exact_fitting as collect_inputs_exact_fitting_cpp,
)

from test.utils import (  # pylint: disable=wrong-import-order
    generate_circle_points,
    generate_ellipse_points,
    generate_grid_points,
)


class TestTreeXAlgorithm:
    """Tests for pointtree.instance_segmentation.TreeXAlgorithm."""

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/instance_segmentation/TestTreeXAlgorithm"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    def generate_layer_circles_and_ellipses(  # pylint: disable=too-many-locals
        self, add_noise_points: bool, variance: float
    ) -> Tuple[
        Dict[str, Any],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Generates inputs and expected values for testing the methods
        :code:`TreeXAlgorithm.fit_preliminary_circles_or_ellipses_to_trunks` and
        :code:`TreeXAlgorithm.fit_exact_circles_and_ellipses_to_trunks`.

        Args:
            add_noise_points: Whether randomly placed noise points not sampled from a circle / ellipse should be added
                to the generated  input points.
            variance: Variance of the distance of the generated points to the circle / ellipse outlines.

        Returns:
            Tuple of six values. The first is a dictionary containing the keyword arguments to be passed to the
            :code:`TreeXAlgorithm` class to produce the expected results returned by this method. The second is an array
            containing the input point coordinates to be passed to the
            :code:`TreeXAlgorithm.fit_preliminary_circles_or_ellipses_to_trunks` method. The third is an array
            containing IDs indicating which points belong to which trunk cluster. The fourth is an array containing the
            unique cluster IDs. The fifth contains the true parameters of the circles / ellipses from which the input
            points were sampled. The sixth contains the heights of the horizontal layers at which points were generated.
        """

        trunk_search_min_z = 1.0
        layer_height = 0.1
        layer_overlap = 0.02
        num_trees = 2
        num_layers = 5
        min_points = 50
        skip_layers = [2]
        generate_valid_ellipses = [0]
        generate_invalid_ellipses = [4]

        settings = {
            "trunk_search_min_z": trunk_search_min_z,
            "trunk_search_circle_fitting_num_layers": num_layers,
            "trunk_search_circle_fitting_layer_height": layer_height + layer_overlap,
            "trunk_search_circle_fitting_layer_overlap": layer_overlap,
            "trunk_search_circle_fitting_min_points": min_points,
            "trunk_search_circle_fitting_min_trunk_diameter": 0.02,
            "trunk_search_circle_fitting_max_trunk_diameter": 1,
            "trunk_search_circle_fitting_min_completeness_idx": 0.6,
        }

        circle_or_ellipse_points: List[float] = []
        cluster_labels: List[npt.NDArray[np.int64]] = []
        expected_circles_or_ellipses = np.full((num_trees, num_layers, 5), fill_value=-1, dtype=np.float64)
        expected_layer_heigths = np.empty(num_layers, dtype=np.float64)

        for tree in range(num_trees):
            offset = tree * 2.0
            for layer_idx in range(num_layers):
                current_height = trunk_search_min_z + layer_height * layer_idx + layer_height / 2 + layer_overlap / 2
                expected_layer_heigths[layer_idx] = current_height
                if layer_idx not in skip_layers:
                    if layer_idx in generate_valid_ellipses or layer_idx in generate_invalid_ellipses:
                        if layer_idx in generate_invalid_ellipses:
                            ellipses = np.array([[0, 0, 0.8, 0.2, np.pi / 3]], dtype=np.float64)
                        else:
                            ellipses = np.array([[0, 0, 0.6, 0.4, np.pi / 3]], dtype=np.float64)
                        points_2d = generate_ellipse_points(
                            ellipses,
                            min_points=10 * min_points,
                            max_points=20 * min_points,
                            seed=layer_idx,
                            add_noise_points=False,
                            variance=variance,
                        )
                        expected_circles_or_ellipses[tree, layer_idx] = ellipses[0]
                    else:
                        circles = np.array([[0, 0, 0.5 - 0.05 * layer_idx]], dtype=np.float64)
                        points_2d = generate_circle_points(
                            circles,
                            min_points=10 * min_points,
                            max_points=20 * min_points,
                            seed=layer_idx,
                            add_noise_points=add_noise_points,
                            variance=variance,
                        )
                        expected_circles_or_ellipses[tree, layer_idx, :3] = circles[0]
                    expected_circles_or_ellipses[tree, layer_idx, :2] += offset

                points_3d = np.empty(
                    (len(points_2d), 3), dtype=np.float64  # pylint: disable=possibly-used-before-assignment
                )
                points_3d[:, :2] = points_2d + offset
                points_3d[:, 2] = current_height
                if layer_idx in skip_layers:
                    points_3d = points_3d[: int(min_points / 2)]
                cluster_labels.extend(np.full(len(points_3d), fill_value=tree, dtype=np.int64))
                circle_or_ellipse_points.extend(points_3d)

        trunk_layer_xyz = np.array(circle_or_ellipse_points)
        cluster_labels_np = np.array(cluster_labels)
        unique_cluster_labels = np.unique(cluster_labels)

        return (
            settings,
            trunk_layer_xyz,
            cluster_labels_np,
            unique_cluster_labels,
            expected_circles_or_ellipses,
            expected_layer_heigths,
        )

    @pytest.mark.parametrize("num_workers", [1, -1])
    @pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
    def test_collect_inputs_preliminary_fitting_cpp(
        self, num_workers: int, scalar_type: np.dtype
    ):  # pylint: disable=too-many-locals
        trunk_layer_xyz = np.array(
            [
                [0, 0, 1],
                [0, 0, 1.1],
                [0, 0, 1.7],
                [0, 0, 2.2],
                [1, 0, 1.1],
                [1, 0, 1.2],
                [1, 0, 2.2],
                [1, 0, 2.3],
                [1, 0, 2.4],
                [0, 0, 1.55],
            ],
            dtype=scalar_type,
            order="F",
        )
        cluster_labels = np.array([0, 0, 0, 0, 2, 2, 2, 2, 2, 0], dtype=np.int64)
        unique_cluster_labels = np.unique(cluster_labels)

        num_layers = 3
        trunk_search_min_z = 1.0
        trunk_search_circle_fitting_layer_height = 0.6
        trunk_search_circle_fitting_layer_overlap = 0.1
        trunk_search_circle_fitting_min_points = 2

        expected_indices = [[0, 1, 9], [2, 9], [], [4, 5], [], [6, 7, 8]]

        trunk_layer_xy, batch_lengths = collect_inputs_trunk_layers_preliminary_fitting_cpp(
            trunk_layer_xyz,
            cluster_labels,
            unique_cluster_labels,
            trunk_search_min_z,
            num_layers,
            trunk_search_circle_fitting_layer_height,
            trunk_search_circle_fitting_layer_overlap,
            trunk_search_circle_fitting_min_points,
            num_workers,
        )

        assert len(batch_lengths) == len(unique_cluster_labels) * num_layers
        assert trunk_layer_xy.dtype == scalar_type

        start_idx = 0
        for i, expected_idx in enumerate(expected_indices):
            end_idx = start_idx + batch_lengths[i]
            xy_batch_item = trunk_layer_xy[start_idx:end_idx]
            expected_xy_batch_item = trunk_layer_xyz[expected_idx, :2]

            sorting_indices = np.lexsort((xy_batch_item[:, 1], xy_batch_item[:, 0]))
            xy_batch_item = xy_batch_item[sorting_indices]

            sorting_indices = np.lexsort((expected_xy_batch_item[:, 1], expected_xy_batch_item[:, 0]))
            expected_xy_batch_item = expected_xy_batch_item[sorting_indices]

            np.testing.assert_array_equal(expected_xy_batch_item, xy_batch_item)
            start_idx = end_idx

    def test_collect_inputs_preliminary_fitting_cpp_invalid_inputs(self):
        trunk_layer_xyz = np.zeros((10, 3), dtype=np.float64, order="F")
        cluster_labels = np.zeros((9), dtype=np.int64)
        unique_cluster_labels = np.array([0], dtype=np.int64)

        num_layers = 3
        trunk_search_min_z = 1.0
        trunk_search_circle_fitting_layer_height = 0.6
        trunk_search_circle_fitting_layer_overlap = 0.1
        trunk_search_circle_fitting_min_points = 2
        num_workers = 1

        with pytest.raises(ValueError):
            collect_inputs_trunk_layers_preliminary_fitting_cpp(
                trunk_layer_xyz,
                cluster_labels,
                unique_cluster_labels,
                trunk_search_min_z,
                num_layers,
                trunk_search_circle_fitting_layer_height,
                trunk_search_circle_fitting_layer_overlap,
                trunk_search_circle_fitting_min_points,
                num_workers,
            )

    @pytest.mark.parametrize("num_workers", [1, -1])
    @pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
    def test_collect_batch_indices_exact_fitting_cpp(
        self, num_workers: int, scalar_type: np.dtype
    ):  # pylint: disable=too-many-locals
        trunk_layer_xyz = np.array(
            [
                [0, 0.62, 1],
                [0.6, 0, 1.1],
                [-0.59, 0, 1.2],
                [0, 0, 1.1],
                [0, 0.61, 1.3],
                [0, 0.52, 1.3],
                [4 + 0.42 / np.sqrt(2), 4 + 0.42 / np.sqrt(2), 2.0],
                [4 + 0.4 / np.sqrt(2), 4 + 0.4 / np.sqrt(2), 2.0],
                [4 + 0.4 / np.sqrt(2), 4 + 0.41 / np.sqrt(2), 2.0],
                [0, 0.51, 2.2],
                [0, 0.52, 2.3],
            ],
            dtype=scalar_type,
            order="F",
        )

        preliminary_layer_circles_or_ellipses = (
            np.array(
                [
                    [[0, 0, 0.6, -1, -1], [-1, -1, -1, -1, -1], [4, 4, 0.4, 0.2, np.pi / 4]],
                    [
                        [-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [0, 0.3, 0.22, -1, -1],
                    ],
                ],
                dtype=scalar_type,
            )
            .reshape((-1, 5))
            .copy(order="F")
        )

        num_layers = 3
        trunk_search_min_z = 1.0
        trunk_search_circle_fitting_layer_height = 0.6
        trunk_search_circle_fitting_layer_overlap = 0.1
        trunk_search_circle_fitting_min_points = 2
        trunk_search_circle_fitting_switch_buffer_threshold = 0.7
        trunk_search_circle_fitting_small_buffer_width = 0.015
        trunk_search_circle_fitting_large_buffer_width = 0.05

        expected_indices = [[0, 1, 2, 4], [], [7, 8], [], [], [9, 10]]
        expected_layer_heights = np.array([1.3, 1.8, 2.3], dtype=scalar_type)

        trunk_layer_xy, batch_lengths, layer_heights = collect_inputs_exact_fitting_cpp(
            trunk_layer_xyz,
            preliminary_layer_circles_or_ellipses,
            trunk_search_min_z,
            num_layers,
            trunk_search_circle_fitting_layer_height,
            trunk_search_circle_fitting_layer_overlap,
            trunk_search_circle_fitting_switch_buffer_threshold,
            trunk_search_circle_fitting_small_buffer_width,
            trunk_search_circle_fitting_large_buffer_width,
            trunk_search_circle_fitting_min_points,
            num_workers,
        )

        assert len(batch_lengths) == len(expected_indices)
        assert trunk_layer_xy.dtype == scalar_type
        assert layer_heights.dtype == scalar_type

        start_idx = 0
        for i, expected_idx in enumerate(expected_indices):
            end_idx = start_idx + batch_lengths[i]
            xy_batch_item = trunk_layer_xy[start_idx:end_idx]
            expected_xy_batch_item = trunk_layer_xyz[expected_idx, :2]

            sorting_indices = np.lexsort((xy_batch_item[:, 1], xy_batch_item[:, 0]))
            xy_batch_item = xy_batch_item[sorting_indices]

            sorting_indices = np.lexsort((expected_xy_batch_item[:, 1], expected_xy_batch_item[:, 0]))
            expected_xy_batch_item = expected_xy_batch_item[sorting_indices]

            np.testing.assert_array_equal(expected_xy_batch_item, xy_batch_item)
            start_idx = end_idx

        np.testing.assert_array_equal(expected_layer_heights, layer_heights)

    @pytest.mark.parametrize("circle_fitting_method", ["ransac", "m-estimator"])
    @pytest.mark.parametrize("variance, add_noise_points", [(0.0, False), (0.01, True)])
    @pytest.mark.parametrize("create_visualization, use_pathlib", [(False, False), (True, False), (True, True)])
    @pytest.mark.parametrize("storage_layout", ["C", "F"])
    @pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
    def test_fit_circles_or_ellipses_to_trunks(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-positional-arguments
        self,
        circle_fitting_method: str,
        variance: float,
        add_noise_points: bool,
        create_visualization: bool,
        use_pathlib: bool,
        storage_layout: str,
        scalar_type: np.dtype,
        cache_dir,
    ):
        (
            settings,
            trunk_layer_xyz,
            cluster_labels,
            unique_cluster_labels,
            expected_circles_or_ellipses,
            expected_layer_heigths,
        ) = self.generate_layer_circles_and_ellipses(add_noise_points, variance)
        expected_circles_or_ellipses = expected_circles_or_ellipses.astype(scalar_type)
        expected_layer_heigths = expected_layer_heigths.astype(scalar_type)
        trunk_layer_xyz = trunk_layer_xyz.astype(scalar_type).copy(order=storage_layout)

        num_trees = len(expected_circles_or_ellipses)
        num_layers = expected_circles_or_ellipses.shape[1]

        visualization_folder = None
        point_cloud_id = None
        if create_visualization:
            point_cloud_id = "test"
            visualization_folder = cache_dir
            if use_pathlib:
                visualization_folder = Path(visualization_folder)

        trunk_search_ellipse_filter_threshold = 0.6

        algorithm = TreeXAlgorithm(
            visualization_folder=visualization_folder,
            trunk_search_circle_fitting_method=circle_fitting_method,
            trunk_search_ellipse_filter_threshold=trunk_search_ellipse_filter_threshold,
            **settings,
        )

        preliminary_circles_or_ellipses = algorithm.fit_preliminary_circles_or_ellipses_to_trunks(
            trunk_layer_xyz, cluster_labels, unique_cluster_labels, point_cloud_id=point_cloud_id
        )
        assert preliminary_circles_or_ellipses.dtype == scalar_type

        decimal = 1 if variance > 0 else 4

        np.testing.assert_almost_equal(expected_circles_or_ellipses, preliminary_circles_or_ellipses, decimal=decimal)

        if create_visualization:
            if not use_pathlib:
                visualization_folder = Path(visualization_folder)  # type: ignore[arg-type]
            visualization_folder = cast(Path, visualization_folder)
            for tree in range(num_trees):
                for layer in range(num_layers):
                    if expected_circles_or_ellipses[tree, layer, 4] == -1:
                        continue
                    if expected_circles_or_ellipses[tree, layer, 3] == -1:
                        expected_visualization_path = (
                            visualization_folder / "test" / f"preliminary_circle_trunk_{tree}_layer_{layer}.png"
                        )
                    else:
                        expected_visualization_path = (
                            visualization_folder / "test" / f"preliminary_ellipse_trunk_{tree}_layer_{layer}.png"
                        )

                    assert expected_visualization_path.exists()

        (
            layer_circles,
            layer_ellipses,
            layer_heights,
            trunk_layer_xy,
            batch_lengths_xy,
        ) = algorithm.fit_exact_circles_and_ellipses_to_trunks(
            trunk_layer_xyz, preliminary_circles_or_ellipses, point_cloud_id=point_cloud_id
        )
        assert layer_circles.dtype == scalar_type
        assert layer_ellipses.dtype == scalar_type
        assert layer_heights.dtype == scalar_type
        assert trunk_layer_xy.dtype == scalar_type

        if create_visualization:
            if not use_pathlib:
                visualization_folder = Path(visualization_folder)  # type: ignore[arg-type]
            visualization_folder = cast(Path, visualization_folder)
            for tree in range(num_trees):
                for layer in range(num_layers):
                    if expected_circles_or_ellipses[tree, layer, 4] == -1:
                        continue
                    if expected_circles_or_ellipses[tree, layer, 3] == -1:
                        expected_visualization_path = (
                            visualization_folder / "test" / f"exact_circle_trunk_{tree}_layer_{layer}.png"
                        )
                    else:
                        expected_visualization_path = (
                            visualization_folder / "test" / f"exact_ellipse_trunk_{tree}_layer_{layer}.png"
                        )

                    assert expected_visualization_path.exists()

        np.testing.assert_almost_equal(expected_layer_heigths, layer_heights)

        assert len(layer_circles) == num_trees
        assert len(layer_ellipses) == num_trees
        assert layer_circles.shape[1] == num_layers
        assert layer_ellipses.shape[1] == num_layers
        assert len(batch_lengths_xy) == num_trees * num_layers
        assert batch_lengths_xy.sum() == len(trunk_layer_xy)

        for tree in range(num_trees):
            for layer in range(num_layers):
                ellipse_radius_ratio = (
                    expected_circles_or_ellipses[tree, layer, 3] / expected_circles_or_ellipses[tree, layer, 2]
                )
                is_skip_layer = expected_circles_or_ellipses[tree, layer, 2] == -1
                is_invalid_ellipse_layer = (
                    expected_circles_or_ellipses[tree, layer, 3] != -1
                    and ellipse_radius_ratio < trunk_search_ellipse_filter_threshold
                )
                if is_skip_layer:
                    assert batch_lengths_xy[tree * num_layers + layer] == 0
                else:
                    assert (
                        batch_lengths_xy[tree * num_layers + layer]
                        >= settings["trunk_search_circle_fitting_min_points"]
                    )

                if is_skip_layer or is_invalid_ellipse_layer:
                    assert (layer_circles[tree, layer] == -1).all()
                    assert (layer_ellipses[tree, layer] == -1).all()
                    continue
                if expected_circles_or_ellipses[tree, layer, 3] != -1:
                    np.testing.assert_almost_equal(
                        expected_circles_or_ellipses[tree, layer], layer_ellipses[tree, layer], decimal=decimal
                    )
                    assert (layer_circles[tree, layer] == -1).all()
                    continue

                np.testing.assert_almost_equal(
                    expected_circles_or_ellipses[tree, layer, :3], layer_circles[tree, layer], decimal=decimal
                )
                assert (layer_ellipses[tree, layer, 2:4] > 0).all()

    def test_filter_instances_trunk_layer_variance(self):
        algorithm = TreeXAlgorithm(
            trunk_search_circle_fitting_std_num_layers=3, trunk_search_circle_fitting_max_std=0.1
        )

        layer_circles = np.array(
            [
                [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                [[0, 0, 2], [0, 0, 0.95], [0, 0, 1.4], [0, 0.1, 0.1]],
                [[-1, -1, -1], [0, 0, 1], [-1, -1, -1], [-1, -1, -1]],
                [[0, 0, 1], [0, 0, 0.95], [0, 0, 1.2], [0, 0.1, 0.9]],
            ],
            dtype=np.float64,
        )
        layer_ellipses = np.array(
            [
                [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]],
                [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]],
                [[0, 0, 1, 0.9, 0], [0, 0, 0.95, 0.9, 0], [0, 0, 0.9, 0.85, 0], [0, 0, 1.2, 0.85, 0]],
                [[0, 0, 1, 0.9, 0], [0, 0, 1, 0.9, 0], [0, 0, 1.2, 1.1, 0], [0, 0.1, 0.9, 0.8, 0]],
            ],
            dtype=np.float64,
        )

        expected_filter_mask = np.array([False, False, True, True])
        expected_best_circle_combinations = np.array([[-1, -1, -1], [0, 1, 3]], dtype=np.int64)
        expected_best_ellipse_combinations = np.array([[0, 1, 2], [-1, -1, -1]], dtype=np.int64)

        filter_mask, best_circle_combination, best_ellipse_combination = algorithm.filter_instances_trunk_layer_std(
            layer_circles, layer_ellipses
        )

        np.testing.assert_array_equal(expected_filter_mask, filter_mask)
        np.testing.assert_array_equal(expected_best_circle_combinations, np.sort(best_circle_combination, axis=-1))
        np.testing.assert_array_equal(expected_best_ellipse_combinations, np.sort(best_ellipse_combination, axis=-1))

    def test_rename_visualizations_after_filtering(self, cache_dir):
        cache_dir = Path(cache_dir)
        algorithm = TreeXAlgorithm(trunk_search_circle_fitting_num_layers=2, visualization_folder=cache_dir)
        filter_mask = np.array([False, True])
        point_cloud_id = "test"

        paths = [
            (
                cache_dir / point_cloud_id / "preliminary_ellipse_trunk_0_layer_1.png",
                cache_dir / point_cloud_id / "preliminary_ellipse_trunk_1_layer_1_invalid.png",
            ),
            (
                cache_dir / point_cloud_id / "preliminary_circle_trunk_1_layer_0.png",
                cache_dir / point_cloud_id / "preliminary_circle_trunk_0_layer_0_valid.png",
            ),
            (
                cache_dir / point_cloud_id / "preliminary_circle_trunk_1_layer_1.png",
                cache_dir / point_cloud_id / "preliminary_circle_trunk_0_layer_1_valid.png",
            ),
            (
                cache_dir / point_cloud_id / "exact_ellipse_trunk_0_layer_1.png",
                cache_dir / point_cloud_id / "exact_ellipse_trunk_1_layer_1_invalid.png",
            ),
            (
                cache_dir / point_cloud_id / "exact_circle_trunk_1_layer_0.png",
                cache_dir / point_cloud_id / "exact_circle_trunk_0_layer_0_valid.png",
            ),
            (
                cache_dir / point_cloud_id / "exact_circle_trunk_1_layer_1.png",
                cache_dir / point_cloud_id / "exact_circle_trunk_0_layer_1_valid.png",
            ),
        ]

        for existing_path, _ in paths:
            existing_path.parent.mkdir(exist_ok=True, parents=True)
            existing_path.touch()

        # create target file for first file to test that existing files are overwritten
        paths[0][1].touch()

        algorithm.rename_visualizations_after_filtering(filter_mask, point_cloud_id=point_cloud_id)

        for _, renamed_path in paths:
            assert renamed_path.exists()

    @pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
    def test_compute_trunk_positions_circles(self, scalar_type: np.dtype):
        algorithm = TreeXAlgorithm(trunk_search_circle_fitting_std_num_layers=3)

        layer_circles = np.array([[[0, 0, 1], [0.5, 0, 1], [1, 0, 1], [2, 0, 0.1]]], dtype=scalar_type)
        layer_ellipses = np.array(
            [[[-1, -1, -1, -1, -1], [0.5, 0, 1.1, 0.99, 0], [1, 0, 1.1, 0.99, 0], [2, 0, 1.1, 0.99, 0.1]]],
            dtype=scalar_type,
        )
        layer_heights = np.array([1, 1.5, 2, 2.5])

        best_circle_combination = np.array([[0, 2, 1]], dtype=np.int64)
        best_ellipse_combination = np.array([[-1, -1, -1]], dtype=np.int64)

        expected_trunk_positions = np.array([[0.3, 0]], dtype=scalar_type)

        trunk_positions = algorithm.compute_trunk_positions(
            layer_circles, layer_ellipses, layer_heights, best_circle_combination, best_ellipse_combination
        )

        assert expected_trunk_positions.dtype == trunk_positions.dtype
        np.testing.assert_almost_equal(expected_trunk_positions, trunk_positions)

    @pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
    def test_compute_trunk_positions_ellipses(self, scalar_type: np.dtype):
        algorithm = TreeXAlgorithm(trunk_search_circle_fitting_std_num_layers=3)

        layer_circles = np.array([[[0, 0, 1], [-1, -1, -1], [1, 0, 1], [-1, -1, -1]]], dtype=scalar_type)
        layer_ellipses = np.array(
            [[[0, 0, 1, 1, 0.1], [0.5, 0, 1, 1, 0], [1, 0, 1, 1, 0.2], [-1, -1, -1, -1, -1]]],
            dtype=scalar_type,
        )
        layer_heights = np.array([1, 1.5, 2, 2.5])

        best_circle_combination = np.array([[-1, -1, -1]], dtype=np.int64)
        best_ellipse_combination = np.array([[0, 2, 1]], dtype=np.int64)

        expected_trunk_positions = np.array([[0.3, 0]], dtype=scalar_type)

        trunk_positions = algorithm.compute_trunk_positions(
            layer_circles, layer_ellipses, layer_heights, best_circle_combination, best_ellipse_combination
        )

        assert expected_trunk_positions.dtype == trunk_positions.dtype
        np.testing.assert_almost_equal(expected_trunk_positions, trunk_positions)

    def test_radius_estimation_gam_circle(self):
        algorithm = TreeXAlgorithm()

        circles = np.array([[1, 1, 1]])
        points = generate_circle_points(circles, min_points=50, max_points=50)

        radius_with_full_circle, polygon_vertices_with_full_ellipse = algorithm.radius_estimation_gam(
            points, circles[0, :2], 6
        )

        assert circles[0, 2] == pytest.approx(radius_with_full_circle, abs=0.001)
        assert polygon_vertices_with_full_ellipse.ndim == 2
        assert (points.min(axis=0) < polygon_vertices_with_full_ellipse.mean(axis=0)).all()
        assert (points.max(axis=0) > polygon_vertices_with_full_ellipse.mean(axis=0)).all()

        radius_with_missing_part, polygon_vertices_with_missing_part = algorithm.radius_estimation_gam(
            points[:30], circles[0, :2], 7
        )

        assert circles[0, 2] == pytest.approx(radius_with_missing_part, abs=0.001)
        assert polygon_vertices_with_missing_part.ndim == 2
        assert (points[:30].min(axis=0) < polygon_vertices_with_missing_part.mean(axis=0)).all()
        assert (points[:30].max(axis=0) > polygon_vertices_with_missing_part.mean(axis=0)).all()

    def test_radius_estimation_gam_ellipse(self):
        algorithm = TreeXAlgorithm()

        ellipses = np.array([[1, 1, 1.2, 0.9, 0]])
        points = generate_ellipse_points(ellipses, min_points=50, max_points=50)

        radius_with_full_ellipse, polygon_vertices_with_full_ellipse = algorithm.radius_estimation_gam(
            points, ellipses[0, :2], 8
        )

        expected_radius = (ellipses[0, 2] + ellipses[0, 3]) / 2

        assert expected_radius == pytest.approx(radius_with_full_ellipse, abs=0.02)
        assert polygon_vertices_with_full_ellipse.ndim == 2
        assert (points.min(axis=0) < polygon_vertices_with_full_ellipse.mean(axis=0)).all()
        assert (points.max(axis=0) > polygon_vertices_with_full_ellipse.mean(axis=0)).all()

        radius_with_missing_part, polygon_vertices_with_missing_part = algorithm.radius_estimation_gam(
            points[:30], ellipses[0, :2], 9
        )

        assert expected_radius == pytest.approx(radius_with_missing_part, abs=0.1)
        assert polygon_vertices_with_missing_part.ndim == 2
        assert (points[:30].min(axis=0) < polygon_vertices_with_missing_part.mean(axis=0)).all()
        assert (points[:30].max(axis=0) > polygon_vertices_with_missing_part.mean(axis=0)).all()

    @pytest.mark.parametrize("create_visualization", [False, True])
    @pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
    def test_compute_trunk_diameters(  # pylint: disable=too-many-locals, too-many-branches
        self, create_visualization: bool, scalar_type: np.dtype, cache_dir
    ):
        point_cloud_id = None
        visualization_folder = None
        if create_visualization:
            point_cloud_id = "test"
            visualization_folder = Path(cache_dir)

        num_instances = 2
        num_layers = 4
        algorithm = TreeXAlgorithm(
            trunk_search_circle_fitting_num_layers=num_layers,
            trunk_search_circle_fitting_std_num_layers=num_layers - 1,
            visualization_folder=visualization_folder,
        )

        trunk_layer_xy = []
        batch_lengths_xy = np.zeros(num_instances * num_layers, dtype=np.int64)
        layer_heights = np.array([1, 2, 3, 4], dtype=scalar_type)
        layer_circles = np.full((num_instances, num_layers, 3), fill_value=-1, dtype=scalar_type)
        layer_ellipses = np.full((num_instances, num_layers, 5), fill_value=-1, dtype=scalar_type)

        for layer in range(num_layers):
            if layer == num_layers - 1:
                circles = np.array([[0, 0, 1.3]])
            else:
                circles = np.array([[0, 0, 1 - 0.1 * layer]])
            layer_circles[0, layer] = circles[0]
            circle_points = generate_circle_points(circles, min_points=50, max_points=50)
            trunk_layer_xy.append(circle_points)
            batch_lengths_xy[layer] = len(circle_points)

        for layer in range(num_layers):
            if layer == 0:
                ellipses = np.array([[0, 0, 1.3, 0.8, 0]])
            else:
                ellipses = np.array([[0, 0, 1.0 - 0.1 * layer, 1.0 - 0.1 * layer, 0]])

            layer_ellipses[1, layer] = ellipses[0]
            ellipse_points = generate_ellipse_points(ellipses, min_points=50, max_points=50)
            trunk_layer_xy.append(ellipse_points)
            batch_lengths_xy[num_layers + layer] = len(ellipse_points)

        trunk_layer_xy_np = np.concatenate(trunk_layer_xy).astype(scalar_type)

        best_circle_combination = np.array([[0, 2, 1], [-1, -1, -1]], dtype=np.int64)
        best_ellipse_combination = np.array([[-1, -1, -1], [2, 1, 3]], dtype=np.int64)

        expected_visualization_paths = []
        if create_visualization:
            for label in range(2):
                if best_circle_combination[label][0] != -1:
                    selected_layers = best_circle_combination[label]
                else:
                    selected_layers = best_ellipse_combination[label]
                for layer in selected_layers:
                    visualization_folder = cast(Path, visualization_folder)
                    expected_visualization_paths.append(
                        visualization_folder / "test" / f"gam_trunk_{label}_layer_{layer}.png"
                    )

        expected_trunk_radii = np.array([1 - 0.03, 1 - 0.03], dtype=np.float64)

        trunk_diameters = algorithm.compute_trunk_diameters(
            layer_circles,
            layer_ellipses,
            layer_heights,
            trunk_layer_xy_np,
            batch_lengths_xy,
            best_circle_combination,
            best_ellipse_combination,
            point_cloud_id=point_cloud_id,
        )

        assert trunk_diameters.dtype == scalar_type
        np.testing.assert_almost_equal(expected_trunk_radii * 2, trunk_diameters, decimal=4)

        if create_visualization:
            for expected_path in expected_visualization_paths:
                assert expected_path.exists()

    @pytest.mark.parametrize("crs", [None, "EPSG:4326"])
    def test_export_dtm(self, crs: Optional[str], cache_dir):
        algorithm = TreeXAlgorithm(visualization_folder=cache_dir)

        dtm = np.array([[1.1, 2, 1], [1.2, 1.4, 1.3]], dtype=np.float64)
        dtm_offset = np.array([10, 20], dtype=np.float64)
        expected_file_path = Path(cache_dir) / "test_dtm.tif"

        algorithm.export_dtm(dtm, dtm_offset, "test", crs=crs)

        assert expected_file_path.exists()

    def test_export_dtm_invalid(self):
        algorithm = TreeXAlgorithm(visualization_folder=None)

        dtm = np.array([[1.1, 2, 1], [1.2, 1.4, 1.3]], dtype=np.float64)
        dtm_offset = np.array([10, 20], dtype=np.float64)

        with pytest.raises(ValueError):
            algorithm.export_dtm(dtm, dtm_offset, "test")

    @pytest.mark.parametrize("storage_layout", ["C", "F"])
    @pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
    def test_segment_crowns(self, storage_layout: str, scalar_type: np.dtype):  # pylint: disable=too-many-locals
        point_spacing = 0.1
        ground_plane = generate_grid_points((100, 100), point_spacing)
        ground_plane = np.column_stack([ground_plane, np.zeros(len(ground_plane), dtype=np.float64)])

        trunk_1 = np.full((30, 3), fill_value=2.05, dtype=np.float64)
        trunk_1[:, 2] = np.arange(30).astype(np.float64) * point_spacing

        trunk_2 = np.full((35, 3), fill_value=4.65, dtype=np.float64)
        trunk_2[:, 2] = np.arange(35).astype(np.float64) * point_spacing

        crown_1 = generate_grid_points((20, 20, 20), point_spacing)
        crown_1[:, :2] += 1.05
        crown_1[:, 2] += 3.1
        crown_2 = generate_grid_points((30, 30, 30), point_spacing)
        crown_2[:, :2] += 3.15
        crown_2[:, 2] += 3.6

        xyz = np.concatenate([ground_plane, crown_1, trunk_1, crown_2, trunk_2]).astype(
            scalar_type, order=storage_layout
        )
        expected_instance_ids = np.full(len(xyz), fill_value=-1, dtype=np.int64)
        start_tree_1 = len(ground_plane)
        end_tree_1 = start_tree_1 + len(crown_1) + len(trunk_1)
        expected_instance_ids[start_tree_1:end_tree_1] = 0
        expected_instance_ids[end_tree_1:] = 1

        distance_to_dtm = xyz[:, 2]

        is_tree = np.zeros(len(xyz), dtype=bool)
        is_tree[len(ground_plane) :] = True

        tree_positions = np.array([[2.05, 2.05], [4.65, 4.65]], dtype=scalar_type, order=storage_layout)
        trunk_diameters = np.array([0.01, 0.01], dtype=scalar_type, order=storage_layout)

        region_growing_z_scale = 2
        region_growing_seed_layer_height = 0.6
        max_cum_search_dist_without_terrain = (1.3 - region_growing_seed_layer_height / 2) / region_growing_z_scale

        algorithm = TreeXAlgorithm(
            region_growing_voxel_size=0.025,
            region_growing_cum_search_dist_include_terrain=max_cum_search_dist_without_terrain + 0.2,
            region_growing_seed_layer_height=region_growing_seed_layer_height,
            region_growing_z_scale=region_growing_z_scale,
        )

        instance_ids = algorithm.segment_crowns(xyz, distance_to_dtm, is_tree, tree_positions, trunk_diameters)

        assert len(xyz) == len(instance_ids)
        assert len(np.unique(instance_ids)) == 3
        np.testing.assert_array_equal(expected_instance_ids[start_tree_1:], instance_ids[start_tree_1:])
        # some of the terrain points around the trunks are assigned to the tree instances
        assert (instance_ids[:start_tree_1] != -1).sum() == 8

    @pytest.mark.parametrize("key", ["xyz", "distance_to_dtm", "is_tree", "tree_positions", "trunk_diameters"])
    def test_segment_crowns_invalid_xyz(self, key: str):
        inputs = {
            "xyz": np.random.randn(50, 3).astype(np.float64),
            "distance_to_dtm": np.random.randn(50).astype(np.float64),
            "is_tree": np.ones(50, dtype=bool),
            "tree_positions": np.zeros((3, 2), dtype=np.float64),
            "trunk_diameters": np.zeros(3, dtype=np.float64),
        }
        inputs[key] = inputs[key][: len(inputs[key]) - 1]

        algorithm = TreeXAlgorithm()

        with pytest.raises(ValueError):
            algorithm.segment_crowns(*inputs.values())

    @pytest.mark.parametrize("create_visualizations", [False, True])
    @pytest.mark.parametrize("use_intensities", [True, False])
    @pytest.mark.parametrize("storage_layout", ["C", "F"])
    @pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
    def test_full_algorithm(
        self, create_visualizations: bool, use_intensities: bool, storage_layout: str, scalar_type: np.dtype, cache_dir
    ):  # pylint: disable=too-many-locals, too-many-statements
        point_spacing = 0.1
        min_intensity = 5000
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
        intensities_trunk_1 = np.full(len(trunk_1), fill_value=min_intensity - 1, dtype=scalar_type)

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
        intensities_trunk_2 = np.full(len(trunk_2), fill_value=min_intensity + 1, dtype=scalar_type)

        crown_1 = generate_grid_points((20, 20, 20), point_spacing)
        crown_1[:, :2] += 1.05
        crown_1[:, 2] += 3.1
        intensities_crown_1 = np.full(len(crown_1), fill_value=min_intensity - 1, dtype=scalar_type)
        crown_2 = generate_grid_points((30, 30, 30), point_spacing)
        crown_2[:, :2] += 4.15
        crown_2[:, 2] += 3.6
        intensities_crown_2 = np.full(len(crown_2), fill_value=min_intensity + 1, dtype=scalar_type)

        expected_trunk_positions = np.array([[2.05, 2.05], [5.65, 5.65]], dtype=np.float64)
        expected_trunk_diameters = np.array([0.3, 0.5], dtype=np.float64)
        expected_tree_heights = np.array([5.0, 6.5], dtype=np.float64)

        xyz = np.concatenate([ground_plane, crown_1, np.array(trunk_1), crown_2, np.array(trunk_2)]).astype(
            scalar_type, order=storage_layout
        )
        if use_intensities:
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

        visualization_folder = None
        point_cloud_id = None
        if create_visualizations:
            visualization_folder = cache_dir
            point_cloud_id = "test"

        algorithm = TreeXAlgorithm(
            trunk_search_dbscan_2d_eps=0.05,
            trunk_search_dbscan_3d_eps_small=0.25,
            trunk_search_dbscan_3d_min_points_small=5,
            trunk_search_dbscan_3d_eps_large=0.25,
            trunk_search_dbscan_3d_min_points_large=5,
            trunk_search_min_cluster_intensity=min_intensity,
            visualization_folder=visualization_folder,
        )

        instance_ids, trunk_positions, trunk_diameters = algorithm(
            xyz, intensities=intensities, point_cloud_id=point_cloud_id
        )

        assert trunk_positions.dtype == scalar_type
        assert trunk_diameters.dtype == scalar_type

        tree_heights = np.empty(len(np.unique(instance_ids)) - 1, dtype=np.float64)
        for instance_id in np.unique(instance_ids):
            instance_points = xyz[instance_ids == instance_id]
            if len(instance_points) > 0:
                tree_heights[instance_id] = instance_points[:, 2].max() - instance_points[:, 2].min()

        assert len(xyz) == len(instance_ids)

        if use_intensities:
            assert len(np.unique(instance_ids)) == 2
            np.testing.assert_almost_equal(expected_trunk_positions[1:], trunk_positions, decimal=2)
            np.testing.assert_almost_equal(expected_trunk_diameters[1:], trunk_diameters, decimal=2)
            np.testing.assert_almost_equal(expected_tree_heights[1:], tree_heights, decimal=2)
        else:
            assert len(np.unique(instance_ids)) == 3
            np.testing.assert_almost_equal(expected_trunk_positions, trunk_positions, decimal=2)
            np.testing.assert_almost_equal(expected_trunk_diameters, trunk_diameters, decimal=2)
            np.testing.assert_almost_equal(expected_tree_heights, tree_heights, decimal=2)

        algorithm = TreeXAlgorithm(trunk_search_min_cluster_points=10000)

        instance_ids, trunk_positions, trunk_diameters = algorithm(xyz)

        assert trunk_positions.dtype == scalar_type
        assert trunk_diameters.dtype == scalar_type
        assert len(xyz) == len(instance_ids)
        assert (instance_ids == -1).all()

    def test_full_algorithm_invalid_inputs(self):
        algorithm = TreeXAlgorithm()

        xyz = np.zeros((10, 3), dtype=np.float64)
        intensities = np.zeros((11), dtype=np.float64)

        with pytest.raises(ValueError):
            algorithm(xyz, intensities)
