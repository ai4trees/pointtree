"""Tests for pointtree.instance_segmentation.MultiStageSegmentationAlgorithm."""

import os
from typing import List, Literal, Optional
import shutil

import numpy as np
import pandas as pd
import pytest

from pointtree.instance_segmentation import MultiStageAlgorithm


class TestMultiStageAlgorithm:  # pylint: disable=too-many-public-methods
    """Tests for pointtree.instance_segmentation.MultiStageSegmentationAlgorithm."""

    @pytest.fixture
    def cache_dir(self):
        cache_dir = "./tmp/test/io/TestMultiStageAlgorithm"
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
        shutil.rmtree(cache_dir)

    def test_cluster_trunk_points_empty(self):
        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1)

        tree_coords = np.empty(0, dtype=float)
        classification = np.empty(0, dtype=np.int64)

        instance_ids, unique_instance_ids = algorithm.cluster_trunks(tree_coords, classification)

        assert len(instance_ids) == 0
        assert len(unique_instance_ids) == 0

    @pytest.mark.parametrize("use_branch_points", [True, False])
    def test_cluster_trunk_points(self, use_branch_points: bool):
        branch_class_id = 2 if use_branch_points else None
        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1, branch_class_id=branch_class_id)

        trunk_coords = np.zeros((10, 3))
        trunk_coords[:, 2] = np.arange(len(trunk_coords))
        trunk_coords = np.concatenate(
            [
                trunk_coords + [0.5, 0.5, 0],
                trunk_coords + [-0.5, 0.5, 0],
                trunk_coords + [0.5, -0.5, 0],
                trunk_coords - [0.5, 0.5, 0],
            ]
        )
        trunk_coords = np.concatenate([trunk_coords, trunk_coords + [5, 0, 0]])

        crown_coords = np.random.randn(20, 3)

        tree_coords = np.concatenate([trunk_coords, crown_coords])

        classification = np.zeros(len(trunk_coords) + len(crown_coords))
        if use_branch_points:
            classification[int(len(trunk_coords) / 2) : len(trunk_coords)] = 2
        classification[len(trunk_coords) :] = 1

        expected_instance_ids = np.array([0] * 40 + [1] * 40 + [-1] * 20)
        expected_unique_ids = np.array([0, 1])

        instance_ids, unique_instance_ids = algorithm.cluster_trunks(tree_coords, classification)

        np.testing.assert_array_equal(expected_instance_ids, instance_ids)
        np.testing.assert_array_equal(expected_unique_ids, unique_instance_ids)

    def test_compute_trunk_positions(self):
        tree_coords = np.array([[0, 0, 0], [1, 2, 3], [4, 4, 4], [5, 6, 6]], dtype=float)
        instance_ids = np.array([0, 0, -1, 1], dtype=np.int64)
        unique_instance_ids = np.array([0, 1], dtype=np.int64)
        expected_trunk_positions = np.array([[0.5, 1], [5, 6]], dtype=float)

        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1)

        trunk_positions = algorithm.compute_trunk_positions(tree_coords, instance_ids, unique_instance_ids)

        np.testing.assert_array_equal(expected_trunk_positions, trunk_positions)

    def test_create_height_map_empty(self):
        points = np.empty((0, 0), dtype=float)
        grid_size = 1
        height_map, count_map, grid_origin = MultiStageAlgorithm.create_height_map(points, grid_size)

        assert len(height_map) == 0
        assert len(count_map) == 0
        assert len(grid_origin) == 0

    @pytest.mark.parametrize("bounding_box", [None, np.array([[1, 0], [1.9, 2.7]], dtype=float)])
    def test_create_height_map(self, bounding_box: Optional[np.ndarray]):
        points = np.array(
            [
                [1, 1, 1],
                [1.5, 1, 2],
                [1.9, 1.8, 1.8],  # first grid cell (1, 1)
                [2.6, 1.1, 5],
                [2.2, 1.8, 1.4],  # second grid cell (2, 1)
                [1.1, 2.7, 3],  # third grid cell (1, 2)
            ],
            dtype=float,
        )
        grid_size = 1

        if bounding_box is None:
            expected_height_map = np.array([[1, 2], [4, 0]], dtype=float)
            expected_count_map = np.array([[3, 1], [2, 0]], dtype=np.int64)
            expected_grid_origin = np.array([1, 1], dtype=float)
        else:
            expected_height_map = np.array([[0, 1, 2]], dtype=float)
            expected_count_map = np.array([[0, 3, 1]], dtype=np.int64)
            expected_grid_origin = bounding_box[0]

        height_map, count_map, grid_origin = MultiStageAlgorithm.create_height_map(
            points, grid_size, bounding_box=bounding_box
        )

        np.testing.assert_array_equal(expected_height_map, height_map)
        np.testing.assert_array_equal(expected_count_map, count_map)
        np.testing.assert_array_equal(expected_grid_origin, grid_origin)

    def test_compute_crown_top_positions_empty(self):
        tree_coords = np.empty((0, 0), dtype=float)
        classification = np.empty(0, dtype=np.int64)

        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1)

        (
            crown_top_positions,
            crown_top_positions_grid,
            canopy_height_model,
            grid_origin,
        ) = algorithm.compute_crown_top_positions(tree_coords, classification)

        assert len(crown_top_positions) == 0
        assert len(crown_top_positions_grid) == 0
        assert len(canopy_height_model) == 0
        assert len(grid_origin) == 0

    @pytest.mark.parametrize("algorithm", ["full", "watershed_crown_top_positions"])
    @pytest.mark.parametrize("min_distance_crown_tops", [1.0, 3.0, 10.0])
    @pytest.mark.parametrize("min_points_crown_detection", [1, 3])
    @pytest.mark.parametrize("min_tree_height", [1.0, 3.0, 10.0])
    def test_compute_crown_top_positions(  # pylint: disable=too-many-locals
        self,
        algorithm: Literal["full", "watershed_crown_top_positions"],
        min_distance_crown_tops: float,
        min_points_crown_detection: int,
        min_tree_height: float,
    ):
        expected_grid_origin = np.array([1, 1], dtype=float)

        tree_coords = np.array(
            [
                [0, 1, 1],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 2],
                [1, 2, 1],
                [1, 3, 1],
                [1, 4, 1],
                [1, 5, 1],
                [2, 0, 1],
                [2, 1, 1],
                [3, 1, 3],
                [3, 1, 1],
                [3, 1, 1],
                [4, 4, 1],
                [4, 5, 1],
                [5, 4, 4],
                [5, 5, 1],
                [5, 0, 1],
            ],
            dtype=float,
        )
        tree_coords[:, :2] += expected_grid_origin
        classification = np.ones(len(tree_coords), dtype=np.int64)
        classification[-1] = 0

        expected_canopy_height_model = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [1, 2, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0],
                [0, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 4, 1],
            ],
            dtype=float,
        )

        if min_points_crown_detection <= 1:
            if min_distance_crown_tops <= 2:
                expected_crown_top_positions_grid_list = [[1, 1], [3, 1], [5, 4]]
                tree_heights_list = [2, 3, 4]
            elif min_distance_crown_tops <= np.linalg.norm([2, 3]):
                expected_crown_top_positions_grid_list = [[3, 1], [5, 4]]
                tree_heights_list = [3, 4]
            else:
                expected_crown_top_positions_grid_list = [[5, 4]]
                tree_heights_list = [4]
        else:
            expected_crown_top_positions_grid_list = [[3, 1]]
            tree_heights_list = [3]

        if algorithm == "watershed_crown_top_positions":
            expected_canopy_height_model[5, 0] = 1
            if min_tree_height <= 1 and min_distance_crown_tops <= np.linalg.norm([1, 2]):
                expected_crown_top_positions_grid_list.append([5, 0])
                tree_heights_list.append(1)

        tree_heights = np.array(tree_heights_list, dtype=float)
        expected_crown_top_positions_grid = np.array(expected_crown_top_positions_grid_list, dtype=np.int64)
        expected_crown_top_positions_grid = expected_crown_top_positions_grid[tree_heights > min_tree_height]
        expected_crown_top_positions = expected_crown_top_positions_grid.astype(np.float64) + expected_grid_origin

        algorithm = MultiStageAlgorithm(
            trunk_class_id=0,
            crown_class_id=1,
            algorithm=algorithm,
            grid_size_canopy_height_model=1,
            min_distance_crown_tops=min_distance_crown_tops,
            min_points_crown_detection=min_points_crown_detection,
            min_tree_height=min_tree_height,
            smooth_canopy_height_model=False,
        )

        (
            crown_top_positions,
            crown_top_positions_grid,
            canopy_height_model,
            grid_origin,
        ) = algorithm.compute_crown_top_positions(tree_coords, classification)

        np.testing.assert_array_equal(
            np.unique(expected_crown_top_positions, axis=0), np.unique(crown_top_positions, axis=0)
        )
        np.testing.assert_array_equal(
            np.unique(expected_crown_top_positions_grid, axis=0), np.unique(crown_top_positions_grid, axis=0)
        )
        np.testing.assert_array_equal(expected_canopy_height_model, canopy_height_model)
        np.testing.assert_array_equal(expected_grid_origin, grid_origin)

    def test_compute_crown_top_positions_smoothed_canopy_height_model(self):
        expected_grid_origin = np.array([0, 0], dtype=float)

        tree_coords = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 2, 1],
                [1, 3, 1],
                [2, 1, 1],
                [2, 2, 2],
                [2, 3, 1],
                [3, 1, 1],
                [3, 2, 1],
                [3, 3, 1],
                [3, 4, 1],
                [3, 5, 1],
                [3, 6, 1],
                [4, 1, 1],
                [4, 2, 1],
                [4, 3, 1],
                [4, 4, 6],
                [4, 5, 1],
                [5, 5, 1],
                [5, 5, 1],
                [5, 6, 1],
            ],
            dtype=float,
        )
        tree_coords[:, :2] += expected_grid_origin
        classification = np.ones(len(tree_coords), dtype=np.int64)
        classification[-1] = 0

        expected_crown_top_positions_grid = np.array([[4, 4]], dtype=np.int64)
        expected_crown_top_positions = expected_crown_top_positions_grid.astype(np.float64) + expected_grid_origin

        algorithm = MultiStageAlgorithm(
            trunk_class_id=0,
            crown_class_id=1,
            grid_size_canopy_height_model=1,
            min_distance_crown_tops=2,
            min_points_crown_detection=1,
            min_tree_height=1,
            smooth_canopy_height_model=True,
        )

        crown_top_positions, crown_top_positions_grid, _, grid_origin = algorithm.compute_crown_top_positions(
            tree_coords, classification
        )

        np.testing.assert_array_equal(
            np.unique(expected_crown_top_positions, axis=0), np.unique(crown_top_positions, axis=0)
        )
        np.testing.assert_array_equal(
            np.unique(expected_crown_top_positions_grid, axis=0), np.unique(crown_top_positions_grid, axis=0)
        )
        np.testing.assert_array_equal(expected_grid_origin, grid_origin)

    def test_match_trunk_and_crown_tops_no_trunks(self):
        trunk_positions = np.empty(0, dtype=np.float64)
        crown_positions = np.array([[0, 0], [1, 1]], dtype=np.float64)

        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1)

        tree_positions = algorithm.match_trunk_and_crown_tops(trunk_positions, crown_positions)

        np.testing.assert_array_equal(crown_positions, tree_positions)

    def test_match_trunk_and_crown_tops_no_crown_tops(self):
        trunk_positions = np.array([[0, 0], [1, 1]], dtype=np.float64)
        crown_positions = np.empty(0, dtype=np.float64)

        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1)

        tree_positions = algorithm.match_trunk_and_crown_tops(trunk_positions, crown_positions)

        np.testing.assert_array_equal(trunk_positions, tree_positions)

    def test_match_trunk_and_crown_tops(self):
        trunk_positions = np.array([[0, 0], [1, 1]], dtype=np.float64)
        crown_positions = np.array([[2, 1], [10, 10]], dtype=np.float64)

        expected_tree_positions = np.array([[0, 0], [2, 1], [10, 10]], dtype=np.float64)

        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1, distance_match_trunk_and_crown_top=5)

        tree_positions = algorithm.match_trunk_and_crown_tops(trunk_positions, crown_positions)

        np.testing.assert_array_equal(expected_tree_positions, tree_positions)

    def test_watershed_segmentation(self):
        canopy_height_model = np.array(
            [
                [1, 2, 2, 0, 0, 0],
                [2, 3, 2, 0, 0, 0],
                [0, 2, 1, 3, 1, 0],
                [0, 0, 2, 5, 1, 1],
                [0, 0, 3, 4, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )

        tree_positions = np.array([[1, 1], [4, 3]], dtype=np.int64)

        expected_watershed_mask_with_border = np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 1, 0, 2, 2, 0],
                [0, 0, 2, 2, 2, 2],
                [0, 0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )

        expected_watershed_mask_without_border = np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 1, 2, 2, 2, 0],
                [0, 0, 2, 2, 2, 2],
                [0, 0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )

        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1, grid_size_canopy_height_model=1)

        watershed_mask_with_border, watershed_mask_without_border = algorithm.watershed_segmentation(
            canopy_height_model, tree_positions
        )

        np.testing.assert_array_equal(expected_watershed_mask_with_border, watershed_mask_with_border)
        np.testing.assert_array_equal(expected_watershed_mask_without_border, watershed_mask_without_border)

    def test_watershed_correction(self, cache_dir: str):
        watershed_mask_with_border = np.array(
            [
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 0, 1, 1],
                [0, 1, 0, 3, 0, 1],
                [0, 1, 1, 0, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 2, 2],
            ],
            dtype=np.int64,
        )

        watershed_mask_without_border = np.array(
            [
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 3, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 2, 2],
            ],
            dtype=np.int64,
        )

        # depending on the version of skimage, the exact position of the Watershed boundary can vary
        expected_corrected_watershed_mask_with_border_1 = np.array(
            [
                [0, 3, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3],
                [0, 3, 0, 1, 0, 3],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 2, 2],
            ],
            dtype=np.int64,
        )

        expected_corrected_watershed_mask_with_border_2 = np.array(
            [
                [0, 3, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3],
                [0, 0, 0, 1, 0, 0],
                [0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 2, 2],
            ],
            dtype=np.int64,
        )

        expected_corrected_watershed_mask_without_border_1 = np.array(
            [
                [0, 3, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3],
                [0, 3, 3, 1, 3, 3],
                [0, 3, 1, 1, 1, 3],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 2, 2],
            ],
            dtype=np.int64,
        )

        expected_corrected_watershed_mask_without_border_2 = np.array(
            [
                [0, 3, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3],
                [0, 3, 3, 1, 3, 3],
                [0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 2, 2],
            ],
            dtype=np.int64,
        )

        tree_positions_grid = np.array([[3, 3], [8, 4], [2, 3]], dtype=np.int64)

        algorithm = MultiStageAlgorithm(
            trunk_class_id=0, crown_class_id=1, grid_size_canopy_height_model=1, visualization_folder=cache_dir
        )

        corrected_watershed_mask_with_border, corrected_watershed_mask_without_border = algorithm.watershed_correction(
            watershed_mask_with_border, watershed_mask_without_border, tree_positions_grid, "test"
        )

        assert np.array_equal(
            expected_corrected_watershed_mask_with_border_1, corrected_watershed_mask_with_border
        ) or np.array_equal(expected_corrected_watershed_mask_with_border_2, corrected_watershed_mask_with_border)
        assert np.array_equal(
            expected_corrected_watershed_mask_without_border_1, corrected_watershed_mask_without_border
        ) or np.array_equal(expected_corrected_watershed_mask_without_border_2, corrected_watershed_mask_without_border)

        assert os.path.exists(os.path.join(cache_dir, "voronoi_labels_with_border_test_3.png"))
        assert os.path.exists(os.path.join(cache_dir, "voronoi_labels_without_border_test_3.png"))

    @pytest.mark.parametrize("correct_watershed", [True, False])
    def test_coarse_segmentation(self, correct_watershed: bool, cache_dir: str):  # pylint: disable=too-many-locals
        grid_size = 0.5
        grid_origin = np.array([0, 0], dtype=np.float64)

        tree_coords = (
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [1, 1, 1],
                    [1, 2, 1],
                    [1, 3, 1],
                    [2, 1, 1],
                    [2, 2, 2],
                    [2, 3, 1],
                    [3, 1, 1],
                    [3, 2, 2],
                    [3, 3, 1],
                    [4, 1, 0.5],
                    [4, 2, 0.5],
                    [4, 3, 0.5],
                    [5, 1, 1],
                    [5, 2, 1],
                    [5, 3, 1],
                    [5, 4, 1],
                    [6, 1, 1],
                    [6, 2, 2],
                    [6, 3, 2],
                    [6, 4, 1],
                    [7, 1, 1],
                    [7, 2, 2],
                    [7, 3, 2],
                    [7, 4, 1],
                    [8, 1, 1],
                    [8, 2, 1],
                    [8, 3, 1],
                    [8, 4, 1],
                    [9, 1, 2],
                ],
                dtype=np.float64,
            )
            * grid_size
        )

        instance_ids = np.ones(len(tree_coords), dtype=np.int64) * -1
        instance_ids[1] = 3
        instance_ids[-1] = 0

        canopy_height_model = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 1, 2, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0.5, 0.5, 0.5, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 2, 2, 2, 2, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )

        tree_positions_grid = np.array([[9, 1], [2, 2], [6, 3], [0, 0]], dtype=np.int64)

        expected_watershed_mask_without_border = np.array(
            [
                [4, 0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0, 0],
                [0, 2, 2, 2, 0, 0],
                [0, 2, 2, 2, 0, 0],
                [0, 3, 3, 3, 0, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )

        expected_watershed_mask_with_border = np.array(
            [
                [4, 0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0, 0],
                [0, 2, 2, 2, 0, 0],
                [0, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )

        expected_instance_ids = np.array(
            [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
            dtype=np.int64,
        )

        expected_unique_instance_ids = np.array([0, 1, 2, 3], dtype=np.int64)

        algorithm = MultiStageAlgorithm(
            trunk_class_id=0,
            crown_class_id=1,
            visualization_folder=cache_dir,
            grid_size_canopy_height_model=grid_size,
            correct_watershed=correct_watershed,
        )

        (
            instance_ids,
            unique_instance_ids,
            watershed_mask_with_border,
            watershed_mask_without_border,
        ) = algorithm.coarse_segmentation(
            tree_coords,
            instance_ids,
            tree_positions_grid,
            canopy_height_model,
            grid_origin,
            point_cloud_id="test",
        )

        np.testing.assert_array_equal(expected_watershed_mask_without_border, watershed_mask_without_border)
        np.testing.assert_array_equal(expected_watershed_mask_with_border, watershed_mask_with_border)
        np.testing.assert_array_equal(expected_instance_ids, instance_ids)
        np.testing.assert_array_equal(expected_unique_instance_ids, unique_instance_ids)

        assert os.path.exists(os.path.join(cache_dir, "watershed_with_border_test.png"))
        assert os.path.exists(os.path.join(cache_dir, "watershed_without_border_test.png"))

        if correct_watershed:
            assert os.path.exists(os.path.join(cache_dir, "watershed_labels_voronoi_with_border_test.png"))
            assert os.path.exists(os.path.join(cache_dir, "watershed_labels_voronoi_without_border_test.png"))

    def test_determine_overlapping_crowns(self):
        tree_coords = np.array(
            [
                [0, 0, 1],
                [0, 1, 1],
                [0, 2, 1],
                [1, 0, 1],
                [1, 1, 2],
                [1, 2, 1],
                [2, 0, 1],
                [2, 1, 1],
                [2, 2, 1],
                [2, 3, 1],
                [3, 1, 1],
                [3, 2, 2],
                [3, 3, 1],
                [4, 1, 1],
                [4, 2, 1],
                [4, 3, 1],
                [6, 1, 1],
                [1, 1, 1],
                [1, 1, 0.5],
                [3, 2, 0.5],
            ],
            dtype=np.float64,
        )

        classification = np.ones(len(tree_coords), dtype=np.int64)
        classification[-3] = 2
        classification[-2:] = 0

        instance_ids = np.array([1] * 8 + [2] * 8 + [0] + [1] * 2 + [2], dtype=np.int64)
        unique_instance_ids = np.array([0, 1, 2], dtype=np.int64)

        canopy_height_model = np.array(
            [[1, 1, 1, 0], [1, 2, 1, 0], [1, 1, 1, 1], [0, 1, 2, 1], [0, 1, 1, 1], [0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float64,
        )

        watershed_labels_with_border = np.array(
            [[2, 2, 2, 0], [2, 2, 2, 0], [2, 2, 0, 0], [0, 0, 3, 3], [0, 3, 3, 3], [0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.int64,
        )

        watershed_labels_without_border = np.array(
            [[2, 2, 2, 0], [2, 2, 2, 0], [2, 2, 3, 0], [0, 3, 3, 3], [0, 3, 3, 3], [0, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.int64,
        )

        expected_seed_mask = np.zeros(len(tree_coords), dtype=bool).flatten()
        expected_seed_mask[-3:] = True

        expected_updated_instance_ids = np.array([-1] * 8 + [-1] * 8 + [0] + [1] * 2 + [2], dtype=np.int64)

        algorithm = MultiStageAlgorithm(
            trunk_class_id=0, crown_class_id=1, branch_class_id=2, max_point_spacing_region_growing=2
        )
        seed_mask, updated_instance_ids = algorithm.determine_overlapping_crowns(
            tree_coords,
            classification,
            instance_ids,
            unique_instance_ids,
            canopy_height_model,
            watershed_labels_with_border,
            watershed_labels_without_border,
        )

        np.testing.assert_array_equal(expected_updated_instance_ids, updated_instance_ids)
        np.testing.assert_array_equal(expected_seed_mask, seed_mask)

    def test_compute_crown_distance_fields(self):
        watershed_labels_without_border = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )

        tree_positions_grid = np.array([[2, 2]], dtype=np.int64)

        expected_distance_fields = np.array(
            [
                [
                    [np.sqrt(2), 1, 1, 1, np.sqrt(2)],
                    [1, -1, -1, -1, 1],
                    [1, -1, -2, -1, 1],
                    [1, -1, -1, -1, 1],
                    [np.sqrt(2), 1, 1, 1, np.sqrt(2)],
                ]
            ],
            dtype=np.float64,
        )

        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1)
        distance_fields = algorithm.compute_crown_distance_fields(watershed_labels_without_border, tree_positions_grid)

        np.testing.assert_array_equal(expected_distance_fields, distance_fields)

    def test_grow_trees_no_seeds(self):
        tree_coords = np.random.randn(20, 3)
        input_instance_ids = np.zeros(len(tree_coords), dtype=np.int64)
        unique_input_instance_ids = np.zeros(1, dtype=np.int64)
        grid_origin = np.array([0, 0], dtype=np.float64)
        crown_distance_fields = np.zeros((0, 0, 0), dtype=np.float64)
        seed_mask = np.zeros(len(tree_coords), dtype=bool)

        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1)
        output_instance_ids = algorithm.grow_trees(
            tree_coords, input_instance_ids, unique_input_instance_ids, grid_origin, crown_distance_fields, seed_mask
        )

        np.testing.assert_array_equal(input_instance_ids, output_instance_ids)

    def test_grow_trees(self):
        tree_coords = np.array(
            [
                [2, 2, 0],
                [2, 2, 0.1],
                [2, 2, 0.2],
                [2, 2, 0.3],
                [2, 2, 1],
                [2, 2, 2],
                [1, 1, 3],
                [1, 2, 3],
                [1, 3, 3],
                [2, 1, 3],
                [2, 2, 3],
                [2, 3, 3],
                [3, 1, 3],
                [3, 2, 3],
                [3, 3, 3],
                [7, 2, 0],
                [7, 2, 0.1],
                [7, 2, 0.2],
                [7, 2, 0.3],
                [7, 2, 1],
                [7, 2, 2],
                [6, 1, 3],
                [6, 2, 3],
                [6, 3, 3],
                [7, 1, 3],
                [7, 2, 3],
                [7, 3, 3],
                [8, 1, 3],
                [8, 2, 3],
                [8, 3, 3],
            ],
            dtype=np.float64,
        )
        instance_ids = np.ones(len(tree_coords), dtype=np.int64) * -1
        instance_ids[0] = 0
        instance_ids[15] = 1
        unique_instance_ids = np.array([0, 1], dtype=np.int64)

        crown_distance_fields = np.array(
            [
                [
                    [np.sqrt(2), 1, 1, 1, np.sqrt(2)],
                    [1, -1, -1, -1, 1],
                    [1, -1, -2, -1, 1],
                    [1, -1, -1, -1, 1],
                    [np.sqrt(2), 1, 1, 1, np.sqrt(2)],
                    [np.sqrt(5), 2, 2, 2, np.sqrt(5)],
                    [np.sqrt(10), 3, 3, 3, np.sqrt(10)],
                    [np.sqrt(15), 4, 4, 4, np.sqrt(15)],
                    [np.sqrt(26), 5, 5, 5, np.sqrt(26)],
                ],
                [
                    [np.sqrt(49), 6, 6, 6, np.sqrt(49)],
                    [np.sqrt(26), 5, 5, 5, np.sqrt(26)],
                    [np.sqrt(15), 4, 4, 4, np.sqrt(15)],
                    [np.sqrt(10), 3, 3, 3, np.sqrt(10)],
                    [np.sqrt(5), 2, 2, 2, np.sqrt(5)],
                    [np.sqrt(2), 1, 1, 1, np.sqrt(2)],
                    [1, -1, -1, -1, 1],
                    [1, -1, -2, -1, 1],
                    [1, -1, -1, -1, 1],
                ],
            ],
            dtype=np.float64,
        )

        seed_mask = np.zeros(len(tree_coords), dtype=bool)
        seed_mask[[0, 15]] = True

        grid_origin = np.array([0, 0], dtype=np.float64)
        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1, grid_size_canopy_height_model=1)
        instance_ids = algorithm.grow_trees(
            tree_coords, instance_ids, unique_instance_ids, grid_origin, crown_distance_fields, seed_mask
        )

        expected_instance_ids = np.array([0] * 15 + [1] * 15, dtype=np.int64)

        np.testing.assert_array_equal(expected_instance_ids, instance_ids)

    def test_upsample_instance_ids(self):
        original_coords = np.array([[0, 0, 0], [0, 1, 0], [0, 10, 0], [1, 10, 0]], dtype=np.float64)
        downsampled_coords = np.array([[0, 1, 0], [0, 10, 0]], dtype=np.float64)
        instance_ids = np.array([1, 0], dtype=np.int64)
        non_predicted_point_indices = np.array([0, 3], dtype=np.int64)
        predicted_point_indices = np.array([1, 2], dtype=np.int64)

        expected_upsampled_instance_ids = np.array([1, 1, 0, 0], dtype=np.int64)

        upsampled_instance_ids = MultiStageAlgorithm.upsample_instance_ids(
            original_coords, downsampled_coords, instance_ids, non_predicted_point_indices, predicted_point_indices
        )

        np.testing.assert_array_equal(expected_upsampled_instance_ids, upsampled_instance_ids)

    @pytest.mark.parametrize("algorithm", ["watershed_crown_top_positions", "watershed_matched_tree_positions", "full"])
    @pytest.mark.parametrize("branch_class_id", [None, 2])
    @pytest.mark.parametrize("downsampling_voxel_size", [None, 0.05])
    def test_algorithm(
        self,
        algorithm: Literal["watershed_crown_top_positions", "watershed_matched_tree_positions", "full"],
        branch_class_id: Optional[int],
        downsampling_voxel_size: Optional[float],
    ):
        num_tree_points = 30
        tree_coords = np.random.rand(num_tree_points, 3)
        classification = np.zeros(num_tree_points, dtype=np.int64)
        point_indices = np.arange(num_tree_points, dtype=np.int64)
        np.random.shuffle(point_indices)
        classification[point_indices[: int(num_tree_points / 2)]] = 1
        if branch_class_id is not None:
            classification[point_indices[int(num_tree_points / 2) : int(num_tree_points * 3 / 4)]] = 2

        point_cloud = pd.DataFrame(
            np.column_stack([tree_coords, classification]), columns=["x", "y", "z", "classification"]
        )

        point_cloud_id = "test"

        algorithm = MultiStageAlgorithm(
            trunk_class_id=0,
            crown_class_id=1,
            branch_class_id=branch_class_id,
            algorithm=algorithm,
            downsampling_voxel_size=downsampling_voxel_size,
        )
        instance_ids = algorithm(point_cloud, point_cloud_id)

        assert len(tree_coords) == len(instance_ids)
        assert len(algorithm.runtime_stats()) > 0

    @pytest.mark.parametrize("algorithm", ["watershed_crown_top_positions", "watershed_matched_tree_positions", "full"])
    @pytest.mark.parametrize("missing_columns", [["x"], ["x", "classification"]])
    def test_invalid_inputs(
        self,
        algorithm: Literal["watershed_crown_top_positions", "watershed_matched_tree_positions", "full"],
        missing_columns: List[str],
    ):
        point_cloud = pd.DataFrame(np.random.rand(20, 4), columns=["x", "y", "z", "classification"])
        point_cloud = point_cloud.drop(missing_columns, axis=1)

        algorithm = MultiStageAlgorithm(trunk_class_id=0, crown_class_id=1, algorithm=algorithm)
        with pytest.raises(ValueError):
            algorithm(point_cloud)

    @pytest.mark.parametrize("algorithm", ["watershed_crown_top_positions", "watershed_matched_tree_positions", "full"])
    def test_no_tree_points(
        self, algorithm: Literal["watershed_crown_top_positions", "watershed_matched_tree_positions", "full"]
    ):
        point_cloud = pd.DataFrame(
            np.column_stack([np.random.rand(20, 3), np.zeros(20, dtype=np.int64)]),
            columns=["x", "y", "z", "classification"],
        )

        algorithm = MultiStageAlgorithm(trunk_class_id=1, crown_class_id=2, algorithm=algorithm)

        instance_ids = algorithm(point_cloud)

        np.testing.assert_array_equal(np.full(len(point_cloud), fill_value=-1, dtype=np.int64), instance_ids)
