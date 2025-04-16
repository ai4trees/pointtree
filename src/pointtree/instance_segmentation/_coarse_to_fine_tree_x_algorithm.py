__all__ = ["CoarseToFineTreeXAlgorithm"]

import os
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from pointtree.evaluation import Profiler
from pointtree.operations import cloth_simulation_filtering, create_digital_terrain_model, distance_to_dtm
from pointtree.visualization import save_tree_map

from pointtree._tree_x_algorithm_cpp import (  # type: ignore[import-not-found] # pylint: disable=import-error, no-name-in-module
    collect_region_growing_seeds as collect_region_growing_seeds_cpp,
)

from ._instance_segmentation_algorithm import InstanceSegmentationAlgorithm
from ._coarse_to_fine_algorithm import CoarseToFineAlgorithm
from ._tree_x_algorithm import TreeXAlgorithm


class CoarseToFineTreeXAlgorithm(InstanceSegmentationAlgorithm):
    def __init__(self):
        super().__init__()
        self._other_class_id = 0
        self._trunk_class_id = 1
        self._crown_class_id = 2
        self._coarse_to_fine_algorithm = CoarseToFineAlgorithm(self._trunk_class_id, self._crown_class_id)
        self._tree_x_algorithm = TreeXAlgorithm()

    def __call__(
        self, xyz: np.ndarray, point_cloud_id: Optional[str] = None
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""
        Runs the tree instance segmentation for the given point cloud.

        Args:
            xyz: 3D coordinates of all points in the point cloud.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. Defaults to
                :code:`None`, which means that no visualizations are created.

        Returns:
            Tree instance labels for all points. For points not belonging to any tree, the label is set to -1.

        Shape:
            - :code:`xyz`: :math:`(N, 3)`
            - Output: :math:`(N)`

            | where
            |
            | :math:`N = \text{ number of points}`
        """

        with Profiler("Terrain filtering", self._performance_tracker):
            self._logger.info("Filter terrain points...")
            terrain_classification = cloth_simulation_filtering(
                xyz,
                classification_threshold=self._tree_x_algorithm._csf_classification_threshold,
                resolution=self._tree_x_algorithm._csf_resolution,
                rigidness=self._tree_x_algorithm._csf_rigidness,
                correct_steep_slope=self._tree_x_algorithm._csf_correct_steep_slope,
                iterations=self._tree_x_algorithm._csf_iterations,
            )
            is_tree = terrain_classification != 0
            del terrain_classification
            tree_xyz = xyz[is_tree]

        classification = np.full(len(tree_xyz), fill_value=self._crown_class_id, dtype=np.int64)
        (
            crown_top_positions,
            crown_top_positions_grid,
            canopy_height_model,
            grid_origin,
        ) = self._coarse_to_fine_algorithm.compute_crown_top_positions(tree_xyz, classification)

        if self._coarse_to_fine_algorithm._algorithm == "watershed_crown_top_positions":
            if self._visualization_folder is not None and point_cloud_id is not None:
                save_tree_map(
                    canopy_height_model,
                    os.path.join(
                        self._tree_x_algorithm._visualization_folder, f"canopy_height_model_{point_cloud_id}.png"
                    ),
                    tree_positions=crown_top_positions_grid,
                )

            (
                instance_ids,
                unique_instance_ids,
                watershed_labels_with_border,
                watershed_labels_without_border,
            ) = self.coarse_segmentation(
                xyz[is_tree],
                np.full(is_tree.sum(), fill_value=-1, dtype=np.int64),
                crown_top_positions_grid,
                canopy_height_model,
                grid_origin,
                point_cloud_id=point_cloud_id,
            )

            return instance_ids, unique_instance_ids

        with Profiler("Construction of digital terrain model", self._performance_tracker):
            self._logger.info("Construct digital terrain model...")
            dtm, dtm_offset = create_digital_terrain_model(
                xyz[np.logical_not(is_tree)],
                grid_resolution=self._tree_x_algorithm._dtm_resolution,
                k=self._tree_x_algorithm._dtm_k,
                p=self._tree_x_algorithm._dtm_p,
                voxel_size=self._tree_x_algorithm._dtm_voxel_size,
                num_workers=self._tree_x_algorithm._num_workers,
            )

        with Profiler("Height normalization", self._performance_tracker):
            self._logger.info("Normalize point heights...")
            dists_to_dtm = distance_to_dtm(xyz, dtm, dtm_offset, self._tree_x_algorithm._dtm_resolution)
            del dtm
            del dtm_offset

        with Profiler("Trunk identification", self._performance_tracker):
            self._logger.info("Identify trunks...")
            trunk_layer_filter = np.logical_and(
                dists_to_dtm >= self._tree_x_algorithm._trunk_search_min_z,
                dists_to_dtm < self._tree_x_algorithm._trunk_search_max_z,
            )
            trunk_layer_filter = np.logical_and(is_tree, trunk_layer_filter)
            trunk_layer_xyz = np.empty((trunk_layer_filter.sum(), 3), dtype=xyz.dtype)
            trunk_layer_xyz[:, :2] = xyz[trunk_layer_filter, :2]
            trunk_layer_xyz[:, 2] = dists_to_dtm[trunk_layer_filter]
            del trunk_layer_filter
            trunk_positions, trunk_diameters = self._tree_x_algorithm.find_trunks(
                trunk_layer_xyz, point_cloud_id=point_cloud_id
            )
            del trunk_layer_xyz

        if len(trunk_positions) == 0:
            return (
                np.full(len(xyz), fill_value=-1, dtype=np.int64),
                trunk_positions,
                trunk_diameters,
            )

        print("trunk_positions", trunk_positions)
        print("crown_top_positions", crown_top_positions)

        tree_positions = self._coarse_to_fine_algorithm.match_trunk_and_crown_tops(
            trunk_positions,
            crown_top_positions,
        )

        # convert tree coordinates to grid indices for height map
        tree_positions_grid = np.copy(tree_positions)
        tree_positions_grid -= grid_origin
        tree_positions_grid /= self._coarse_to_fine_algorithm._grid_size_canopy_height_model
        tree_positions_grid = tree_positions_grid.astype(int)

        if self._coarse_to_fine_algorithm._visualization_folder is not None and point_cloud_id is not None:
            save_tree_map(
                canopy_height_model,
                os.path.join(
                    self._coarse_to_fine_algorithm._visualization_folder,
                    f"canopy_height_model_{point_cloud_id}_without_positions.png",
                ),
            )
            save_tree_map(
                canopy_height_model,
                os.path.join(
                    self._coarse_to_fine_algorithm._visualization_folder, f"canopy_height_model_{point_cloud_id}.png"
                ),
                tree_positions=tree_positions_grid[: len(crown_top_positions)],
                trunk_positions=tree_positions_grid[len(crown_top_positions) :],
            )

        instance_ids, _ = collect_region_growing_seeds_cpp(
            tree_xyz,
            dists_to_dtm[is_tree],
            tree_positions,
            trunk_diameters,
            self._tree_x_algorithm._region_growing_seed_layer_height,
            self._tree_x_algorithm.region_growing_seed_radius_factor,
            num_workers=self._tree_x_algorithm._num_workers,
        )

        (
            instance_ids,
            unique_instance_ids,
            watershed_labels_with_border,
            watershed_labels_without_border,
        ) = self._coarse_to_fine_algorithm.coarse_segmentation(
            tree_xyz,
            instance_ids,
            tree_positions_grid,
            canopy_height_model,
            grid_origin,
            point_cloud_id=point_cloud_id,
            trunk_positions_grid=tree_positions_grid[len(crown_top_positions) :],
        )

        if self._coarse_to_fine_algorithm._algorithm == "watershed_matched_tree_positions":
            return instance_ids, unique_instance_ids

        seed_mask, instance_ids = self._coarse_to_fine_algorithm.determine_overlapping_crowns(
            tree_xyz,
            classification,
            instance_ids,
            unique_instance_ids,
            canopy_height_model,
            watershed_labels_with_border,
            watershed_labels_without_border,
        )

        instance_ids_to_refine = np.sort(np.unique(instance_ids[seed_mask]))

        crown_distance_fields = self._coarse_to_fine_algorithm.compute_crown_distance_fields(
            watershed_labels_without_border, tree_positions_grid[instance_ids_to_refine]
        )

        instance_ids = self._coarse_to_fine_algorithm.grow_trees(
            tree_xyz, instance_ids, unique_instance_ids, grid_origin, crown_distance_fields, seed_mask
        )

        self._logger.info("Finished segmentation.")

        print(self.runtime_stats())

        return instance_ids, trunk_positions, trunk_diameters
