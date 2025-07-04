"""Coarse-to-fine algorithm algorithm for tree instance segmentation."""  # pylint: disable=too-many-lines

__all__ = ["CoarseToFineAlgorithm"]

import os
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, cast

from scipy.spatial import KDTree
import numpy as np
import numpy.typing as npt
from pointtorch.operations.numpy import make_labels_consecutive, voxel_downsampling
import scipy.ndimage as ndi
from skimage.morphology import diamond, disk, footprint_rectangle, dilation, erosion
from skimage.feature import peak_local_max  # pylint: disable=no-name-in-module
from skimage.segmentation import watershed, find_boundaries
from sklearn.cluster import DBSCAN
import torch
from torch_scatter import scatter_max

from pointtree.type_aliases import BoolArray, FloatArray, LongArray
from pointtree._coarse_to_fine_algorithm_cpp import (  # type: ignore[import-untyped] # pylint: disable=import-error, no-name-in-module
    grow_trees as grow_trees_cpp,
)
from pointtree.evaluation import Profiler
from pointtree.visualization import save_tree_map
from .filters import filter_instances_min_points
from ._instance_segmentation_algorithm import InstanceSegmentationAlgorithm


class CoarseToFineAlgorithm(InstanceSegmentationAlgorithm):  # pylint: disable=too-many-instance-attributes
    r"""
    Coarse-to-fine algorithm for tree instance segmentation proposed in `Burmeister, Josafat-Mattias, et al. "Tree \
    Instance Segmentation in Urban 3D Point Clouds Using a Coarse-to-Fine Algorithm Based on Semantic Segmentation." \
    ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences 10 (2024): 79-86. \
    <https://doi.org/10.5194/isprs-annals-X-4-W5-2024-79-2024>`__.

    Args:
        trunk_class_id: Integer class ID that designates the tree trunk points.
        crown_class_id: Integer class ID that designates the tree crown points.
        branch_class_id: Integer class ID that designates the tree branch points. If set to :code:`None`, it is assumed
            that branch points are not separately labeled. If branches are separately labeled, the branch points are
            treated as trunk points by this algorithm.
        algorithm: Variant of the algorithm to be used: :code:`"full"`: The full algorithm is used.
            :code:`"watershed_crown_top_positions"`: A marker-controlled Watershed segmentation is performed, using the
            crown top positions as markers. :code:`"watershed_matched_tree_positions"`: A marker-controlled Watershed
            segmentation is performed, using the tree positions as markers, resulting from the matching of crown top
            positions and trunk positions.

        downsampling_voxel_size: Voxel size for the voxel-based downsampling of the tree points before performing the
            tree instance segmentation. If set to :code:`None`, the tree instance segmentation is performed with the
            full resolution of the point cloud.
        visualization_folder: Path of a directory in which to store visualizations of intermediate results of the
            algorithm. If set to :code:`None`, no visualizations are stored.
        num_workers: Number of workers to use for parallel processing. If :code:`workers` is set to -1, all CPU threads
            are used.
            
    Parameters for the DBSCAN clustering of trunk points:
        eps_trunk_clustering: Parameter :math:`\epsilon` for clustering the trunk points using the DBSCAN
            algorithm. Further details on the meaning of the parameter can be found in the documentation of
            `sklearn.cluster.DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`__.
        min_samples_trunk_clustering: Parameter :math:`MinSamples` for clustering the trunk points using the
            DBSCAN algorithm. Further details on the meaning of the parameter can be found in the documentation of
            `sklearn.cluster.DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`__.
        min_trunk_points: Minimum number of points a cluster of trunk points must contain to be considered as a trunk.

    Parameters for the construction and maximum filtering of the canopy height model:
        grid_size_canopy_height_model: Width of the 2D grid cells used to create the canopy height model (in meters).
        min_distance_crown_tops: Minimum horizontal distance that local maxima of the canopy height model must have in
            order to be considered as separate crown tops (in meters).
        min_points_crown_detection: Minimum number of points that must be contained in a cell of the canopy height model
            for the cell to be considered a possible crown top.
        min_tree_height: Minimum height that a local maximum of the canopy height model must have to be considered as a
            crown top (in meters).
        smooth_canopy_height_model: Whether to smooth the canopy height model using a Gaussian blur filter.
        smooth_sigma: Parameter :math:`\sigma` of the Gaussian blur filter used to smooth the canopy height model.

    Parameters for the matching of trunk positions and crown top positions:
        distance_match_trunk_and_crown_top: Maximum horizontal distance between trunk positions and crown top positions
            up to which both are considered to belong to the same tree.

    Parameters for the Watershed segmentation:
        correct_watershed: Whether erroneous parts of the watershed segmentation should be replaced by a Voronoi
            segmentation.

    Parameters for the region growing segmentation:
        max_point_spacing_region_growing: The results of the Watershed segmentation are only refined in sufficiently
            dense point cloud regions. For this purpose, the average distance of the points to their nearest neighbor is
            determined for each tree. If this average distance is less than :code:`max_point_spacing_region_growing` the
            tree is considered for refining its segmentation using the region growing approach (in meters).
        max_radius_region_growing: Maximum radius in which to search for neighboring points during region growing (in
            meters).
        multiplier_outside_coarse_border: In our region growing approach, the points are processed in a sorted order,
            with points with the smallest distance to an already assigned tree point being processed first. For points
            that lie outside the crown boundary, the distance is multiplied by a constant factor to restrict growth in
            areas outside the crown boundary. This parameter defines this constant factor. The larger the factor, the
            more growth is restricted in areas outside the crown boundaries.
        num_neighbors_region_growing: In our region growing approach, the k-nearest neighbors of each seed point a are
            searched and may be added to the same tree instance. This parameter specifies the number of neighbors to
            search.
        z_scale: Factor by which the z-coordinates are multiplied in region growing. Using a value between zero and one
            favors upward growth.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        trunk_class_id: int,
        crown_class_id: int,
        *,
        branch_class_id: Optional[int] = None,
        algorithm: Literal["full", "watershed_crown_top_positions", "watershed_matched_tree_positions"] = "full",
        downsampling_voxel_size: Optional[float] = None,
        visualization_folder: Optional[Union[str, Path]] = None,
        num_workers: int = -1,
        eps_trunk_clustering: float = 2.5,
        min_samples_trunk_clustering: int = 1,
        min_trunk_points: int = 100,
        grid_size_canopy_height_model: float = 0.5,
        min_distance_crown_tops: float = 7,
        min_points_crown_detection: float = 100,
        min_tree_height: float = 2.5,
        smooth_canopy_height_model: bool = True,
        smooth_sigma: float = 1,
        distance_match_trunk_and_crown_top: float = 5,
        correct_watershed: bool = True,
        max_point_spacing_region_growing: float = 0.08,
        max_radius_region_growing: float = 1,
        multiplier_outside_coarse_border: float = 2,
        num_neighbors_region_growing: int = 27,
        z_scale: float = 0.5,
    ):
        super().__init__()

        self._algorithm = algorithm

        if visualization_folder is None or isinstance(visualization_folder, Path):
            self._visualization_folder = visualization_folder
        else:
            self._visualization_folder = Path(visualization_folder)

        self._trunk_class_id = trunk_class_id
        self._crown_class_id = crown_class_id
        self._branch_class_id = branch_class_id
        self._downsampling_voxel_size = downsampling_voxel_size
        self._num_workers = num_workers

        # Parameters for the DBSCAN clustering of trunk points
        self._eps_trunk_clustering = eps_trunk_clustering
        self._min_samples_trunk_clustering = min_samples_trunk_clustering
        self._min_trunk_points = min_trunk_points

        # Parameters for the construction and maximum filtering of the canopy height model
        self._grid_size_canopy_height_model = grid_size_canopy_height_model
        self._min_distance_crown_tops = min_distance_crown_tops
        self._min_points_crown_detection = min_points_crown_detection
        self._min_tree_height = min_tree_height
        self._smooth_canopy_height_model = smooth_canopy_height_model
        self._smooth_sigma = smooth_sigma

        # Parameters for the matching of trunk positions and crown top positions
        self._distance_match_trunk_and_crown_top = distance_match_trunk_and_crown_top

        # Parameters for the Watershed segmentation
        self._correct_watershed = correct_watershed

        # Parameters for the region growing segmentation:
        self._max_point_spacing_region_growing = max_point_spacing_region_growing
        self._max_radius_region_growing = max_radius_region_growing
        self._num_neighbors_region_growing = num_neighbors_region_growing
        self._multiplier_outside_coarse_border = multiplier_outside_coarse_border
        self._z_scale = z_scale

    def _segment_tree_points(  # pylint: disable=too-many-locals
        self, tree_xyz: FloatArray, classification: LongArray, point_cloud_id: Optional[str] = None
    ) -> Tuple[LongArray, LongArray]:
        r"""
        Multi-stage tree instance segmentation algorithm.

        Args:
            tree_xyz: Coordinates of all tree points.
            classification: Class IDs of each tree point.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. If set to
                :code:`None`, no visualizations are created.

        Returns:
            :Tuple of two arrays:
                - Instance IDs of each point. Points that do not belong to any instance are assigned the ID :code:`-1`.
                - Unique instance IDs.

        Shape:
            - :code:`tree_xyz`: :math:`(N, 3)`
            - :code:`classification`: :math:`(N)`
            - Output: :math:`(N)`, :math:`(I)`

            | where
            |
            | :math:`N = \text{ number of tree points}`
            | :math:`I = \text{ number of tree instances}`
        """

        (
            crown_top_positions,
            crown_top_positions_grid,
            canopy_height_model,
            grid_origin,
        ) = self.compute_crown_top_positions(tree_xyz, classification)

        if self._algorithm == "watershed_crown_top_positions":
            if self._visualization_folder is not None and point_cloud_id is not None:
                save_tree_map(
                    canopy_height_model,
                    os.path.join(self._visualization_folder, f"canopy_height_model_{point_cloud_id}.png"),
                    tree_positions=crown_top_positions_grid,
                )

            (
                instance_ids,
                unique_instance_ids,
                watershed_labels_with_border,
                watershed_labels_without_border,
            ) = self.coarse_segmentation(
                tree_xyz,
                np.full(len(tree_xyz), fill_value=-1, dtype=np.int64),
                crown_top_positions_grid,
                canopy_height_model,
                grid_origin,
                point_cloud_id=point_cloud_id,
            )

            return instance_ids, unique_instance_ids

        instance_ids, unique_instance_ids = self.cluster_trunks(tree_xyz, classification)
        with Profiler("Filtering of trunk clusters", self._performance_tracker):
            self._logger.info("Filter trunk clusters...")
            instance_ids, unique_instance_ids = filter_instances_min_points(
                instance_ids, unique_instance_ids, self._min_trunk_points
            )
        trunk_positions = self.compute_trunk_positions(tree_xyz, instance_ids, unique_instance_ids)

        tree_positions = self.match_trunk_and_crown_tops(
            trunk_positions,
            crown_top_positions,
        )

        # convert tree coordinates to grid indices for height map
        tree_positions_grid = ((tree_positions - grid_origin) / self._grid_size_canopy_height_model).astype(np.int64)

        if self._visualization_folder is not None and point_cloud_id is not None:
            save_tree_map(
                canopy_height_model,
                os.path.join(self._visualization_folder, f"canopy_height_model_{point_cloud_id}_without_positions.png"),
            )
            save_tree_map(
                canopy_height_model,
                os.path.join(self._visualization_folder, f"canopy_height_model_{point_cloud_id}.png"),
                tree_positions=tree_positions_grid[: len(crown_top_positions)],
                trunk_positions=tree_positions_grid[len(crown_top_positions) :],
            )

        (
            instance_ids,
            unique_instance_ids,
            watershed_labels_with_border,
            watershed_labels_without_border,
        ) = self.coarse_segmentation(
            tree_xyz,
            instance_ids,
            tree_positions_grid,
            canopy_height_model,
            grid_origin,
            point_cloud_id=point_cloud_id,
            trunk_positions_grid=tree_positions_grid[len(crown_top_positions) :],
        )

        if self._algorithm == "watershed_matched_tree_positions":
            return instance_ids, unique_instance_ids

        seed_mask, instance_ids = self.determine_overlapping_crowns(
            tree_xyz,
            classification,
            instance_ids,
            unique_instance_ids,
            canopy_height_model,
            watershed_labels_with_border,
            watershed_labels_without_border,
        )

        instance_ids_to_refine = np.sort(np.unique(instance_ids[seed_mask]))

        crown_distance_fields = self.compute_crown_distance_fields(
            watershed_labels_without_border, tree_positions_grid[instance_ids_to_refine]
        )

        instance_ids = self.grow_trees(
            tree_xyz, instance_ids, unique_instance_ids, grid_origin, crown_distance_fields, seed_mask
        )

        return instance_ids, unique_instance_ids

    def cluster_trunks(self, tree_xyz: FloatArray, classification: LongArray) -> Tuple[LongArray, LongArray]:
        r"""
        Clusters tree trunk points using the
        `DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`__ algorithm and assigns
        the same unique instance ID to the points belonging to the same cluster.

        Args:
            tree_xyz: Coordinates of all tree points.
            classification: Class ID of all tree points.

        Returns:
            :Tuple of two arrays:
                - Instance ID of each tree point. Points that do not belong to any instance are assigned the ID
                  :code:`-1`.
                - Unique instance IDs.

        Shape:
            - :code:`tree_xyz`: :math:`(N, 3)`
            - :code:`classification`: :math:`(N)`
            - Output: :math:`(N)`, :math:`(T)`

            | where
            |
            | :math:`N = \text{ number of tree points}`
            | :math:`T = \text{ number of trunk instances}`
        """

        with Profiler("Clustering of trunk points", self._performance_tracker):
            self._logger.info("Cluster trunk points...")
            trunk_mask = classification == self._trunk_class_id
            if self._branch_class_id is not None:
                trunk_mask = np.logical_or(trunk_mask, classification == self._branch_class_id)

            if trunk_mask.sum() == 0:
                return np.full(len(tree_xyz), dtype=np.int64, fill_value=-1), np.empty(0, dtype=np.int64)

            trunk_xyz = tree_xyz[trunk_mask]

            dbscan = DBSCAN(eps=self._eps_trunk_clustering, min_samples=self._min_samples_trunk_clustering)
            clustering = dbscan.fit(trunk_xyz)

            instance_ids = np.full(len(tree_xyz), fill_value=-1, dtype=np.int64)
            instance_ids[trunk_mask] = clustering.labels_

            return make_labels_consecutive(  # type: ignore[return-value]
                instance_ids, ignore_id=-1, inplace=True, return_unique_labels=True
            )

    def compute_trunk_positions(
        self, tree_xyz: FloatArray, instance_ids: LongArray, unique_instance_ids: LongArray
    ) -> FloatArray:
        r"""
        Computes the 2D position of each trunk.

        Args:
            tree_xyz: Coordinates of all tree points.
            instance_ids: Instance IDs of each tree point.
            unique_instance_ids: Unique instance IDs.
            min_points: Minimum number of points an instance must have to not be discarded.

        Returns:
            2D position of each trunk.

        Shape:
            - :code:`tree_xyz`: :math:`(N, 3)`
            - :code:`instance_ids`: :math:`(N)`
            - :code:`unique_instance_ids`: :math:`(T)`
            - Output: :math:`(T, 2)`

            | where
            |
            | :math:`N = \text{ number of tree points}`
            | :math:`T = \text{ number of trunk instances}`
        """
        with Profiler("Computation of trunk positions", self._performance_tracker):
            self._logger.info("Detect trunk positions...")
            trunk_positions = np.empty((len(unique_instance_ids), 2))

            for instance_id in unique_instance_ids:
                instance_points = tree_xyz[instance_ids == instance_id]
                trunk_positions[instance_id] = instance_points[:, :2].mean(axis=0)

        return trunk_positions

    @staticmethod
    def create_height_map(  # pylint: disable=too-many-locals
        xyz: FloatArray,
        grid_size: float,
        bounding_box: Optional[FloatArray] = None,
    ) -> Tuple[FloatArray, LongArray, FloatArray]:
        r"""
        Creates a 2D height map from a given point cloud. For this purpose, the 3D point cloud is projected onto a 2D
        grid and the maximum z-coordinate within each grid cell is recorded. The value of grid cells that do not contain
        any point is set to zero. Before creating the height map, the point cloud is normalized by subtracting the
        minimum z-coordinate.

        Args:
            xyz: Coordinates of each point.
            grid_size: Grid size of the height map.
            bounding_box: Bounding box coordinates specifying the area for which to compute the height map. The first
                element is expected to be the minimum xy-coordinate of the bounding box and the second the maximum
                xy-coordinate.

        Returns:
            :Tuple of three arrays:
                - Height map.
                - A map of the same size containing the number of points within each grid cell.
                - Position of the grid origin.

        Shape:
            - :code:`xyz`: :math:`(N, 3)`
            - :code:`bounding_box`: :math:`(2, 2)`
            - Output: :math:`(W, H)`, :math:`(W, H)`, :math:`(2)`.

            | where
            |
            | :math:`N = \text{ number of points}`
            | :math:`W = \text{ extent of the height map in x-direction}`
            | :math:`H = \text{ extent of the height map in y-direction}`
        """

        if len(xyz) == 0:
            return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)

        if bounding_box is None:
            bounding_box = np.row_stack([xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)])

        xyz = xyz.copy()
        xyz = xyz[(xyz[:, :2] <= bounding_box[1]).all(axis=-1)]
        xyz[:, :2] -= bounding_box[0]
        xyz[:, 2] -= xyz[:, 2].min(axis=0)

        device = torch.device("cpu")

        # check if there is a GPU with sufficient memory to run the processing on GPU
        if torch.cuda.is_available():
            available_memory = torch.cuda.mem_get_info(device=torch.device("cuda:0"))[0]
            float_size = torch.empty((0,)).float().element_size()
            long_size = torch.empty((0,)).long().element_size()
            approx_required_memory = len(xyz) * (4 * float_size + 6 * long_size)

            if available_memory > approx_required_memory:
                device = torch.device("cuda:0")

        xyz_torch = torch.from_numpy(xyz).to(device)
        grid_indices = torch.floor(xyz_torch[:, :2] / grid_size).long()
        first_cell = np.floor(bounding_box[0] / grid_size).astype(np.int64)
        last_cell = np.floor(bounding_box[1] / grid_size).astype(np.int64)

        num_cells = last_cell - first_cell + 1

        unique_grid_indices, inverse_indices, point_counts_per_grid_cell = torch.unique(
            grid_indices, sorted=True, return_counts=True, return_inverse=True, dim=0
        )
        del grid_indices
        inverse_indices, sorting_indices = torch.sort(inverse_indices)

        max_height, _ = scatter_max(xyz_torch[sorting_indices, 2], inverse_indices, dim=0)

        unique_grid_indices_np = unique_grid_indices.cpu().numpy()
        del unique_grid_indices

        height_map = np.zeros(num_cells)
        height_map[unique_grid_indices_np[:, 0], unique_grid_indices_np[:, 1]] = max_height.cpu().numpy()
        point_counts = np.zeros(num_cells, dtype=np.int64)
        point_counts[unique_grid_indices_np[:, 0], unique_grid_indices_np[:, 1]] = (
            point_counts_per_grid_cell.cpu().numpy()
        )

        return height_map, point_counts, first_cell * grid_size

    def compute_crown_top_positions(
        self, tree_xyz: FloatArray, classification: LongArray
    ) -> Tuple[FloatArray, LongArray, FloatArray, FloatArray]:
        r"""
        Constructs a 2D canopy height model, identifies the local maxima corresponding to the crown tops and calculates
        their 2D position.

        Args:
            tree_xyz: Coordinates of all tree points.
            classification: Class IDs of each tree point.

        Returns:
            :Tuple of four arrays:
                - 2D position of each crown top as floating point coordinates.
                - Tree positions as integer coordinates in the grid coordinate system of the canopy height model.
                - Canopy height model
                - Position of the grid origin.

        Shape:
            - :code:`tree_xyz`: :math:`(N, 3)`
            - :code:`classification`: :math:`(N)`
            - Output: :math:`(C, 2)`, :math:`(C, 2)`, :math:`(W, H)`, :math:`(2)`.

            | where
            |
            | :math:`N = \text{ number of tree points}`
            | :math:`C = \text{ number of crown instances}`
            | :math:`W = \text{ extent of the canopy height model in x-direction}`
            | :math:`H = \text{ extent of the canopy height model in y-direction}`
        """
        with Profiler("Detection of crown top positions", self._performance_tracker):
            if len(tree_xyz) == 0:
                self._logger.info("Detect crown positions...")
                return (
                    np.empty((0, 0), dtype=np.float64),
                    np.empty((0, 0), dtype=np.int64),
                    np.empty((0, 0), dtype=np.float64),
                    np.empty((0, 0), dtype=np.float64),
                )

        with Profiler("Height map computation", self._performance_tracker):
            bounding_box = None
            if self._algorithm != "watershed_crown_top_positions":
                bounding_box = np.row_stack([tree_xyz[:, :2].min(axis=0), tree_xyz[:, :2].max(axis=0)])
                tree_xyz = tree_xyz[classification == self._crown_class_id]
            canopy_height_model, count_map, grid_origin = self.create_height_map(
                tree_xyz, grid_size=self._grid_size_canopy_height_model, bounding_box=bounding_box
            )

        with Profiler("Detection of crown top positions", self._performance_tracker):
            self._logger.info("Detect crown positions...")
            min_distance = int(self._min_distance_crown_tops / self._grid_size_canopy_height_model)
            footprint_size = int(self._min_distance_crown_tops / self._grid_size_canopy_height_model * 0.5)
            footprint = disk(footprint_size)

            if self._smooth_canopy_height_model:
                smoothed_canopy_height_model = cast(
                    FloatArray, ndi.gaussian_filter(canopy_height_model, sigma=self._smooth_sigma)
                )
                weights = ndi.gaussian_filter((canopy_height_model > 0).astype(float), sigma=self._smooth_sigma)
                weights[weights == 0] = 1
                smoothed_canopy_height_model = smoothed_canopy_height_model / weights.astype(  # type:ignore[assignment]
                    smoothed_canopy_height_model.dtype
                )
                smoothed_canopy_height_model[canopy_height_model == 0] = 0
                canopy_height_model = smoothed_canopy_height_model

            filtered_canopy_height_model = canopy_height_model.copy()
            filtered_canopy_height_model[count_map < self._min_points_crown_detection] = 0

            crown_top_positions_grid = peak_local_max(
                filtered_canopy_height_model,
                exclude_border=False,
                min_distance=min_distance,
                threshold_abs=self._min_tree_height,
                footprint=footprint,
            )

            # map grid coordinates to point coordinates
            crown_top_positions = crown_top_positions_grid.astype(float)
            crown_top_positions = crown_top_positions * self._grid_size_canopy_height_model + grid_origin

        return crown_top_positions, crown_top_positions_grid, canopy_height_model, grid_origin

    def match_trunk_and_crown_tops(self, trunk_positions: FloatArray, crown_top_positions: FloatArray) -> FloatArray:
        r"""
        Merges trunk and crown topm positions corresponding to the same tree.

        Args:
            trunk_positions: 2D position of each trunk.
            crown_top_positions: 2D position of each crown top.

        Returns:
            Merged tree positions.

        Shape:
            - :code:`trunk_positions`: :math:`(T, 2)`
            - :code:`crown_top_positions`: :math:`(C, 2)`
            - Output: :math:`(I, 2)`

            | where
            |
            | :math:`T = \text{ number of trunk instances}`
            | :math:`C = \text{ number of crown instances}`
            | :math:`I = \text{ number of merged tree instances}`
        """
        with Profiler("Position matching", self._performance_tracker):
            self._logger.info("Match trunk and crown top positions...")
            if len(trunk_positions) == 0:
                return crown_top_positions
            if len(crown_top_positions) == 0:
                return trunk_positions

            crown_distances, crown_indices = KDTree(crown_top_positions).query(trunk_positions, k=1)
            crown_distances = cast(np.ndarray, crown_distances).flatten()
            crown_indices = cast(np.ndarray, crown_indices).flatten()

            trunk_distances, trunk_indices = KDTree(trunk_positions).query(crown_top_positions, k=1)
            trunk_distances = cast(npt.NDArray, trunk_distances).flatten()
            trunk_indices = cast(npt.NDArray, trunk_indices).flatten()

            tree_positions = []
            matched_crown_positions = []
            for idx, trunk_position in enumerate(trunk_positions):
                crown_idx = crown_indices[idx]
                if crown_distances[idx] <= self._distance_match_trunk_and_crown_top and trunk_indices[crown_idx] == idx:
                    crown_index = crown_indices[idx]
                    tree_positions.append(crown_top_positions[crown_index])
                    matched_crown_positions.append(crown_index)
                else:
                    self._logger.info("Trunk %d cannot be matched to any crown.", idx)
                    tree_positions.append(trunk_position)
            non_matched_crown_indices = np.setdiff1d(np.arange(len(crown_top_positions)), matched_crown_positions)
            tree_positions.extend(crown_top_positions[non_matched_crown_indices])

        return np.array(tree_positions)

    def coarse_segmentation(  # pylint: disable=too-many-locals
        self,
        tree_xyz: FloatArray,
        instance_ids: LongArray,
        tree_positions_grid: LongArray,
        canopy_height_model: FloatArray,
        grid_origin: FloatArray,
        *,
        point_cloud_id: Optional[str] = None,
        trunk_positions_grid: Optional[LongArray] = None,
    ) -> Tuple[LongArray, LongArray, LongArray, LongArray]:
        r"""
        Coarse tree instance segmentation using the marker-controlled Watershed algorithm.

        Args:
            tree_xyz: Coordinates of all tree points.
            instance_ids: Instance IDs of each tree point.
            tree_positions_grid: The 2D positions of all tree instances as in integer coordinates in the grid coordinate
                system of the canopy height model.
            canopy_height_model: The canopy height model to segment.
            grid_origin: 2D coordinates of the origin of the canopy height model.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. If set to
                :code:`None`, no visualizations are created.
            trunk_positions_grid: The 2D positions of tree trunks that were not matched with a tree crown as in integer
                coordinates in the grid coordinate system of the canopy height model. Only used for visualization
                purposes.

        Returns:
            :Tuple of four arrays:
                - Instance ID of each tree point. Points that do not belong to any instance are assigned the ID
                  :code:`-1`.
                - Unique instance IDs.
                - 2D segmentation mask with the pixels at the boundary lines between neighboring trees are assigned the
                  background value of 0.
                - Segmentation mask without boundary lines.

        Shape:
            - :code:`tree_xyz`: :math:`(N, 3)`
            - :code:`instance_ids`: :math:`(N)`
            - :code:`tree_positions_grid`: :math:`(N, 2)`
            - :code:`canopy_height_model`: :math:`(W, H)`
            - :code:`grid_origin`: :math:`(2)`
            - Output: :math:`(N)`, :math:`(I)`, :math:`(W, H)`, :math:`(W, H)`

            | where
            |
            | :math:`N = \text{ number of tree points}`
            | :math:`I = \text{ number of tree instances}`
            | :math:`W = \text{ extent of the canopy height model in x-direction}`
            | :math:`H = \text{ extent of the canopy height model in y-direction}`
        """

        with Profiler("Coarse segmentation", self._performance_tracker):
            self._logger.info("Coarse segmentation...")

            foreground_mask = canopy_height_model > 0

            watershed_labels_with_border, watershed_labels_without_border = self.watershed_segmentation(
                canopy_height_model, tree_positions_grid
            )

            crown_positions_grid = None
            if trunk_positions_grid is not None:
                crown_positions_grid = tree_positions_grid[: len(tree_positions_grid) - len(trunk_positions_grid)]

            if self._visualization_folder is not None and point_cloud_id is not None:
                crown_borders = np.logical_and(watershed_labels_with_border == 0, foreground_mask)
                crown_borders = np.logical_or(crown_borders, watershed_labels_with_border == 11)

                save_tree_map(
                    watershed_labels_with_border,
                    output_path=f"{self._visualization_folder}/watershed_with_border_{point_cloud_id}.png",
                    is_label_image=True,
                    crown_borders=crown_borders,
                    tree_positions=tree_positions_grid if trunk_positions_grid is None else None,
                    crown_positions=crown_positions_grid,
                    trunk_positions=trunk_positions_grid,
                )
                save_tree_map(
                    watershed_labels_without_border,
                    output_path=f"{self._visualization_folder}/watershed_without_border_{point_cloud_id}.png",
                    is_label_image=True,
                    tree_positions=tree_positions_grid if trunk_positions_grid is None else None,
                    crown_positions=crown_positions_grid,
                    trunk_positions=trunk_positions_grid,
                )

            if self._correct_watershed:
                watershed_labels_with_border, watershed_labels_without_border = self.watershed_correction(
                    watershed_labels_with_border,
                    watershed_labels_without_border,
                    tree_positions_grid,
                    point_cloud_id=point_cloud_id,
                )

                if self._visualization_folder is not None and point_cloud_id is not None:
                    crown_borders = np.logical_and(watershed_labels_with_border == 0, foreground_mask)
                    img_path = f"{self._visualization_folder}/watershed_labels_voronoi_with_border_{point_cloud_id}.png"
                    save_tree_map(
                        watershed_labels_with_border,
                        output_path=img_path,
                        is_label_image=True,
                        crown_borders=crown_borders,
                        tree_positions=tree_positions_grid if trunk_positions_grid is None else None,
                        crown_positions=crown_positions_grid,
                        trunk_positions=trunk_positions_grid,
                    )
                    img_path = (
                        f"{self._visualization_folder}/watershed_labels_voronoi_without_border_{point_cloud_id}.png"
                    )
                    save_tree_map(
                        watershed_labels_without_border,
                        output_path=img_path,
                        is_label_image=True,
                        tree_positions=tree_positions_grid if trunk_positions_grid is None else None,
                        crown_positions=crown_positions_grid,
                        trunk_positions=trunk_positions_grid,
                    )

            grid_indices = np.floor((tree_xyz[:, :2] - grid_origin) / self._grid_size_canopy_height_model).astype(
                np.int64
            )
            mask = np.logical_and(
                instance_ids == -1, (grid_indices < watershed_labels_without_border.shape).all(axis=1)
            )
            grid_indices = grid_indices[mask]
            instance_ids[mask] = watershed_labels_without_border[grid_indices[:, 0], grid_indices[:, 1]] - 1
            instance_ids, unique_instance_ids = make_labels_consecutive(
                instance_ids, ignore_id=-1, inplace=True, return_unique_labels=True
            )

        return (
            instance_ids,
            cast(LongArray, unique_instance_ids),
            watershed_labels_with_border,
            watershed_labels_without_border,
        )

    def watershed_segmentation(
        self, canopy_height_model: FloatArray, tree_positions_grid: LongArray
    ) -> Tuple[LongArray, LongArray]:
        r"""
        Marker-controlled Watershed segmentation of the canopy height model.

        Args:
            canopy_height_model: The canopy height model to segment.
            tree_positions_grid: Indices of the grid cells of the canopy height model in which the tree positions are
                located.

        Returns:
            :Tuple of two arrays:
                - Watershed segmentation mask with the pixels at the boundary lines between neighboring trees are
                  assigned the background value of 0.
                - Watershed segmentation mask without boundary lines.

        Shape:
            - :code:`canopy_height_model`: :math:`(W, H)`
            - :code:`tree_positions_grid`: :math:`(I, 2)`
            - Output: Tuple of two arrays. Both have shape :math:`(W, H)`.

            | where
            |
            | :math:`I = \text{ number of tree instances}`
            | :math:`W = \text{ extent of the canopy height model in x-direction}`
            | :math:`H = \text{ extent of the canopy height model in y-direction}`
        """

        with Profiler("Watershed segmentation", self._performance_tracker):
            watershed_markers = np.zeros_like(canopy_height_model, dtype=np.int64)
            watershed_markers[tree_positions_grid[:, 0], tree_positions_grid[:, 1]] = np.arange(
                1, len(tree_positions_grid) + 1, dtype=np.int64
            )

            foreground_mask = canopy_height_model > 0

            # retrieve crown border and seed areas
            watershed_labels_with_border = watershed(
                -canopy_height_model, watershed_markers, mask=foreground_mask, watershed_line=True
            )
            border_mask = np.logical_and(watershed_labels_with_border == 0, foreground_mask)
            watershed_labels_without_border = watershed_labels_with_border.copy()
            watershed_labels_without_border[border_mask] = dilation(
                watershed_labels_with_border, footprint_rectangle((3, 3))
            )[border_mask]

        return watershed_labels_with_border, watershed_labels_without_border

    def watershed_correction(  # pylint: disable=too-many-locals
        self,
        watershed_labels_with_border: LongArray,
        watershed_labels_without_border: LongArray,
        tree_positions_grid: LongArray,
        point_cloud_id: Optional[str] = None,
    ) -> Tuple[LongArray, LongArray]:
        r"""
        Detects erroneous parts within a Watershed segmentation mask and replaces them by a Voronoi segmentation.

        Args:
            watershed_labels_with_border: Uncorrected Watershed labels with boundary lines between neighboring
                trees are assigned the background value of 0.
            watershed_labels_without_border: Uncorrected Watershed labels without boundary lines.
            tree_positions_grid: Indices of the grid cells of the canopy height model in which the tree positions are
                located.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. If set to
                :code:`None`, no visualizations are created.

        Returns:
            :Tuple of two arrays:
                - Corrected Watershed segmentation mask with the pixels at the boundary lines between neighboring trees
                  are assigned the background value of 0.
                - Corrected Watershed segmentation mask without boundary lines.

        Shape:
            - :code:`watershed_labels_with_border`: :math:`(W, H)`
            - :code:`watershed_labels_without_border`: :math:`(W, H)`
            - :code:`tree_positions_grid`: :math:`(I, 2)`
            - Output: :math:`(W, H)`, :math:`(W, H).

            | where
            |
            | :math:`I = \text{ number of tree instances}`
            | :math:`W = \text{ extent of the canopy height model in x-direction}`
            | :math:`H = \text{ extent of the canopy height model in y-direction}`
        """
        with Profiler('"Watershed correction', self._performance_tracker):
            self._logger.info("Correct Watershed segmentation...")
            for instance_id in np.unique(watershed_labels_without_border):
                if instance_id == 0:  # background
                    continue

                instance_mask = watershed_labels_without_border == instance_id

                dilated_instance_mask = dilation(instance_mask, footprint_rectangle((3, 3)))
                neighbor_instance_ids = np.unique(watershed_labels_without_border[dilated_instance_mask])

                if instance_mask.sum() == 1 or (len(neighbor_instance_ids) == 2) and (0 not in neighbor_instance_ids):
                    neighbor_instance_ids = neighbor_instance_ids[neighbor_instance_ids != 0]

                    self._logger.info(
                        "Detected inaccurate watershed segmentation for tree %d. Falling back to Voronoi"
                        + " segmentation.",
                        instance_id,
                    )

                    neighborhood_mask_without_border = instance_mask.copy()
                    for neighbor_id in neighbor_instance_ids:
                        neighborhood_mask_without_border = np.logical_or(
                            neighborhood_mask_without_border, watershed_labels_without_border == neighbor_id
                        )

                    current_tree_positions = tree_positions_grid[neighbor_instance_ids - 1]
                    voronoi_input = np.zeros_like(watershed_labels_without_border, dtype=np.uint32)

                    voronoi_input[current_tree_positions[:, 0], current_tree_positions[:, 1]] = neighbor_instance_ids

                    # when all pixels are assigned the same height value, the Watershed algorithm approximates a Voronoi
                    # segmentation
                    voronoi_labels_with_border = watershed(
                        np.zeros_like(voronoi_input),
                        voronoi_input,
                        mask=neighborhood_mask_without_border,
                        watershed_line=True,
                    )
                    border_mask = np.logical_and(voronoi_labels_with_border == 0, neighborhood_mask_without_border)
                    voronoi_labels_without_border = voronoi_labels_with_border.copy()
                    voronoi_labels_without_border[border_mask] = dilation(
                        voronoi_labels_with_border, footprint_rectangle((3, 3))
                    )[border_mask]

                    if self._visualization_folder is not None and point_cloud_id is not None:

                        img_path = (
                            f"{self._visualization_folder}/voronoi_labels_without_border_{point_cloud_id}_"
                            + f"{instance_id}.png"
                        )
                        save_tree_map(
                            voronoi_labels_without_border,
                            output_path=img_path,
                            is_label_image=True,
                            tree_positions=current_tree_positions,
                        )

                        img_path = (
                            f"{self._visualization_folder}/voronoi_labels_with_border_{point_cloud_id}_"
                            + f"{instance_id}.png"
                        )
                        save_tree_map(
                            voronoi_labels_with_border,
                            output_path=img_path,
                            is_label_image=True,
                            tree_positions=current_tree_positions,
                        )

                    voronoi_labels_without_border_remapped = np.zeros_like(voronoi_labels_without_border)
                    voronoi_labels_with_border_remapped = np.zeros_like(voronoi_labels_with_border)
                    for instance_id in neighbor_instance_ids:
                        tree_position = tree_positions_grid[instance_id - 1]
                        voronoi_id = voronoi_labels_without_border[tree_position[0], tree_position[1]]
                        voronoi_labels_without_border_remapped[voronoi_labels_without_border == voronoi_id] = (
                            instance_id
                        )
                        voronoi_labels_with_border_remapped[voronoi_labels_with_border == voronoi_id] = instance_id

                    watershed_labels_without_border[neighborhood_mask_without_border] = (
                        voronoi_labels_without_border_remapped[neighborhood_mask_without_border]
                    )

                    # find outer boundaries to other trees that were not included in the Voronoi segmentation
                    outer_boundaries = find_boundaries(neighborhood_mask_without_border, mode="inner", background=0)
                    watershed_labels_without_border_without_neighborhood = watershed_labels_without_border.copy()
                    watershed_labels_without_border_without_neighborhood[neighborhood_mask_without_border] = 0
                    dilated_mask = dilation(watershed_labels_without_border_without_neighborhood, diamond(1))
                    outer_boundaries = np.logical_and(outer_boundaries, dilated_mask != 0)

                    neighborhood_mask_with_border = neighborhood_mask_without_border.copy()
                    neighborhood_mask_with_border[outer_boundaries] = False

                    watershed_labels_with_border[neighborhood_mask_with_border] = voronoi_labels_with_border_remapped[
                        neighborhood_mask_with_border
                    ]

        return watershed_labels_with_border, watershed_labels_without_border

    def determine_overlapping_crowns(  # pylint: disable=too-many-locals
        self,
        tree_xyz: FloatArray,
        classification: LongArray,
        instance_ids: LongArray,
        unique_instance_ids: LongArray,
        canopy_height_model: FloatArray,
        watershed_labels_with_border: LongArray,
        watershed_labels_without_border: LongArray,
    ) -> Tuple[BoolArray, LongArray]:
        r"""
        Identifies trees whose crowns overlap with neighboring trees. If such trees have sufficient point density and
        contain trunk points, their segmentation is refined. For this purpose, the trunk points are selected as seed
        points for region growing and the instance IDs of the crown points are reset to -1.


        Args:
            tree_xyz: Coordinates of all tree points.
            classification: Class ID of all tree points.
            instance_ids: Instance IDs of each tree point.
            unique_instance_ids: Unique instance IDs.
            canopy_height_model: The canopy height model.
            watershed_labels_with_border: Watershed labels with boundary lines between neighboring
                trees are assigned the background value of 0.

        Returns:
            :Tuple of two arrays:
                - A boolean mask indicating which points were selected as seed points for region growing.
                - Updated instance IDs for each tree points. For trees whose segmentation is to be refined the instance
                  IDs of the crown points is set to :code:`-1`.

        Shape:
            - :code:`tree_xyz`: :math:`(N, 3)`
            - :code:`classification`: :math:`(N)`
            - :code:`instance_ids`: :math:`(N)`
            - :code:`unique_instance_ids`: :math:`(I)`
            - :code:`canopy_height_model`: :math:`(W, H)`
            - :code:`watershed_labels_with_border`: :math:`(W, H)`
            - Output: :math:`(N)`, :math:`(N)`.

            | where
            |
            | :math:`N = \text{ number of tree points}`
            | :math:`I = \text{ number of tree instances}`
            | :math:`W = \text{ extent of the canopy height model in x-direction}`
            | :math:`H = \text{ extent of the canopy height model in y-direction}`
        """
        with Profiler("Seed Mask computation", self._performance_tracker):
            self._logger.info("Compute region growing masks...")
            foreground_mask = canopy_height_model > 0

            watershed_labels_for_erosion = watershed_labels_with_border + 1
            watershed_labels_for_erosion[np.logical_and(watershed_labels_for_erosion == 1, foreground_mask)] = 0

            eroded_watershed_labels = erosion(watershed_labels_for_erosion, disk(1))
            mask_background_erosion = np.logical_and(foreground_mask, eroded_watershed_labels == 1)
            eroded_watershed_labels[mask_background_erosion] = watershed_labels_for_erosion[mask_background_erosion]

            kd_tree = KDTree(tree_xyz)

            neighbor_distances = cast(np.ndarray, kd_tree.query(tree_xyz, k=2)[0])
            neighbor_distances = cast(np.ndarray, neighbor_distances)
            neighbor_distances = neighbor_distances[:, 1].flatten()

            instances_to_refine = []
            average_point_spacings = []
            instance_seed_masks = []

            for instance_id in unique_instance_ids:
                watershed_mask = watershed_labels_with_border == instance_id + 1
                eroded_watershed_mask = eroded_watershed_labels == instance_id + 2

                if (watershed_mask == eroded_watershed_mask).all():
                    # the segmentation is not refined in that case
                    continue

                instance_mask = instance_ids == instance_id
                instance_trunk_mask = classification == self._trunk_class_id
                if self._branch_class_id is not None:
                    instance_trunk_mask = np.logical_or(instance_trunk_mask, classification == self._branch_class_id)

                instance_trunk_mask = np.logical_and(instance_trunk_mask, instance_mask)

                average_point_spacing = neighbor_distances[instance_mask].mean()

                if instance_trunk_mask.sum() > 0 and average_point_spacing <= self._max_point_spacing_region_growing:
                    instances_to_refine.append(instance_id)
                    average_point_spacings.append(average_point_spacing)
                    instance_seed_masks.append(instance_trunk_mask)

            seed_mask = np.zeros(len(tree_xyz), dtype=bool)

            for instance_id, average_point_spacing, instance_seed_mask in zip(
                instances_to_refine, average_point_spacings, instance_seed_masks
            ):
                instance_mask = watershed_labels_without_border == instance_id + 1
                dilated_instance_mask = dilation(instance_mask, footprint_rectangle((3, 3)))
                neighbor_instance_ids = np.unique(watershed_labels_without_border[dilated_instance_mask]) - 1

                has_neighbor_to_refine = False
                for neighbor_instance_id in neighbor_instance_ids:
                    if neighbor_instance_id in (instance_id, -1):
                        continue
                    if neighbor_instance_id in instances_to_refine:
                        has_neighbor_to_refine = True
                        break

                if has_neighbor_to_refine:
                    self._logger.info(
                        "Segmentation of %d will be refined (Average point spacing: %.3f)...",
                        instance_id,
                        average_point_spacing,
                    )
                    instance_crown_mask = np.logical_and(
                        instance_ids == instance_id, classification == self._crown_class_id
                    )
                    instance_ids[instance_crown_mask] = -1
                    seed_mask = np.logical_or(seed_mask, instance_seed_mask)

        return seed_mask, instance_ids

    def compute_crown_distance_fields(
        self, watershed_labels_without_border: LongArray, tree_positions_grid: LongArray
    ) -> FloatArray:
        r"""
        Calculates signed distance fields from the 2D segmentation mask of each tree. The distance values specify the 2D
        distance to the boundary of the segmentation mask. The distance value is negative for pixels that belong to the
        tree and positive for pixels that do not belong to the tree.

        Args:
            watershed_labels_without_border: Watershed labels without boundary lines.
            tree_positions_grid: Indices of the grid cells of the canopy height model in which the tree positions are
                located.

        Returns:
            Signed distance masks for all trees.

        Shape:
            - :code:`watershed_labels_without_border`: :math:`(W, H)`
            - :code:`tree_positions_grid`: :math:`(N, 2)`
            - Output: :math:`(I, W, H)`.

            | where
            |
            | :math:`N = \text{ number of tree points}`
            | :math:`I = \text{ number of tree instances}`
            | :math:`W = \text{ extent of the distance fields in x-direction}`
            | :math:`H = \text{ extent of the distance fields in y-direction}`
        """
        with Profiler("Computation of crown distance fields", self._performance_tracker):
            self._logger.info("Compute crown distance fields...")
            crown_distance_fields = np.empty((len(tree_positions_grid), *watershed_labels_without_border.shape))
            for idx, tree_position in enumerate(tree_positions_grid):
                instance_id = watershed_labels_without_border[tree_position[0], tree_position[1]]
                mask = watershed_labels_without_border == instance_id
                inverse_mask = np.logical_not(mask)
                distance_mask = -cast(  # pylint: disable=invalid-unary-operand-type
                    np.ndarray, ndi.distance_transform_edt(mask)
                )
                distance_mask[inverse_mask] = ndi.distance_transform_edt(inverse_mask)[inverse_mask]
                crown_distance_fields[idx] = distance_mask

        return crown_distance_fields

    def grow_trees(  # pylint: disable=too-many-locals
        self,
        tree_xyz: FloatArray,
        instance_ids: LongArray,
        unique_instances_ids: LongArray,
        grid_origin: FloatArray,
        crown_distance_fields: FloatArray,
        seed_mask: BoolArray,
    ) -> LongArray:
        r"""
        Region growing segmentation.

        Args:
            tree_xyz: Coordinates of all tree points.
            instance_ids: Instance IDs of each tree point.
            grid_origin: 2D coordinates of the origin of the crown distance fields.
            crown_distance_fields: 2D signed distance fields idicating the distance to the crown boundary of each tree
                to segment.
            seed_mask: Boolean mask indicating which points were selected as seed points for region growing.

        Returns:
            Updated instance IDs.

        Shape:
            - :code:`tree_xyz`: :math:`(N, 3)`
            - :code:`instance_ids`: :math:`(N)`
            - :code:`grid_origin`: :math:`(2)`
            - :code:`crown_distance_fields`: :math:`(I', W, H)`
            - :code:`seed_mask`: :math:`(N)`
            - Output: :math:`(N)`.

            | where
            |
            | :math:`N = \text{ number of tree points}`
            | :math:`I' = \text{ number of tree instances to segment}`
            | :math:`W = \text{ extent of the distance fields in x-direction}`
            | :math:`H = \text{ extent of the distance fields in y-direction}`

        """
        with Profiler("Region growing", self._performance_tracker):
            self._logger.info("Region growing...")

            if not tree_xyz.flags.f_contiguous:
                tree_xyz = tree_xyz.copy(order="F")
            grid_origin = grid_origin.astype(tree_xyz.dtype)

            return grow_trees_cpp(
                tree_xyz,
                instance_ids,
                unique_instances_ids,
                grid_origin,
                crown_distance_fields,
                seed_mask,
                float(self._z_scale),
                int(self._num_neighbors_region_growing),
                float(self._max_radius_region_growing),
                float(self._grid_size_canopy_height_model),
                float(self._multiplier_outside_coarse_border),
                int(self._num_workers),
            )

    def __call__(  # pylint: disable=too-many-locals
        self,
        xyz: FloatArray,
        classification: LongArray,
        point_cloud_id: Optional[str] = None,
    ) -> LongArray:
        r"""
        Segments tree instances in a point cloud.

        Args:
            xyz: 3D coordinates of the point cloud to be segmented.
            classification: Semantic segmentation class IDs
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. If set to
                :code:`None`, no visualizations are created.

        Returns:
            Instance IDs of each point. Points that do not belong to any instance are assigned the ID :math:`-1`.

        Shape:
            - :code:`xyz`: :math:`(N, 3)`
            - :code:`classification`: :math:`(N)`
            - Output: :math:`(N)`.

            | where
            |
            | :math:`N = \text{ number of points}`
        """

        self._performance_tracker.reset()
        with Profiler("Total", self._performance_tracker):
            # check that point cloud contains all required variables
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                raise ValueError("xyz must have shape (N, 3).")
            if len(xyz) != len(classification):
                raise ValueError("xyz and classification must contain the same number of entries.")

            point_cloud_size = len(xyz)

            tree_mask = np.logical_or(
                classification == self._trunk_class_id,
                classification == self._crown_class_id,
            )
            if self._branch_class_id is not None:
                tree_mask = np.logical_or(tree_mask, classification == self._branch_class_id)

            if tree_mask.sum() == 0:
                return np.full(len(xyz), fill_value=-1, dtype=np.int64)

            tree_xyz = xyz[tree_mask]
            tree_classification = classification[tree_mask]

            if self._downsampling_voxel_size is not None:
                with Profiler("Voxel-based downsampling", self._performance_tracker):
                    self._logger.info("Downsample point cloud...")
                    downsampled_tree_xyz, predicted_point_indices, inverse_indices = voxel_downsampling(
                        tree_xyz, self._downsampling_voxel_size, point_aggregation="nearest_neighbor"
                    )
                    downsampled_classification = tree_classification[predicted_point_indices]
                    self._logger.info("Points after downsampling: %d", len(downsampled_tree_xyz))
            else:
                downsampled_tree_xyz = tree_xyz
                downsampled_classification = tree_classification

            instance_ids, unique_instance_ids = self._segment_tree_points(
                downsampled_tree_xyz, downsampled_classification, point_cloud_id=point_cloud_id
            )

            if self._downsampling_voxel_size is not None:
                with Profiler("Upsampling of labels", self._performance_tracker):
                    instance_ids = instance_ids[inverse_indices]

            full_instance_ids = np.full(point_cloud_size, fill_value=-1, dtype=np.int64)
            full_instance_ids[tree_mask] = instance_ids

        self._logger.info("Detected %d trees.", len(unique_instance_ids))

        return full_instance_ids
