import abc
import logging
from typing import Optional

from numba_kdtree import KDTree
import numpy as np
import pandas as pd
from pointtorch.operations.numpy import voxel_downsampling

from forest3d.evaluation import Timer, TimeTracker


class InstanceSegmentationAlgorithm(abc.ABC):
    def __init__(self, downsampling_voxel_size: Optional[float] = None):
        self._downsampling_voxel_size = downsampling_voxel_size
        self._time_tracker = TimeTracker()
        self._logger = logging.getLogger(__name__)

    def __call__(self, point_cloud: pd.DataFrame, point_cloud_id: Optional[str] = None) -> np.ndarray:
        r"""
        Segments tree instances in a point cloud.

        Args:
            tree_coords: Coordinates of all tree points.
            classification: Class IDs of each tree point.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. Defaults to
                `None`, which means that no visualizations are created.

        Returns:
            Instance IDs of each point. Points that do not belong to any instance are assigned the ID :math:`-1`.

        Shape:
            - :code:`tree_coords`: :math:`(N, 3)`
            - :code:`classification`: :math:`(N)`
            - Output: :math:`(N)`.

            | where
            |
            | :math:`N = \text{ number of tree points}`
        """

        self._time_tracker.reset()
        with Timer("Total", self._time_tracker):
            # check that point cloud contains all required variables
            required_columns = ["x", "y", "z", "classification"]
            missing_columns = list(set(required_columns).difference(point_cloud.columns))
            if len(missing_columns) > 0:
                if len(missing_columns) > 1:
                    missing_columns_str = ", ".join(missing_columns[:-1]) + ", and " + missing_columns[-1]
                else:
                    missing_columns_str = missing_columns[0]
                raise ValueError(f"The point cloud must contain the columns {missing_columns_str}.")

            point_cloud_size = len(point_cloud)

            tree_mask = np.logical_or(
                point_cloud["classification"] == self._trunk_class_id,
                point_cloud["classification"] == self._crown_class_id,
            )
            if self._branch_class_id is not None:
                tree_mask = np.logical_or(tree_mask, point_cloud["classification"] == self._branch_class_id)

            if tree_mask.sum() == 0:
                return np.full(len(point_cloud), fill_value=-1, dtype=np.int64)

            tree_points = point_cloud[tree_mask]
            del point_cloud

            if self._downsampling_voxel_size is not None:
                with Timer("Grid subsampling", self._time_tracker):
                    self._logger.info("Downsample point cloud...")
                    print("start downsampling")
                    _, predicted_point_indices = voxel_downsampling(tree_points.to_numpy(), self._downsampling_voxel_size, point_aggregation="nearest_neighbor")
                    downsampled_tree_points = tree_points.iloc[predicted_point_indices]
                    self._logger.info("Points after downsampling:", len(downsampled_tree_points))
                    print("downsampled_tree_points", downsampled_tree_points)
                point_indices = np.arange(len(tree_points), dtype=np.int64)
                non_predicted_point_indices = np.setdiff1d(point_indices, predicted_point_indices)
            else:
                downsampled_tree_points = tree_points

            tree_coords = downsampled_tree_points[["x", "y", "z"]].to_numpy()
            classification = downsampled_tree_points["classification"].to_numpy()
            del downsampled_tree_points

            instance_ids = self._segment_tree_points(tree_coords, classification, point_cloud_id=point_cloud_id)

            if self._downsampling_voxel_size is not None:
                instance_ids = self.upsample_instance_ids(
                    tree_points[["x", "y", "z"]].to_numpy(),
                    tree_coords,
                    instance_ids,
                    non_predicted_point_indices,
                    predicted_point_indices,
                )

            full_instance_ids = np.full(point_cloud_size, fill_value=-1, dtype=np.int64)
            full_instance_ids[tree_mask] = instance_ids

        return full_instance_ids

    @abc.abstractmethod
    def _segment_tree_points(self, tree_coords: np.ndarray, classification: np.ndarray, point_cloud_id: Optional[str] = None):
        r"""
        Performs tree instance segmentation. Has to be overridden by subclasses.

        Args:
            tree_coords: Coordinates of all tree points.
            classification: Class IDs of each tree point.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. Defaults to
                `None`, which means that no visualizations are created.

        Returns:
            Instance IDs of each point. Points that do not belong to any instance are assigned the ID :math:`-1`.

        Shape:
            - :code:`tree_coords`: :math:`(N, 3)`
            - :code:`classification`: :math:`(N)`
            - Output: :math:`(N)`.

            | where
            |
            | :math:`N = \text{ number of tree points}`
        """

    @staticmethod
    def upsample_instance_ids(
        original_coords: np.ndarray,
        downsampled_coords: np.ndarray,
        instance_ids: np.ndarray,
        non_predicted_point_indices: np.ndarray,
        predicted_point_indices: np.ndarray,
    ) -> np.ndarray:
        r"""
        Upsamples instance segmentation labels to a higher point cloud resolution.

        Args:
            original_coords: Coordinates of the points from the higher-resolution point cloud
            downsampled_coords: Coordinates of the points from the lower-resolution point cloud.
            instance_ids: Instance segmentation labels to be upsampled.
            non_predicted_point_indices: Indices of the points from the higher-resolution point cloud that were
                discarded during downsampling.
            predicted_point_indices: Indices of the points from the higher-resolution point cloud that were
                kept during downsampling.

        Returns:
            Upsampled instance segmentation labels.

        Shape:
            - :code:`original_coords`: :math:`(N, 3)`
            - :code:`downsampled_coords`: :math:`(N', 3)`
            - :code:`instance_ids`: :math:`(N')`
            - :code:`instance_ids`: :math:`(N - N')`
            - :code:`instance_ids`: :math:`(N')`
            - Output: :math:`(N)`.

            | where
            |
            | :math:`N = \text{ number of points before in the higher-resolution point cloud}`
            | :math:`N' = \text{ number of points before in the lower-resolution point cloud}`
        """

        query_points = original_coords[non_predicted_point_indices]
        nearest_neighbor_indices = KDTree(downsampled_coords).query(query_points, k=1)[1]

        upsampled_instance_ids = np.full(len(original_coords), fill_value=-1, dtype=np.int64)
        upsampled_instance_ids[predicted_point_indices] = instance_ids
        upsampled_instance_ids[non_predicted_point_indices] = instance_ids[nearest_neighbor_indices[:, 0]]

        return upsampled_instance_ids

    def runtime_stats(self,) -> pd.DataFrame:
        """
        Returns:
            Tracked execution times as
            `pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`__ with the columns
            :code:`"Description"` and :code:`"Runtime"`.
        """

        return self._time_tracker.to_pandas()
