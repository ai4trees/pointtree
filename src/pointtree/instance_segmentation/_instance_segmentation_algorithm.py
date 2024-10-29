""" Base class for implementing tree instance segmentation algorithms. """

__all__ = ["InstanceSegmentationAlgorithm"]

import abc
import logging
from typing import Optional, Tuple
import sys

from numba_kdtree import KDTree
import numpy as np
import pandas as pd

from pointtree.evaluation import Timer, TimeTracker
from pointtree.operations import voxel_downsampling


class InstanceSegmentationAlgorithm(abc.ABC):
    """
    Base class for implementing tree instance segmentation algorithms.

    Args:
        trunk_class_id: Integer class ID that designates the tree trunk points.
        crown_class_id: Integer class ID that designates the tree crown points.
        branch_class_id: Integer class ID that designates the tree branch points. Defaults to `None`, assuming that
            branch points are not separately labeled.
    downsampling_voxel_size: Voxel size for the voxel-based downsampling of the tree points before performing the
        tree instance segmentation. Defaults to :code:`None`, which means that the tree instance segmentation is
        performed with the full resolution of the point cloud.
    """

    def __init__(
        self,
        trunk_class_id: int,
        crown_class_id: int,
        branch_class_id: Optional[int] = None,
        downsampling_voxel_size: Optional[float] = None,
    ):
        self._trunk_class_id = trunk_class_id
        self._crown_class_id = crown_class_id
        self._branch_class_id = branch_class_id

        self._downsampling_voxel_size = downsampling_voxel_size
        self._time_tracker = TimeTracker()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

    def __call__(  # pylint: disable=too-many-locals
        self,
        point_cloud: pd.DataFrame,
        point_cloud_id: Optional[str] = None,
        semantic_segmentation_column: str = "classification",
    ) -> np.ndarray:
        r"""
        Segments tree instances in a point cloud.

        Args:
            point_cloud: Point cloud to be segmented.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. Defaults to
                `None`, which means that no visualizations are created.
            semantic_segmentation_column: Point cloud column containing the semantic segmentation class IDs. Defaults to
                `"classification"`.

        Returns:
            Instance IDs of each point. Points that do not belong to any instance are assigned the ID :math:`-1`.

        Shape:
            - :code:`point_cloud`: :math:`(N, D)`
            - :code:`classification`: :math:`(N)`
            - Output: :math:`(N)`.

            | where
            |
            | :math:`N = \text{ number of tree points}`
            | :math:`N = \text{ number of feature channels per point}`
        """

        self._time_tracker.reset()
        with Timer("Total", self._time_tracker):
            # check that point cloud contains all required variables
            required_columns = ["x", "y", "z", semantic_segmentation_column]
            missing_columns = list(set(required_columns).difference(point_cloud.columns))
            if len(missing_columns) > 0:
                if len(missing_columns) > 1:
                    missing_columns_str = ", ".join(missing_columns[:-1]) + ", and " + missing_columns[-1]
                else:
                    missing_columns_str = missing_columns[0]
                raise ValueError(f"The point cloud must contain the columns {missing_columns_str}.")

            point_cloud_size = len(point_cloud)

            tree_mask = np.logical_or(
                point_cloud[semantic_segmentation_column].to_numpy() == self._trunk_class_id,
                point_cloud[semantic_segmentation_column].to_numpy() == self._crown_class_id,
            )
            if self._branch_class_id is not None:
                tree_mask = np.logical_or(
                    tree_mask, point_cloud[semantic_segmentation_column].to_numpy() == self._branch_class_id
                )

            if tree_mask.sum() == 0:
                return np.full(len(point_cloud), fill_value=-1, dtype=np.int64)

            tree_points = point_cloud[tree_mask]
            del point_cloud

            if self._downsampling_voxel_size is not None:
                with Timer("Voxel-based downsampling", self._time_tracker):
                    self._logger.info("Downsample point cloud...")
                    _, predicted_point_indices = voxel_downsampling(
                        tree_points.to_numpy(), self._downsampling_voxel_size, point_aggregation="nearest_neighbor"
                    )
                    downsampled_tree_points = tree_points.iloc[predicted_point_indices]
                    self._logger.info("Points after downsampling: %d", len(downsampled_tree_points))
                non_predicted_point_indices = np.setdiff1d(
                    np.arange(len(tree_points), dtype=np.int64), predicted_point_indices
                )
            else:
                downsampled_tree_points = tree_points

            tree_coords = downsampled_tree_points[["x", "y", "z"]].to_numpy()
            classification = downsampled_tree_points[semantic_segmentation_column].to_numpy()
            del downsampled_tree_points

            instance_ids, unique_instance_ids = self._segment_tree_points(
                tree_coords, classification, point_cloud_id=point_cloud_id
            )

            if self._downsampling_voxel_size is not None:
                with Timer("Upsampling of labels", self._time_tracker):
                    instance_ids = self.upsample_instance_ids(
                        tree_points[["x", "y", "z"]].to_numpy(),
                        tree_coords,
                        instance_ids,
                        non_predicted_point_indices,
                        predicted_point_indices,
                    )

            full_instance_ids = np.full(point_cloud_size, fill_value=-1, dtype=np.int64)
            full_instance_ids[tree_mask] = instance_ids

        self._logger.info("Detected %d trees.", len(unique_instance_ids))

        return full_instance_ids

    @abc.abstractmethod
    def _segment_tree_points(
        self, tree_coords: np.ndarray, classification: np.ndarray, point_cloud_id: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Performs tree instance segmentation. Has to be overridden by subclasses.

        Args:
            tree_coords: Coordinates of all tree points.
            classification: Class IDs of each tree point.
            point_cloud_id: ID of the point cloud to be used in the file names of the visualization results. Defaults to
                `None`, which means that no visualizations are created.

        Returns:
            Instance IDs of each point and unique instance IDs. Points that do not belong to any instance are assigned
            the ID :math:`-1`.

        Shape:
            - :code:`tree_coords`: :math:`(N, 3)`
            - :code:`classification`: :math:`(N)`
            - Output: Tuple of two arrays. The first has shape :math:`(N)` and the second :math:`(I)`.

            | where
            |
            | :math:`N = \text{ number of tree points}`
            | :math:`I = \text{ number of tree instances}`
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

    def runtime_stats(
        self,
    ) -> pd.DataFrame:
        """
        Returns:
            Tracked execution times as
            `pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`__ with the columns
            :code:`"Description"` and :code:`"Runtime"`.
        """

        return self._time_tracker.to_pandas()
