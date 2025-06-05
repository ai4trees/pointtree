__all__ = ["statistical_outlier_removal"]

from typing import Tuple

import numpy as np
from scipy.spatial import KDTree

from pointtree.type_aliases import FloatArray, LongArray


def statistical_outlier_removal(xyz: FloatArray, k: int, std_multiplier: float) -> Tuple[FloatArray, LongArray]:
    r"""
    Statistical outlier filter for 3D point clouds: First, it computes the average distance that each point has to
    its k nearest neighbors. Next, the mean and standard deviation of these average distances are computed to determine
    a distance threshold. The distance threshold is set to: :code:`mean + std_multiplier * stddev`. Then, points are
    classified as outlier and removed if their average neighbor distance is above this threshold.

    Args:
        xyz: Coordinates of the points to be filtered.
        k: Number of neighboring points to consider for the filtering (the point itself is not included).
        std_multiplier: Multiplier for the standard deviation in the calculation of the distance threshold.

    Returns:
        : Tuple of two arrays:
            - Coordinates of the inlier points remaining after filtering.
            - Indices of the inlier points remaining after filtering with respect to the input array.

    Shape:
        - :code:`xyz`: :math:`(N, 3)`
        - Output: :math:`(N', 3)`, :math:`(N')`

          | where
          |
          | :math:`N = \text{ number of points before the filtering}`
          | :math:`N' = \text{ number of points after the filtering}`
    """

    if len(xyz) <= 2:
        return xyz, np.arange(len(xyz), dtype=np.int64)

    k = min(k, len(xyz) - 1)

    kd_tree = KDTree(xyz)
    neighbor_dists, neighbor_indices = kd_tree.query(xyz, k=k + 1)

    neighbor_dists = neighbor_dists[:, 1:]
    neighbor_indices = neighbor_indices[:, 1:]

    average_dist_to_neighbors = neighbor_dists.mean(axis=1)
    dist_to_neighbors_mean = average_dist_to_neighbors.mean()
    dist_to_neighbors_std = average_dist_to_neighbors.std()

    selected_indices = np.flatnonzero(
        average_dist_to_neighbors <= dist_to_neighbors_mean + dist_to_neighbors_std * std_multiplier
    )

    return xyz[selected_indices], selected_indices
