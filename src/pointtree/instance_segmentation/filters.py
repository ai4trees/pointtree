""" Methods to filter instances. """

__all__ = ["filter_instances_min_points", "filter_instances_vertical_extent"]

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from pointtorch.operations.numpy import make_labels_consecutive


def filter_instances_min_points(
    instance_ids: npt.NDArray[np.int64],
    unique_instance_ids: npt.NDArray[np.int64],
    min_points: Optional[int],
    inplace: bool = False,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r"""
    Removes instances with less than :code:`min_points`.

    Args:
        instance_ids: Instance IDs of the points.
        unique_instance_ids: Unique instance IDs.
        min_points: Minimum number of points an instance must have to not be discarded. If set to :code:`None`, the
            instances are not filtered.
        inplace: Whether the filtering should be applied inplace to the :code:`instance_ids` array. Defaults to
            :code:`False`.

    Returns:
        Tuple of two arrays. The first contains the updated instance ID of each point. Points that do not
        belong to any instance are assigned the ID :math:`-1`. The second contains the unique instance IDs.

    Shape:
        - :code:`instance_ids`: :math:`(N)`
        - :code:`unique_instance_ids`: :math:`(I)`
        - Output: Tuple of two arrays. The first has shape :math:`(N)` and the second :math:`(I')`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`I = \text{ number of instances before filtering}`
        | :math:`I' = \text{ number of instances after filtering}`
    """

    if min_points is None:
        return instance_ids, unique_instance_ids

    unique_instance_ids, point_counts = np.unique(instance_ids, return_counts=True)
    discarded_instance_ids = unique_instance_ids[point_counts < min_points]
    instance_ids[np.in1d(instance_ids, discarded_instance_ids)] = -1

    return make_labels_consecutive(instance_ids, ignore_id=-1, inplace=inplace, return_unique_labels=True)


def filter_instances_vertical_extent(
    coords: npt.NDArray[np.float64],
    instance_ids: npt.NDArray[np.int64],
    unique_instance_ids: npt.NDArray[np.int64],
    min_vertical_extent: Optional[float],
    inplace: bool = False,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r"""
    Removes instances whose extent in z-direction is less than :code:`min_vertical_extent`.

    Args:
        instance_ids: Instance IDs of the points.
        unique_instance_ids: Unique instance IDs.
        min_vertical_extent: Minimum vertical extent an instance must have to not be discarded. If set to :code:`None`,
            the instances are not filtered.
        inplace: Whether the filtering should be applied inplace to the :code:`instance_ids` array. Defaults to
            :code:`False`.

    Returns:
        Tuple of two arrays. The first contains the updated instance ID of each point. Points that do not
        belong to any instance are assigned the ID :math:`-1`. The second contains the unique instance IDs.

    Shape:
        - :code:`instance_ids`: :math:`(N)`
        - :code:`unique_instance_ids`: :math:`(I)`
        - Output: Tuple of two arrays. The first has shape :math:`(N)` and the second :math:`(I')`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`I = \text{ number of instances before filtering}`
        | :math:`I' = \text{ number of instances after filtering}`
    """

    if len(instance_ids) != len(coords):
        raise ValueError("coords and instance_ids must have the same length.")

    if min_vertical_extent is None:
        return instance_ids, unique_instance_ids

    unique_instance_ids, inverse_indices = np.unique(instance_ids, return_inverse=True)

    min_z_per_cluster = np.full(len(unique_instance_ids), fill_value=np.inf, dtype=coords.dtype)
    max_z_per_cluster = np.full(len(unique_instance_ids), fill_value=-np.inf, dtype=coords.dtype)

    np.minimum.at(min_z_per_cluster, inverse_indices, coords[:, 2])
    np.maximum.at(max_z_per_cluster, inverse_indices, coords[:, 2])

    discarded_instance_ids = unique_instance_ids[max_z_per_cluster - min_z_per_cluster < min_vertical_extent]
    instance_ids[np.in1d(instance_ids, discarded_instance_ids)] = -1

    return make_labels_consecutive(instance_ids, ignore_id=-1, inplace=inplace, return_unique_labels=True)
