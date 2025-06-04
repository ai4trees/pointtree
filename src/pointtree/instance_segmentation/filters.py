"""Methods to filter instances."""

__all__ = [
    "filter_instances_intensity",
    "filter_instances_min_points",
    "filter_instances_pca",
    "filter_instances_vertical_extent",
]

from typing import Optional, Tuple

import numpy as np
from pointtorch.operations.numpy import make_labels_consecutive
from sklearn.decomposition import PCA

from pointtree.type_aliases import FloatArray, LongArray


def filter_instances_min_points(
    instance_ids: LongArray,
    unique_instance_ids: LongArray,
    min_points: Optional[int],
    inplace: bool = False,
) -> Tuple[LongArray, LongArray]:
    r"""
    Removes instances with less than :code:`min_points`.

    Args:
        instance_ids: Instance IDs of the points.
        unique_instance_ids: Unique instance IDs.
        min_points: Minimum number of points an instance must have to not be discarded. If set to :code:`None`, the
            instances are not filtered.
        inplace: Whether the filtering should be applied inplace to the :code:`instance_ids` array.

    Returns:
        :Tuple of two arrays:
            - Updated instance ID of each point. Points that do not belong to any instance are assigned the ID
              :code:`-1`.
            - Unique instance IDs.

    Shape:
        - :code:`instance_ids`: :math:`(N)`
        - :code:`unique_instance_ids`: :math:`(I)`
        - Output: :math:`(N)`, :math:`(I')`

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

    return make_labels_consecutive(  # type: ignore[return-value]
        instance_ids, ignore_id=-1, inplace=inplace, return_unique_labels=True
    )


def filter_instances_vertical_extent(
    xyz: FloatArray,
    instance_ids: LongArray,
    unique_instance_ids: LongArray,
    min_vertical_extent: Optional[float],
    inplace: bool = False,
) -> Tuple[LongArray, LongArray]:
    r"""
    Removes instances whose extent in z-direction is less than :code:`min_vertical_extent`.

    Args:
        instance_ids: Instance IDs of the points.
        unique_instance_ids: Unique instance IDs.
        min_vertical_extent: Minimum vertical extent an instance must have to not be discarded. If set to :code:`None`,
            the instances are not filtered.
        inplace: Whether the filtering should be applied inplace to the :code:`instance_ids` array.

    Returns:
        :Tuple of two arrays:
            - Updated instance ID of each point. Points that do not belong to any instance are assigned the ID
              :code:`-1`.
            - Unique instance IDs.

    Shape:
        - :code:`xyz`: :math:`(N, 3)`
        - :code:`instance_ids`: :math:`(N)`
        - :code:`unique_instance_ids`: :math:`(I)`
        - Output: :math:`(N)`, :math:`(I')`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`I = \text{ number of instances before filtering}`
        | :math:`I' = \text{ number of instances after filtering}`
    """

    if len(instance_ids) != len(xyz):
        raise ValueError("xyz and instance_ids must have the same length.")

    if min_vertical_extent is None:
        return instance_ids, unique_instance_ids

    unique_instance_ids, inverse_indices = np.unique(instance_ids, return_inverse=True)

    min_z_per_cluster = np.full(len(unique_instance_ids), fill_value=np.inf, dtype=xyz.dtype)
    max_z_per_cluster = np.full(len(unique_instance_ids), fill_value=-np.inf, dtype=xyz.dtype)

    np.minimum.at(min_z_per_cluster, inverse_indices, xyz[:, 2])
    np.maximum.at(max_z_per_cluster, inverse_indices, xyz[:, 2])

    discarded_instance_ids = unique_instance_ids[max_z_per_cluster - min_z_per_cluster < min_vertical_extent]

    if not inplace:
        instance_ids = instance_ids.copy()

    instance_ids[np.in1d(instance_ids, discarded_instance_ids)] = -1

    return make_labels_consecutive(  # type: ignore[return-value]
        instance_ids, ignore_id=-1, inplace=inplace, return_unique_labels=True
    )


def filter_instances_pca(
    xyz: FloatArray,
    instance_ids: LongArray,
    unique_instance_ids: LongArray,
    min_explained_variance: Optional[float],
    max_inclination: Optional[float],
    inplace: bool = False,
) -> Tuple[LongArray, LongArray]:
    r"""
    Performs a principal component analysis on the points of each instance and removes instances for which the first
    principal component explains less than :code:`min_explained_variance` of the point's variance or the inclination
    angle between the z-axis and the first principal component is greater than :code:`max_inclination`.

    Args:
        instance_ids: Instance IDs of the points.
        unique_instance_ids: Unique instance IDs.
        min_explained_variance: Minimum percentage of variance that the first principal component of an instance
            must explain in order to not be discarded.
        max_inclination: Maximum inclination angle to the z-axis that the first principal component of an instance
            can have before being discarded (in degree).
        inplace: Whether the filtering should be applied inplace to the :code:`instance_ids` array.

    Returns:
        :Tuple of two arrays:
            - Updated instance ID of each point. Points that do not belong to any instance are assigned the ID
              :code:`-1`.
            - Unique instance IDs.
    Shape:
        - :code:`xyz`: :math:`(N, 3)`
        - :code:`instance_ids`: :math:`(N)`
        - :code:`unique_instance_ids`: :math:`(I)`
        - Output: :math:`(N)`, :math:`(I')`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`I = \text{ number of instances before filtering}`
        | :math:`I' = \text{ number of instances after filtering}`
    """

    if min_explained_variance is None and max_inclination is None:
        return instance_ids, unique_instance_ids

    pca = PCA(n_components=3)

    discarded_instance_ids = []
    for label in unique_instance_ids:
        if label == -1:
            continue

        pca.fit(xyz[instance_ids == label])

        cos_angle = np.dot(pca.components_[0] / (np.linalg.norm(pca.components_[0])), np.array([0, 0, 1]))
        angle_degrees = np.degrees(np.arccos(cos_angle))

        if max_inclination is not None and angle_degrees > max_inclination:
            discarded_instance_ids.append(label)
            continue

        if min_explained_variance is not None and pca.explained_variance_ratio_[0] < min_explained_variance:
            discarded_instance_ids.append(label)

    if not inplace:
        instance_ids = instance_ids.copy()

    instance_ids[np.in1d(instance_ids, discarded_instance_ids)] = -1

    return make_labels_consecutive(  # type: ignore[return-value]
        instance_ids, ignore_id=-1, inplace=inplace, return_unique_labels=True
    )


def filter_instances_intensity(
    intensities: FloatArray,
    instance_ids: LongArray,
    unique_instance_ids: LongArray,
    min_intensity: Optional[float],
    threshold_percentile: float = 0.8,
    inplace: bool = False,
) -> Tuple[LongArray, LongArray]:
    r"""
    Removes instances, for which the specified percentile of the reflection intensities of the instance points is below
    than or equal to :code:`min_intensity`.

    Args:
        instance_ids: Instance IDs of the points.
        unique_instance_ids: Unique instance IDs.
        min_intensity: Maximum intensity at which instances are discarded. If set to :code:`None`, the instances are
            not filtered.
        threshold_percentile: Percentile of the reflection intensity values to be used for the filtering. The percentile
            must be specified as a value in :math:`[0, 1]`.
        inplace: Whether the filtering should be applied inplace to the :code:`instance_ids` array.

    Returns:
        :Tuple of two arrays:
            - Updated instance ID of each point. Points that do not belong to any instance are assigned the ID
              :code:`-1`.
            - Unique instance IDs.
    Shape:
        - :code:`instance_ids`: :math:`(N)`
        - :code:`unique_instance_ids`: :math:`(I)`
        - Output: :math:`(N)`, :math:`(I')`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`I = \text{ number of instances before filtering}`
        | :math:`I' = \text{ number of instances after filtering}`
    """

    if len(instance_ids) != len(intensities):
        raise ValueError("intensities and instance_ids must have the same length.")

    if min_intensity is None:
        return instance_ids, unique_instance_ids

    discarded_instance_ids = []
    for label in unique_instance_ids:
        if label == -1:
            continue
        if np.quantile(intensities[instance_ids == label], q=threshold_percentile) <= min_intensity:
            discarded_instance_ids.append(label)

    instance_ids[np.in1d(instance_ids, discarded_instance_ids)] = -1

    return make_labels_consecutive(  # type: ignore[return-value]
        instance_ids, ignore_id=-1, inplace=inplace, return_unique_labels=True
    )
