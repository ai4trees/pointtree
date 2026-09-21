"""Computation of tree attributes from individual tree point clouds."""

__all__ = ["tree_attributes", "crown_volume", "crown_width", "tree_height", "stem_direction"]


from typing import Any, Dict, List, Literal, Optional

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA


def tree_attributes(
    tree_xyz: npt.NDArray,
    attributes: Optional[
        List[Literal["crown_volume", "crown_width", "tree_height", "stem_direction"]]
    ] = None,
    classification: Optional[npt.NDArray] = None,
    ground_height: Optional[float] = None,
    stem_class_ids: Optional[List[int]] = None,
    leaf_class_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Computes attributes for a single tree.

    Args:
        tree_xyz: Coordinates of all points belonging to the tree.
        attributes: Names of the attributes to compute. If :code:`None`, all supported attributes are computed.
        classification: Semantic class ID for each point in :code:`tree_xyz`. Used together with
            :code:`stem_class_ids` and :code:`leaf_class_ids` to restrict the points used to compute the stem and
            crown attributes, respectively. If :code:`None`, all points of :code:`tree_xyz` are used for every
            attribute.
        ground_height: Height of the ground surface underneath the tree, used to compute the tree height. If
            :code:`None`, the tree height is computed as the difference between the maximum and minimum
            z-coordinate of :code:`tree_xyz`.
        stem_class_ids: Class IDs that identify stem points. Used together with :code:`classification` to select the
            points passed to :code:`stem_diameter` and :code:`stem_direction`. If :code:`None`, or if
            :code:`classification` is :code:`None`, all points of :code:`tree_xyz` are used.
        branch_class_ids: Class IDs that identify branch points. Currently unused.
        leaf_class_ids: Class IDs that identify leaf points. Used together with :code:`classification` to select the
            points passed to :code:`crown_volume` and :code:`crown_width`. If :code:`None`, or if
            :code:`classification` is :code:`None`, all points of :code:`tree_xyz` are used.

    Returns:
        Dictionary mapping the name of each computed attribute to its value.
    """

    tree_attributes: Dict[str, Any] = {}

    stem_xyz = tree_xyz
    crown_xyz = tree_xyz
    if classification is not None and stem_class_ids is not None:
        stem_xyz = tree_xyz[np.isin(classification, stem_class_ids)]
    if classification is not None and leaf_class_ids is not None:
        crown_xyz = tree_xyz[np.isin(classification, leaf_class_ids)]

    if attributes is None or "crown_volume" in attributes:
        tree_attributes["crown_volume"] = crown_volume(crown_xyz)

    if attributes is None or "crown_width" in attributes:
        tree_attributes["crown_width"] = crown_width(crown_xyz)

    if attributes is None or "tree_height" in attributes:
        tree_attributes["tree_height"] = tree_height(tree_xyz, ground_height=ground_height)

    if attributes is None or "stem_direction" in attributes:
        tree_attributes["stem_direction"] = stem_direction(stem_xyz)

    return tree_attributes


def crown_volume(crown_xyz: npt.NDArray, voxel_size: float = 0.5) -> float:
    """
    Computes the crown volume by summing the volumes of the 2.5D voxel columns spanned by the crown points.

    Args:
        crown_xyz: Coordinates of the points belonging to the tree crown.
        voxel_size: Edge length of the (square) voxel columns in meters.

    Returns:
        Crown volume in cubic meters. :code:`0.0` if :code:`crown_xyz` is empty.
    """

    if len(crown_xyz) == 0:
        return 0.0

    bins = np.floor(crown_xyz / voxel_size).astype(np.int64)
    volume = 0.0

    sorting_indices = np.lexsort((bins[:, 1], bins[:, 0]))
    bins_sorted = bins[sorting_indices]
    z_sorted = crown_xyz[sorting_indices, 2]

    # split into contiguous (bin_x, bin_y) columns
    column_keys = bins_sorted[:, :2]
    column_boundaries = np.any(column_keys[1:] != column_keys[:-1], axis=1)
    split_idx = np.flatnonzero(column_boundaries) + 1

    for column in np.split(z_sorted, split_idx):
        volume += voxel_size * voxel_size * (column.max() - column.min())

    return float(volume)


def crown_width(crown_xyz: npt.NDArray) -> float:
    """
    Computes the crown width as the largest extent of the crown points along the x- or y-axis.

    Args:
        crown_xyz: Coordinates of the points belonging to the tree crown.

    Returns:
        Crown width in meters. :code:`0.0` if :code:`crown_xyz` is empty.
    """

    if len(crown_xyz) == 0:
        return 0.0

    return (crown_xyz[:, :2].max(axis=0) - crown_xyz[:, :2].min(axis=0)).max()


def tree_height(xyz: npt.NDArray, ground_height: Optional[float] = None) -> float:
    """
    Computes the tree height.

    Args:
        xyz: Coordinates of all points belonging to the tree.
        ground_height: Height of the ground surface underneath the tree. If provided, the tree height is computed
            as the difference between the maximum z-coordinate of :code:`xyz` and :code:`ground_height`. If
            :code:`None`, the tree height is computed as the difference between the maximum and minimum
            z-coordinate of :code:`xyz`.

    Returns:
        Tree height in meters. :code:`0.0` if :code:`xyz` is empty.
    """

    if len(xyz) == 0:
        return 0.0

    if ground_height is not None:
        return xyz[:, 2].max() - ground_height

    return xyz[:, 2].max() - xyz[:, 2].min()


def stem_direction(stem_xyz: npt.NDArray) -> npt.NDArray:
    """
    Computes the dominant 3D stem direction using principal component analysis (PCA).

    Args:
        stem_xyz: Coordinates of the points belonging to the tree stem.

    Returns:
        Unit vector describing the dominant stem direction, oriented so that its z-component is non-negative. An
        array of NaN values if :code:`stem_xyz` contains fewer than two points.
    """

    if len(stem_xyz) < 2:
        return np.array([np.nan, np.nan, np.nan])

    direction = PCA(n_components=1).fit(stem_xyz).components_[0]

    if direction[2] < 0:
        direction = -direction

    return direction
