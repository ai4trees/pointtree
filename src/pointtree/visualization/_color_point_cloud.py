""" Methods for coloring point clouds. """

__all__ = ["color_semantic_segmentation", "color_instance_segmentation"]

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._color_palette import color_palette, acm_red, acm_blue
from ._hex_to_rgb import hex_to_rgb


def color_instance_segmentation(
    point_cloud: pd.DataFrame,
    instance_id_column: str,
    target_instance_id_column: Optional[str] = None,
    fp_ids: Optional[List[int]] = None,
    fn_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Sets the color of each point based on its instance ID.

    Args:
        point_cloud: Point cloud to be colored.
        instance_id_column: Name of the column containing the instance IDs.
        target_instance_id_column: Name of the column containing the ground truth instance IDs.
        fp_ids: List of the predicted tree IDs that represent false positives. Defaults to `None`, which means that
            false positives are not colored by a dedicated color.
        fn_ids: List of the ground truth tree IDs that represents false negatives. Defaults to `None`, which means that
            false negatives are not colored by a dedicated color.

    Returns:
        pandas.DataFrame: Point cloud with added or modified "r", "g", "b", "a" attributes.
    """

    instance_ids = np.unique(point_cloud[instance_id_column].to_numpy())
    instance_ids = instance_ids[instance_ids >= 0]

    colors = color_palette * math.ceil(len(instance_ids) / len(color_palette))  # pylint: disable=c-extension-no-member

    color_idx = 0

    background_gray = np.array([220, 219, 209, 127], dtype=int)

    point_cloud[["r", "g", "b", "a"]] = background_gray

    for instance_id in instance_ids:
        point_cloud.loc[
            point_cloud[instance_id_column] == instance_id,
            ["r", "g", "b", "a"],
        ] = colors[color_idx]

        color_idx += 1

    if fp_ids is not None:
        for instance_id in fp_ids:
            if instance_id == -1:
                continue
            point_cloud.loc[
                point_cloud[instance_id_column] == instance_id,
                ["r", "g", "b"],  # type: ignore[call-overload]
            ] = acm_red

    if fn_ids is not None and target_instance_id_column:
        for instance_id in fn_ids:
            if instance_id == -1:
                continue
            point_cloud.loc[  # type: ignore[call-overload]
                point_cloud[target_instance_id_column] == instance_id,
                ["r", "g", "b"],
            ] = acm_blue

    return point_cloud


def color_semantic_segmentation(
    point_cloud: pd.DataFrame, classes_to_colors: Dict[int, str], semantic_segmentation_column: str = "classification"
) -> pd.DataFrame:
    """
    Sets the color of each point based on its semantic class.

    Args:
        point_cloud: Point cloud to be colored.
        classes_to_colors: Mapping of class IDs to hex color codes.
        semantic_segmentation_column: Name of the column containing semantic class IDs. Defaults to `"classification"`.

    Returns:
        Point cloud with added or modified "r", "g", "b", "a" attributes.
    """

    background_gray = np.array([220, 219, 209, 127], dtype=int)

    point_cloud[["r", "g", "b", "a"]] = background_gray

    for class_id, hex_color in classes_to_colors.items():
        rgb_color = hex_to_rgb(hex_color)

        point_cloud.loc[
            point_cloud[semantic_segmentation_column] == class_id,
            ["r", "g", "b", "a"],
        ] = list(rgb_color)

    point_cloud[["r", "g", "b", "a"]] = (point_cloud[["r", "g", "b", "a"]] * 255).astype(int)

    return point_cloud
