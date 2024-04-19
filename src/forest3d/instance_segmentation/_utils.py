""" Utilities for tree instance segmentation. """

__all__ = ["remap_instance_ids"]

from typing import Tuple

import numpy as np


def remap_instance_ids(instance_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Remaps instance IDs that may not be continuous to a continuous range.

    Args:
        instance_ids: Instance IDs to remap.

    Returns:
        Tuple of two arrays. The first contains the remapped instance IDs. The second contains the unique instance IDs
        after remapping.

    Shape:
        - :code:`instance_ids`: :math:`(N)`
        - Output: Tuple of two arrays. The first has shape :math:`(N)` and the second :math:`(I)`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`I = \text{ number of instances}`
    """
    if len(instance_ids) == 0:
        return instance_ids, np.empty(0, dtype=np.int64)

    unique_instance_ids = np.unique(instance_ids)
    unique_instance_ids = unique_instance_ids[unique_instance_ids != -1]

    # map to continuous range of IDs
    if len(unique_instance_ids) == 0 or unique_instance_ids.max() >= len(unique_instance_ids):
        updated_instance_ids = np.copy(instance_ids)
        for new_istance_id, instance_id in enumerate(unique_instance_ids):
            updated_instance_ids[instance_ids == instance_id] = new_istance_id
        unique_instance_ids = np.arange(len(unique_instance_ids), dtype=np.int64)
    else:
        updated_instance_ids = instance_ids

    return updated_instance_ids, unique_instance_ids
