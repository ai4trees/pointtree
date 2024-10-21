""" Transformation of input labels into consecutive integer labels. """

__all__ = ["make_labels_consecutive"]

from typing import Optional, Tuple, Union

import numpy


def make_labels_consecutive(
    labels: numpy.ndarray,
    start_id: int = 0,
    ignore_id: Optional[int] = None,
    inplace: bool = False,
    return_unique_labels: bool = False,
) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    """
    Transforms the input labels into consecutive integer labels starting from a given :code:`start_id`.

    Args:
        labels: An array of original labels.
        start_id: The starting ID for the consecutive labels. Defaults to zero.
        ignore_id: A label ID that should not be changed when transforming the labels.
        inplace: Whether the transformation should be applied inplace to the :code:`labels` array. Defaults to
            :code:`False`.
        return_unique_labels: Whether the unique labels after applying the transformation (excluding :code:`ignore_id`)
            should be returned. Defaults to :code:`False`.

    Returns:
        An array with the transformed consecutive labels. If :code:`return_unique_labels` is set to :code:`True`, a
        tuple of two arrays is returned, where the second array contains the unique labels after the transformation.
    """

    if len(labels) == 0:
        if return_unique_labels:
            return labels, numpy.empty_like(labels)
        return labels

    if not inplace:
        labels = labels.copy()

    labels_to_remap: Union[numpy.ndarray, numpy.ma.MaskedArray]
    if ignore_id is not None:
        mask = labels == ignore_id
        labels_to_remap = numpy.ma.masked_array(labels, mask)
    else:
        labels_to_remap = labels

    unique_labels = numpy.unique(labels_to_remap)
    unique_labels = numpy.sort(unique_labels)
    key = numpy.arange(0, len(unique_labels))
    index = numpy.digitize(labels_to_remap, unique_labels, right=True)
    labels_to_remap[:] = key[index]
    labels_to_remap = labels_to_remap + start_id

    if return_unique_labels:
        return labels, key

    return labels
