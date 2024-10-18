""" Transformation of input labels into consecutive integer labels. """

__all__ = ["make_labels_consecutive"]

import numpy


def make_labels_consecutive(labels: numpy.ndarray, start_id: int = 0) -> numpy.ndarray:
    """
    Transforms the input labels into consecutive integer labels starting from a given :code:`start_id`.

    Args:
        labels: An array of original labels.
        start_id: The starting ID for the consecutive labels. Defaults to zero.

    Returns:
        numpy.ndarray: An array with the transformed consecutive labels.
    """

    unique_labels = numpy.unique(labels)
    unique_labels = numpy.sort(unique_labels)
    key = numpy.arange(0, len(unique_labels))
    index = numpy.digitize(labels, unique_labels, right=True)
    labels = key[index]
    labels = labels + start_id
    return labels
