"""Calculates the point's height above the terrain."""

__all__ = ["distance_to_dtm"]

import numpy.typing as npt

from pointtree._operations_cpp import distance_to_dtm as distance_to_dtm_cpp  # type: ignore[import-not-found] # pylint: disable=import-error, no-name-in-module


def distance_to_dtm(  # pylint: disable=too-many-locals
    xyz: npt.NDArray,
    dtm: npt.NDArray,
    dtm_offset: npt.NDArray,
    dtm_resolution: float,
    allow_outside_points: bool = True,
) -> npt.NDArray:
    r"""
    Compute the height above the terrain for each point of a point cloud by subtracting the corresponding terrain height
    from the z-coordinate of the point. The terrain height for a given point is obtained by bilinearly interpolating the
    terrain heights of the four closest grid points of the digital terrain model.

    Args:
        xyz: Point coordinates of the point cloud to normalize.
        dtm: Rasterized digital terrain model.
        dtm_offset: X and y-coordinates of the top left corner of the DTM grid.
        allow_outside_points: If this option is set to :code:`True` and a point in the point cloud to be normalized is
            not in the area covered by the DTM, the height of the nearest DTM points is still determined and used for
            normalization. Otherwise, a :code:`ValueError` is thrown if points are outside the area covered by the DTM.
            Defaults to :code:`True`.

    Returns:
        Height above the terrain of each point.

    Raises:
        ValueError: If the point cloud to be normalized covers a larger base area than the DTM.

    Shape:
        - :code:`xyz`: :math:`(N, 3)`
        - :code:`dtm`: :math:`(H, W)`
        - :code:`dtm_offset`: :math:`(2)`
        - Output: :math:`(N)`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`H = \text{ extent of the DTM in grid in y-direction}`
        | :math:`W = \text{ extent of the DTM in grid in x-direction}`
    """

    # ensure that the input arrays are in column-major format
    if not xyz.flags.f_contiguous:
        xyz = xyz.copy(order="F")

    dtm = dtm.astype(xyz.dtype)
    dtm_offset = dtm_offset.astype(xyz.dtype)

    if not dtm.flags.f_contiguous:
        dtm = dtm.copy(order="F")

    dtm_offset = dtm_offset.reshape(-1, 2)
    if not dtm_offset.flags.f_contiguous:
        dtm_offset = dtm_offset.reshape(-1, 2).copy(order="F")

    return distance_to_dtm_cpp(xyz, dtm, dtm_offset, float(dtm_resolution), allow_outside_points)
