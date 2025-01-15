""" Calculates the point's height above the terrain. """

__all__ = ["distance_to_dtm"]

import numpy as np
import numpy.typing as npt


def distance_to_dtm(  # pylint: disable=too-many-locals
    coords: npt.NDArray,
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
        coords: Point coordinates of the point cloud to normalize.
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
        - :code:`coords`: :math:`(N, 3)`
        - :code:`dtm`: :math:`(H, W)`
        - :code:`dtm_offset`: :math:`(2)`
        - Output: :math:`(N)`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`H = \text{ extent of the DTM in grid in y-direction}`
        | :math:`W = \text{ extent of the DTM in grid in x-direction}`
    """

    grid_positions = (coords[:, :2] - dtm_offset) / dtm_resolution
    if allow_outside_points:
        grid_positions = np.clip(grid_positions, 0, np.array(dtm.shape, dtype=coords.dtype) - 1)
    grid_indices = np.floor(grid_positions)
    grid_fractions = grid_positions - grid_indices
    grid_indices = grid_indices.astype(np.int64)

    if not allow_outside_points and (
        (grid_positions < 0).any()
        or (grid_positions[:, 0] >= dtm.shape[1]).any()
        or (grid_positions[:, 1] >= dtm.shape[0]).any()
    ):
        raise ValueError("The DTM does not completely cover the point cloud to be normalized.")

    height_1 = dtm[grid_indices[:, 1], grid_indices[:, 0]]
    height_2 = dtm[grid_indices[:, 1], np.minimum(grid_indices[:, 0] + 1, dtm.shape[1] - 1)]
    height_3 = dtm[np.minimum(grid_indices[:, 1] + 1, dtm.shape[0] - 1), grid_indices[:, 0]]
    height_4 = dtm[
        np.minimum(grid_indices[:, 1] + 1, dtm.shape[0] - 1),
        np.minimum(grid_indices[:, 0] + 1, dtm.shape[1] - 1),
    ]

    interp_height_1 = height_1 * (1 - grid_fractions[:, 0]) + height_2 * (grid_fractions[:, 0])
    interp_height_2 = height_3 * (1 - grid_fractions[:, 0]) + height_4 * (grid_fractions[:, 0])
    terrain_height = interp_height_1 * (1 - grid_fractions[:, 1]) + interp_height_2 * (grid_fractions[:, 1])

    return coords[:, 2] - terrain_height
