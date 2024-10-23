""" Point cloud normalization. """

__all__ = ["normalize_height"]

import numpy as np
import numpy.typing as npt


def normalize_height(  # pylint: disable=too-many-locals
    coords: npt.NDArray[np.float64],
    dtm: npt.NDArray[np.float64],
    dtm_offset: npt.NDArray[np.float64],
    dtm_resolution: float,
    inplace: bool = False,
) -> npt.NDArray[np.float64]:
    r"""
    Normalizes the height of a point cloud by subtracting the corresponding terrain height from the z-coordinate of
    each point. The terrain height for a given point is obtained by bilinearly interpolating the terrain heights of
    the four closest grid points of the digital terrain model.

    Args:
        coords: Point coordinates of the point cloud to normalize.
        dtm: Rasterized digital terrain model.
        dtm_offset: X and y-coordinates of the top left corner of the DTM grid.
        inplace: Whether the normalization should be applied in place to the :code:`coords` array. Defaults to
            :code:`False`.

    Returns:
        Height-normalized point cloud.

    Shape:
        - :code:`coords`: :math:`(N, 3)`
        - :code:`dtm`: :math:`(H, W)`
        - :code:`dtm_offset`: :math:`(2)`
        - Output: :math:`(N, 3)`

        | where
        |
        | :math:`N = \text{ number of points}`
        | :math:`H = \text{ extent of the DTM in grid in y-direction}`
        | :math:`W = \text{ extent of the DTM in grid in x-direction}`
    """

    grid_positions = (coords[:, :2] - dtm_offset) / dtm_resolution
    grid_positions = np.clip(grid_positions, 0, np.array(dtm.shape) - 1)
    grid_indices = np.floor(grid_positions).astype(np.int64)
    grid_fractions = grid_positions - grid_indices

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

    if inplace:
        normalized_coords = coords
    else:
        normalized_coords = coords.copy()
    normalized_coords[:, 2] -= terrain_height

    return normalized_coords
