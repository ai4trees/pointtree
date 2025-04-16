"""Construction of rasterized digital terrain models."""

__all__ = ["create_digital_terrain_model"]

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from pointtorch.operations.numpy import voxel_downsampling
from scipy.spatial import KDTree


def create_digital_terrain_model(  # pylint: disable=too-few-public-methods, too-many-locals
    terrain_coords: npt.NDArray,
    grid_resolution: float,
    k: int,
    p: float,
    voxel_size: Optional[float] = None,
    num_workers: int = 1,
) -> Tuple[npt.NDArray, npt.NDArray]:
    r"""
    Constructs a rasterized digital terrain model (DTM) from a set of terrain points. The DTM is constructed by
    creating a grid of regularly arranged DTM points and interpolating the height of the :math:`k` closest terrain
    points for each DTM point on the grid. In the interpolation, terrain points :math:`x_t` are weighted with a
    factor proportional to a power of :math:`p` of their inverse distance to the corresponding DTM point
    :math:`x_{dtm}`, i.e., :math:`\frac{1}{||(x_{dtm} - x_{t})||^p}`. If there are terrain points whose distance to the
    DTM point is zero, only these points are used to calculate the DTM height and more distant points are ignored.
    Before constructing the DTM, the terrain points can optionally be downsampled using voxel-based subsampling.

    Args:
        terrain_coords: Coordinates of the terrain points from which to construct the DTM.
        grid_resolution: Resolution of the DTM grid (in meter).
        k: Number of terrain points between which interpolation is performed to obtain the terrain height of a DTM
            point.
        p: Power :math:`p` for inverse-distance weighting in the interpolation of terrain points.
        voxel_size: Voxel size with which the terrain points are downsampled before the DTM is
            created. If set to :code:`None`, no downsampling is performed. Defaults to :code:`None`.
        num_workers: Number of workers to use for parallel processing. If :code:`workers` is set to -1, all CPU threads
            are used. Defaults to :code:`1`.

    Returns:
        Tuple of two arrays. The first is the DTM. The second contains the x- and y-coordinate of the top left
        corner of the DTM grid.

    Shape:
        - :code:`terrain_coords`: :math:`(N, 3)`
        - Output: Tuple of two arrays. The first has shape :math:`(H, W)` and second has shape :math:`(2)`.

        | where
        |
        | :math:`N = \text{ number of terrain points}`
        | :math:`H = \text{ extent of the DTM grid in y-direction}`
        | :math:`W = \text{ extent of the DTM grid in x-direction}`
    """

    if voxel_size is not None:
        terrain_coords, _, _ = voxel_downsampling(terrain_coords, voxel_size=voxel_size)

    min_coords = terrain_coords[:, :2].min(axis=0)
    max_coords = terrain_coords[:, :2].max(axis=0)

    dtm_grid_offset = np.floor(min_coords / grid_resolution)
    dtm_size = np.floor(max_coords / grid_resolution) - dtm_grid_offset + 1

    dtm_points_x, dtm_points_y = np.meshgrid(np.arange(dtm_size[0]), np.arange(dtm_size[1]))
    dtm_points_x = np.expand_dims(dtm_points_x, axis=-1)
    dtm_points_y = np.expand_dims(dtm_points_y, axis=-1)

    dtm_points = (dtm_grid_offset + np.concatenate([dtm_points_x, dtm_points_y], axis=-1)) * grid_resolution

    kd_tree = KDTree(terrain_coords[:, :2])
    neighbor_distances, neighbor_indices = kd_tree.query(dtm_points, k=min(k, len(terrain_coords)), workers=num_workers)
    if isinstance(neighbor_distances, float):
        neighbor_distances = np.array(neighbor_distances, dtype=terrain_coords.dtype)
    else:
        neighbor_distances = neighbor_distances.astype(terrain_coords.dtype)
    if isinstance(neighbor_indices, int):
        neighbor_indices = np.array(neighbor_indices, dtype=np.int64)

    # ignore divide by zero warnings for this operation since those values are replaced afterwards
    with np.errstate(divide="ignore"):
        neighbor_weights = 1 / (neighbor_distances**p)

    is_exact_match = neighbor_distances == 0

    neighbor_weights[is_exact_match.any(axis=-1)] = 0
    neighbor_weights[is_exact_match] = 1
    neighbor_weights /= neighbor_weights.sum(axis=-1, keepdims=True)

    dtm = (terrain_coords[neighbor_indices, 2] * neighbor_weights).sum(axis=-1)

    return dtm, dtm_grid_offset * grid_resolution
