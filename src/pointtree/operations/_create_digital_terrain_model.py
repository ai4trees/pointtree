"""Construction of rasterized digital terrain models."""

__all__ = ["create_digital_terrain_model"]

from typing import Optional, Tuple, cast

import numpy as np
from pointtorch.operations.numpy import voxel_downsampling
from scipy.spatial import KDTree

from pointtree.type_aliases import FloatArray


def create_digital_terrain_model(  # pylint: disable=too-few-public-methods, too-many-locals
    terrain_xyz: FloatArray,
    grid_resolution: float,
    k: int,
    p: float,
    voxel_size: Optional[float] = None,
    num_workers: int = 1,
) -> Tuple[FloatArray, FloatArray]:
    r"""
    Constructs a rasterized digital terrain model (DTM) from a set of terrain points. The DTM is constructed by
    creating a grid of regularly arranged DTM points and interpolating the height of the :math:`k` closest terrain
    points for each DTM point on the grid. In the interpolation, terrain points :math:`x_t` are weighted with a
    factor proportional to a power of :math:`p` of their inverse distance to the corresponding DTM point
    :math:`x_{dtm}`, i.e., :math:`\frac{1}{||(x_{dtm} - x_{t})||^p}`. If there are terrain points whose distance to the
    DTM point is zero, only these points are used to calculate the DTM height and more distant points are ignored.
    Before constructing the DTM, the terrain points can optionally be downsampled using voxel-based subsampling.

    Args:
        terrain_xyz: Coordinates of the terrain points from which to construct the DTM.
        grid_resolution: Resolution of the DTM grid (in meter).
        k: Number of terrain points between which interpolation is performed to obtain the terrain height of a DTM
            point.
        p: Power :math:`p` for inverse-distance weighting in the interpolation of terrain points.
        voxel_size: Voxel size with which the terrain points are downsampled before the DTM is
            created. If set to :code:`None`, no downsampling is performed.
        num_workers: Number of workers to use for parallel processing. If :code:`workers` is set to -1, all CPU threads
            are used.

    Returns:
        :Tuple of two arrays:
            - DTM
            - X- and y-coordinate of the top left corner of the DTM grid

    Shape:
        - :code:`terrain_xyz`: :math:`(N, 3)`
        - Output: :math:`(H, W)`, :math:`(2)`

        | where
        |
        | :math:`N = \text{ number of terrain points}`
        | :math:`H = \text{ extent of the DTM grid in y-direction}`
        | :math:`W = \text{ extent of the DTM grid in x-direction}`
    """

    if voxel_size is not None:
        terrain_xyz, _, _ = voxel_downsampling(terrain_xyz, voxel_size=voxel_size)

    min_coords = terrain_xyz[:, :2].min(axis=0)
    max_coords = terrain_xyz[:, :2].max(axis=0)

    dtm_grid_offset = np.floor(min_coords / grid_resolution)
    dtm_size = np.floor(max_coords / grid_resolution) - dtm_grid_offset + 1

    dtm_points_x, dtm_points_y = np.meshgrid(np.arange(dtm_size[0]), np.arange(dtm_size[1]))
    dtm_points_x = np.expand_dims(dtm_points_x, axis=-1)
    dtm_points_y = np.expand_dims(dtm_points_y, axis=-1)

    dtm_points = (dtm_grid_offset + np.concatenate([dtm_points_x, dtm_points_y], axis=-1)) * grid_resolution

    kd_tree = KDTree(terrain_xyz[:, :2])
    neighbor_distances, neighbor_indices = kd_tree.query(dtm_points, k=min(k, len(terrain_xyz)), workers=num_workers)
    neighbor_distances = cast(np.ndarray, neighbor_distances)
    neighbor_indices = cast(np.ndarray, neighbor_indices)
    neighbor_distances = neighbor_distances.astype(terrain_xyz.dtype)

    # ignore divide by zero warnings for this operation since those values are replaced afterwards
    with np.errstate(divide="ignore"):
        neighbor_weights = 1 / (neighbor_distances**p)

    is_exact_match = neighbor_distances == 0

    neighbor_weights[is_exact_match.any(axis=-1)] = 0
    neighbor_weights[is_exact_match] = 1
    neighbor_weights /= neighbor_weights.sum(axis=-1, keepdims=True)

    dtm = (terrain_xyz[neighbor_indices, 2] * neighbor_weights).sum(axis=-1)

    return dtm, dtm_grid_offset * grid_resolution
