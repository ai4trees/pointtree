""" Voxel-based downsampling of a point cloud. """

__all__ = ["voxel_downsampling"]

from typing import Literal, Optional, Tuple

from numba_kdtree import KDTree
import numpy
import torch

from ._knn_search import knn_search
from ._ravel_index import ravel_multi_index


def voxel_downsampling(  # pylint: disable=too-many-locals, too-many-statements
    points: numpy.ndarray,
    voxel_size: float,
    point_aggregation: Literal["nearest_neighbor", "random"] = "random",
    preserve_order: bool = True,
    start: Optional[numpy.ndarray] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Voxel-based downsampling of a point cloud.

    Args:
        points: The point cloud to downsample.
        voxel_size: The size of the voxels used for downsampling. If :code:`voxel_size` is set to zero or less, no
            downsampling is applied.
        point_aggregation: Method to be used to aggregate the points within the same voxel. Defaults to
            `nearest_neighbor`. `"nearest_neighbor"`: The point closest to the voxel center is selected. `"random"`:
            One point is randomly sampled from the voxel.
        preserve_order: If set to `True`, the point order is preserved during downsampling. This means that for any two
            points included in the downsampled point cloud, the point that is first in the original point cloud is
            also first in the downsampled point cloud. Defaults to `True`.
        start: Coordinates of a point at which the voxel grid is to be aligned, i.e., the grid is placed so that
            :code:`start` is at a corner point of a voxel. Defaults to `None`, which means that the grid is aligned at
            the coordinate origin.

    Returns:
        Tuple of two arrays. The first contains the points remaining after downsampling. The second contains the \
        indices of the points remaining after downsampling within the original point cloud.^

    Raises:
        ValueError: If `start` is not `None` and has an invalid shape.

    Shape:
        - :code:`points`: :math:`(N, 3 + D)`.
        - :code:`start`: :math:`(3)`
        - Output: Tuple of two arrays. The first has shape :math:`(N', 3 + D)` and the second :math:`(N')`.

          | where
          |
          | :math:`N = \text{ number of points before downsampling}`
          | :math:`N' = \text{ number of points after downsampling}`
          | :math:`D = \text{ number of feature channels excluding coordinate channels }`
    """

    if voxel_size <= 0:
        return points, numpy.arange(len(points), dtype=numpy.int64)

    if start is None:
        start_coords = numpy.array([0.0, 0.0, 0.0])
    else:
        if start.shape != numpy.array([3]):
            raise ValueError(f"The shape of the 'start' array is invalid: {start.shape}. ")
        start_coords = start

    shifted_points = points[:, :3] - start_coords
    voxel_indices = numpy.floor_divide(shifted_points, voxel_size).astype(numpy.int64)
    shift = voxel_indices.min(axis=0)
    voxel_indices -= shift
    shifted_points -= shift.astype(float) * voxel_size

    if point_aggregation == "random":
        _, selected_indices = numpy.unique(voxel_indices, axis=0, return_index=True)
        if preserve_order:
            selected_indices.sort()
        return points[selected_indices], selected_indices

    # for the "nearest_neighbor" option, a two-stage approach is used to select the point closest to the center of each
    # filled voxel: first the entire point cloud is searched for the nearest neighbor of each voxel center
    # if the nearest neighbor of a voxel center is outside the voxel, the search is refined and only the points within
    # the voxel are searched for the nearest neighbor of the voxel center
    # this two-stage approach is used because the first step can be implemented for large point clouds using
    # a CPU-based KD tree
    # in the second step, the number of points to be processed is usually much smaller, so the second step can be run on
    # GPU if one is available

    filled_voxel_indices, inverse_indices, point_count_per_voxel = numpy.unique(
        voxel_indices, axis=0, return_inverse=True, return_counts=True
    )

    kd_tree = KDTree(shifted_points)
    voxel_centers = filled_voxel_indices.astype(float) * voxel_size + 0.5 * voxel_size
    _, selected_indices, _ = kd_tree.query(voxel_centers, k=1)
    selected_indices = selected_indices.flatten()

    # test which neighbors are within the voxel
    invalid_selection_mask = (voxel_indices[selected_indices] != filled_voxel_indices).any(axis=1)

    if invalid_selection_mask.sum() > 0:
        # save valid selections from the first step
        selected_indices = selected_indices[numpy.logical_not(invalid_selection_mask)]

        # refine remaining selections
        points_to_refine_mask = invalid_selection_mask[inverse_indices]

        device = torch.device("cpu")

        # check if there is a GPU with sufficient memory to run the second step on GPU
        # if this is not the case, the second step is run on CPU
        if torch.cuda.is_available():
            available_memory = torch.cuda.mem_get_info(device=torch.device("cuda:0"))[0]
            float_size = torch.empty((0,)).float().element_size()
            long_size = torch.empty((0,)).long().element_size()
            approx_required_memory = points_to_refine_mask.sum() * (
                3 * float_size + long_size
            ) + invalid_selection_mask.sum() * (3 * float_size + 2 * long_size)

            if available_memory > approx_required_memory:
                device = torch.device("cuda:0")

        support_points = torch.from_numpy(shifted_points[points_to_refine_mask]).float().to(device)
        del shifted_points
        voxel_indices_torch = torch.from_numpy(voxel_indices[points_to_refine_mask]).long().to(device)
        del voxel_indices
        voxel_centers_torch = torch.from_numpy(voxel_centers[invalid_selection_mask]).float().to(device)
        del voxel_centers
        filled_voxel_indices_torch = torch.from_numpy(filled_voxel_indices[invalid_selection_mask]).long().to(device)
        del filled_voxel_indices

        point_cloud_sizes_support_points = (
            torch.from_numpy(point_count_per_voxel[invalid_selection_mask]).long().to(device)
        )
        point_cloud_sizes_query_points = torch.ones(len(voxel_centers_torch), dtype=torch.long, device=device)

        dimensions = voxel_indices_torch.amax(dim=0) + 1
        batch_indices_support_points = ravel_multi_index(voxel_indices_torch.transpose(1, 0), dimensions)
        batch_indices_query_points = ravel_multi_index(filled_voxel_indices_torch.transpose(1, 0), dimensions)
        _, inverse_indices = torch.unique(
            torch.concat([batch_indices_support_points, batch_indices_query_points]), return_inverse=True
        )
        batch_indices_support_points = inverse_indices[: len(batch_indices_support_points)]
        batch_indices_query_points = inverse_indices[len(batch_indices_support_points) :]

        batch_indices_support_points, sorting_indices = torch.sort(batch_indices_support_points)

        voxel_indices_torch = voxel_indices_torch[sorting_indices]
        support_points = support_points[sorting_indices]
        point_indices = torch.arange(len(points), dtype=torch.long, device=device)
        point_indices = point_indices[points_to_refine_mask][sorting_indices]

        refined_indices_torch, _ = knn_search(
            support_points,
            voxel_centers_torch,
            batch_indices_support_points,
            batch_indices_query_points,
            point_cloud_sizes_support_points,
            point_cloud_sizes_query_points,
            k=1,
        )

        refined_indices = point_indices[refined_indices_torch.flatten()].cpu().numpy()

        selected_indices = numpy.concatenate([selected_indices, refined_indices])

    if preserve_order:
        selected_indices.sort()

    return points[selected_indices], selected_indices
