""" Different implementations of kNN search. """

__all__ = ["knn_search", "knn_search_pytorch3d", "knn_search_torch_cluster"]

from typing import Tuple

import torch
import torch_cluster

from pointtree.config import pytorch3d_is_available
from ._pack_batch import pack_batch


def knn_search(
    coords_support_points: torch.Tensor,
    coords_query_points: torch.Tensor,
    batch_indices_support_points: torch.Tensor,
    batch_indices_query_points: torch.Tensor,
    point_cloud_sizes_support_points: torch.Tensor,
    point_cloud_sizes_query_points: torch.Tensor,
    k: int,
    return_sorted: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computes the indices of `k` nearest neighbors. Decides between different implementations:

    - **Implementation from PyTorch3D:** This implementation is always used if PyTorch3D is installed because its more \
      efficient in terms of runtime and memory consumption than the other available implementations.
    - **Implementation from torch-cluster:** This implementation is used when PyTorch3D is not installed. It is \
      similar to the PyTorch3D implementation but is slighlty slower.

    Args:
        coords_support_points: Coordinates of the support points to be searched for neighbors.
        coords_query_points: Coordinates of the query points.
        batch_indices_support_points: Indices indicating to which point cloud in the batch each support point belongs.
        batch_indices_query_points: Indices indicating to which point cloud in the batch each query point belongs.
        point_cloud_sizes_support_points: Number of points in each point cloud in the batch of support points.
        point_cloud_sizes_query_points: Number of points in each point cloud in the batch of query points.
        k: The number of nearest neighbors to search.
        return_sorted: Whether the returned neighbors should be sorted by their distance to the query point. Defaults to
            `True`. Setting it to `False` can improve performance for some implementations.

    Returns:
        Tuple of two tensors. The first tensor contains the indices of the neighbors of each query point. The
        second tensor contains the distances between the neighbors and the query points.

    Shape:
        - :attr:`coords_support_points`: :math:`(N, 3)`
        - :attr:`coords_query_points`: :math:`(N', 3)`
        - :attr:`batch_indices_support_points`: :math:`(N)`
        - :attr:`batch_indices_query_points`: :math:`(N')`
        - :attr:`point_cloud_sizes_support_points`: :math:`(B)`
        - :attr:`point_cloud_sizes_query_points`: :math:`(B)`
        - Output: Tuple of two tensors, both with shape :math:`(N', k)` if :attr:`k` :math:`\leq n_{max}`, \
          otherwise :math:`(N', n_{max})`.

          | where
          |
          | :math:`B = \text{ batch size}`
          | :math:`N = \text{ number of support points}`
          | :math:`N' = \text{ number of query points}`
          | :math:`n_{max} = \text{ maximum number of neighbors a query point has}`
    """

    if pytorch3d_is_available():
        return knn_search_pytorch3d(
            coords_support_points,
            coords_query_points,
            batch_indices_query_points,
            point_cloud_sizes_support_points,
            point_cloud_sizes_query_points,
            k,
            return_sorted=return_sorted,
        )

    return knn_search_torch_cluster(
        coords_support_points,
        coords_query_points,
        batch_indices_support_points,
        batch_indices_query_points,
        point_cloud_sizes_support_points,
        k,
    )


def knn_search_pytorch3d(
    coords_support_points: torch.Tensor,
    coords_query_points: torch.Tensor,
    batch_indices_query_points: torch.Tensor,
    point_cloud_sizes_support_points: torch.Tensor,
    point_cloud_sizes_query_points: torch.Tensor,
    k: int,
    return_sorted: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computes the indices of `k` nearest neighbors. This implementation is based on
    `PyTorch3D's knn_points <https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.knn_points>`_
    function.

    The GPU-based KNN search implementation from PyTorch3D launches one CUDA thread per query point and each thread then
    loops through all the support points to find the k-nearest neighbors. It is similar to the torch-cluster
    implementation but it requires input batches of regular shape. Therefore, the variable size point cloud batches are
    packed into regular shaped batches before passing them to PyTorch3D.

    Args:
        coords_support_points: Coordinates of the support points to be searched for neighbors.
        coords_query_points: Coordinates of the query points.
        batch_indices_query_points: Indices indicating to which point cloud in the batch each query point belongs.
        point_cloud_sizes_support_points: Number of points in each point cloud in the batch of support points.
        point_cloud_sizes_query_points: Number of points in each point cloud in the batch of query points.
        k: The number of nearest neighbors to search.
        return_sorted: Whether the returned neighbors should be sorted by their distance to the query point. Defaults to
            `True`. Setting it to `False` can improve performance for some implementations.

    Returns:
        Tuple of two tensors. The first tensor contains the indices of the neighbors of each query point. The
        second tensor contains the distances between the neighbors and the query points.

    Shape:
        - :attr:`coords_support_points`: :math:`(N, 3)`
        - :attr:`coords_query_points`: :math:`(N', 3)`
        - :attr:`batch_indices_query_points`: :math:`(N')`
        - :attr:`point_cloud_sizes_support_points`: :math:`(B)`
        - :attr:`point_cloud_sizes_query_points`: :math:`(B)`
        - Output: Tuple of two tensors, both with shape :math:`(N', k)` if :attr:`k` :math:`\leq n_{max}`, \
          otherwise :math:`(N', n_{max})`.

          | where
          |
          | :math:`B = \text{ batch size}`
          | :math:`N = \text{ number of support points}`
          | :math:`N' = \text{ number of query points}`
          | :math:`n_{max} = \text{ maximum number of neighbors a query point has}`
    """

    # the import is placed inside this method because the Pytorch3D package is an optional dependency and might not be
    # available on all systems
    from pytorch3d.ops import knn_points  # pylint: disable=import-error,import-outside-toplevel

    invalid_neighbor_index = len(coords_support_points)

    k = min(k, int(point_cloud_sizes_support_points.amax().item()))

    coords_support_points = coords_support_points.float()
    coords_query_points = coords_query_points.float()

    coords_query_points, mask_query_points = pack_batch(
        coords_query_points, point_cloud_sizes_query_points, fill_value=-torch.inf
    )
    coords_support_points, _ = pack_batch(coords_support_points, point_cloud_sizes_support_points, fill_value=torch.inf)

    neighbor_distances, neighbor_indices, _ = knn_points(
        coords_query_points,
        coords_support_points,
        point_cloud_sizes_query_points,
        point_cloud_sizes_support_points,
        K=k,
        return_sorted=return_sorted,
        return_nn=False,
    )

    # PyTorch3D return squared distances
    neighbor_distances = torch.sqrt(neighbor_distances)

    # flatten packed batch
    batch_start_index = torch.cumsum(point_cloud_sizes_support_points, dim=0) - point_cloud_sizes_support_points
    batch_start_index = batch_start_index.unsqueeze(-1).unsqueeze(-1)  # convert to shape (B, 1, 1)
    neighbor_indices += batch_start_index

    neighbor_indices = neighbor_indices[mask_query_points].view(-1, k)  # (N' k)
    neighbor_distances = neighbor_distances[mask_query_points].view(-1, k)

    if not (point_cloud_sizes_support_points >= k).all():
        invalid_neighbor_mask = torch.arange(
            k, dtype=point_cloud_sizes_support_points.dtype, device=point_cloud_sizes_support_points.device
        ).unsqueeze(
            0
        )  # (1, k)
        invalid_neighbor_mask = invalid_neighbor_mask.repeat((len(batch_indices_query_points), 1))  # (N', k)
        max_neighbors = (point_cloud_sizes_support_points[batch_indices_query_points]).unsqueeze(-1)
        invalid_neighbor_mask = invalid_neighbor_mask >= max_neighbors
        neighbor_indices[invalid_neighbor_mask] = invalid_neighbor_index
        neighbor_distances[invalid_neighbor_mask] = torch.inf

    return neighbor_indices, neighbor_distances


def knn_search_torch_cluster(  # pylint: disable=too-many-locals
    coords_support_points: torch.Tensor,
    coords_query_points: torch.Tensor,
    batch_indices_support_points: torch.Tensor,
    batch_indices_query_points: torch.Tensor,
    point_cloud_sizes_support_points: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computes the indices of `k` nearest neighbors. This implementation is based on the
    `knn method from torch-cluster <https://github.com/rusty1s/pytorch_cluster>`_.

    The GPU-based KNN search implementation from torch-cluster launches one CUDA thread per query point and each thread
    then loops through all the support points to find the k-nearest neighbors. It is similar to the PyTorch3D
    implementation but can handle variable size point clouds directly.

    Args:
        coords_support_points: Coordinates of the support points to be searched for neighbors.
        coords_query_points: Coordinates of the query points.
        batch_indices_support_points: Indices indicating to which point cloud in the batch each support point belongs.
        batch_indices_query_points: Indices indicating to which point cloud in the batch each query point belongs.
        point_cloud_sizes_support_points: Number of points in each point cloud in the batch of support points.
        k: The number of nearest neighbors to search.

    Returns:
        Tuple of two tensors. The first tensor contains the indices of the neighbors of each query point. The
        second tensor contains the distances between the neighbors and the query points.

    Shape:
        - :attr:`coords_support_points`: :math:`(N, 3)`
        - :attr:`coords_query_points`: :math:`(N', 3)`
        - :attr:`batch_indices_support_points`: :math:`(N)`
        - :attr:`batch_indices_query_points`: :math:`(N')`
        - :attr:`point_cloud_sizes_support_points`: :math:`(B)`
        - Output: Tuple of two tensors, both with shape :math:`(N', k)` if :attr:`k` :math:`\leq n_{max}`, \
          otherwise :math:`(N', n_{max})`.

          | where
          |
          | :math:`B = \text{ batch size}`
          | :math:`N = \text{ number of support points}`
          | :math:`N' = \text{ number of query points}`
          | :math:`n_{max} = \text{ maximum number of neighbors a query point has}`
    """

    device = coords_query_points.device
    num_query_points = len(coords_query_points)
    invalid_neighbor_index = len(coords_support_points)

    k = min(k, int(point_cloud_sizes_support_points.amax().item()))

    neighbor_graph_edge_indices = torch_cluster.knn(
        coords_support_points,
        coords_query_points,
        k,
        batch_indices_support_points,
        batch_indices_query_points,
        batch_size=len(torch.unique(batch_indices_query_points)),
    )

    query_indices = neighbor_graph_edge_indices[0]
    support_indices = neighbor_graph_edge_indices[1]

    # compute neighbor distances
    query_coords = coords_query_points[query_indices]
    support_coords = coords_support_points[support_indices]
    neighbor_distances = torch.linalg.norm(query_coords - support_coords, dim=-1)  # pylint: disable=not-callable

    if (point_cloud_sizes_support_points >= k).all():
        neighbor_indices = support_indices.view(num_query_points, k)
        neighbor_distances = neighbor_distances.view(num_query_points, k)
    else:
        _, neighbor_counts = torch.unique(query_indices, return_counts=True)

        neighbor_indices = torch.full(
            (num_query_points * k,), fill_value=invalid_neighbor_index, device=device, dtype=torch.long
        )
        valid_neighbor_mask = torch.arange(k, device=device).unsqueeze(0).repeat(num_query_points, 1)
        valid_neighbor_mask = (valid_neighbor_mask < neighbor_counts.unsqueeze(-1)).view(-1)

        neighbor_indices[valid_neighbor_mask] = support_indices
        neighbor_indices = neighbor_indices.view(num_query_points, k)

        neighbor_distances_full = torch.full(
            (num_query_points * k,), fill_value=torch.inf, device=device, dtype=torch.float
        )
        neighbor_distances_full[valid_neighbor_mask] = neighbor_distances
        neighbor_distances = neighbor_distances_full.view(num_query_points, k)

    return neighbor_indices, neighbor_distances
