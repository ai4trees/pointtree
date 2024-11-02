""" Packing of batches containing point clouds of varying size into a regular batch structures. """

__all__ = ["pack_batch"]

from typing import Tuple

import torch


def pack_batch(
    input_batch: torch.Tensor, point_cloud_sizes: torch.Tensor, fill_value: float = torch.inf
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Packs a batch containing point clouds of varying size into a regular batch structure by padding all point clouds to
    the same size.

    Args:
        input_batch: Batch to be packed.
        point_cloud_sizes: Number of points in each point cloud in the batch.
        fill_value: Value to be used to pad point clouds that contain less points than the largest point cloud in the
            batch. Defaults to `torch.inf`.

    Returns:
        Tuple of two tensors. The first tensor is the packed batch. Point clouds containing less than :math:`N_{max}`
        points are padded with :code:`fill_value`. The second tensor is a boolean mask, which is `True` in all positions
        where the packed batch contains valid points and `False` in all positions filled with :code:`fill_value`.

    Shape:
        - | :code:`input_batch`: :math:`(N_1 + ... + N_B, D)`
        - | :code:`point_cloud_sizes`: :math:`(B)`
        - | Output: Tuple of two tensors with shape :math:`(B, N_{max}, D)` and :math:`(B, N_{max})`.
          |
          | where
          |
          | :math:`B = \text{ batch size}`
          | :math:`D = \text{ number of feature channels}`
          | :math:`N_i = \text{ number of points in the i-th point cloud}`
          | :math:`N_{max} = \text{ number of points in the largest point cloud in the batch}`
    """

    batch_size = len(point_cloud_sizes)
    max_point_cloud_size = int(point_cloud_sizes.max().item())
    num_channels = input_batch.size(-1)

    if (point_cloud_sizes == max_point_cloud_size).all():
        # in this case, all point clouds already have the same size and no padding is needed

        mask = input_batch.new_ones((batch_size, max_point_cloud_size), dtype=torch.bool)

        return input_batch.view(batch_size, max_point_cloud_size, num_channels), mask

    packed_batch = input_batch.new_full((batch_size * max_point_cloud_size, num_channels), fill_value=fill_value)

    mask = torch.arange(max_point_cloud_size, device=input_batch.device).unsqueeze(0).repeat(batch_size, 1)
    mask = (mask < point_cloud_sizes.unsqueeze(-1)).view(-1)

    packed_batch[mask] = input_batch
    packed_batch = packed_batch.view(batch_size, max_point_cloud_size, num_channels)
    mask = mask.view(batch_size, max_point_cloud_size)

    return packed_batch, mask
