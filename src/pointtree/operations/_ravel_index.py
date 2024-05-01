""" Mapping of multi-dimensional indices to one-dimensional indices. """

__all__ = ["ravel_index", "ravel_multi_index", "unravel_flat_index"]

from typing import Union

import torch


def ravel_index(index: torch.Tensor, input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Converts index argument of a multi-dimensional
    `torch.gather() <https://pytorch.org/docs/stable/generated/torch.gather.html#torch-gather>`__ or
    `torch.scatter_add() <https://pytorch.org/docs/stable/generated/torch.scatter_add.html>`__ operation to an index
    that can be used to apply the operation on the flattened input tensor.

    Args:
        index: Multi-dimensional index.
        input_tensor: Input tensor of the `torch.gather()` or `torch.scatter_add()` operation.
        dim: Dimension in which the `torch.gather()` or `torch.scatter_add()` operation is to be applied.

    Returns:
        Index for applying `torch.gather()` or `torch.scatter_add()` on the flattened input tensor.
    """

    index_shape = index.size()
    input_shape = input_tensor.size()

    if dim >= input_tensor.ndim or dim < -1 * input_tensor.ndim:
        raise ValueError("Invalid dim")

    if dim < 0:
        dim = index.ndim + dim

    raveled_index = torch.zeros_like(index)
    for idx, shape in enumerate(index_shape):
        if idx != dim:
            unsqueezed_shape = [1] * len(index_shape)
            unsqueezed_shape[idx] = shape
            multi_index = torch.arange(shape, device=index.device).reshape(unsqueezed_shape)
        else:
            multi_index = index

        raveled_index *= input_shape[idx]
        raveled_index += multi_index

    return raveled_index.view(-1)


def ravel_multi_index(multi_index: torch.Tensor, dims: Union[torch.Size, torch.Tensor]) -> torch.Tensor:
    r"""
    PyTorch implementation of numpy.ravel_multi_index. This operation  is inverse to
    :py:meth:`pointtorch.operations.torch.unravel_flat_index`.

    Args:
        multi_index: Tensor containing the indices for each dimension.
        dims: The shape of the tensor into which the indices from `multi_index` apply.

    Returns:
        Indices for the flattened version of the tensor, referring to the same elements as referenced by
        :attr:`multi_index` for the non-flattened version of the tensor.

    Shape:
        - :attr:`multi_index`: :math:`(N, d_1, ..., d_D)`
        - :attr:`dims`: :math:`(D)`
        - Output: :math:`(N \cdot d_1 \cdot ... \cdot d_D)`

          | where
          |
          | :math:`N = \text{ number of items}`
          | :math:`D = \text{ number of index dimensions}`
          | :math:`d_i = \text{ number of elements along dimension } i`
    """

    if isinstance(dims, torch.Size):
        dims = torch.tensor(dims, device=multi_index.device, dtype=torch.long)

    dimension = len(multi_index)
    hash_values = multi_index[0].clone()
    for i in range(1, dimension):
        hash_values *= dims[i]
        hash_values += multi_index[i]
    return hash_values


def unravel_flat_index(flat_index: torch.Tensor, dims: Union[torch.Size, torch.Tensor]) -> torch.Tensor:
    r"""
    Converts an index for a 1-dimensional tensor into an index for an equivalent multi-dimensional tensor. This
    operation is inverse to :py:meth:`pointtorch.operations.torch.ravel_multi_index`.

    Args:
        flat_index: Tensor containing the indices for the flat array.
        dims: The shape of the tensor into which the returned indices should apply.

    Returns:
        Indices for the multi-dimensional version of the tensor, referring to the same elements as referenced by
        :attr:`flat_index` for the flattened version of the tensor.

    Shape:
        - :attr:`flat_index`: :math:`(N \cdot d_1 \cdot ... \cdot d_D)`
        - :attr:`dims`: :math:`(D)`
        - Output: :math:`(N, d_1, ..., d_D)`

          | where
          |
          | :math:`N = \text{ number of items}`
          | :math:`D = \text{ number of index dimensions}`
          | :math:`d_i = \text{ number of elements along dimension } i`
    """

    if isinstance(dims, torch.Size):
        dims = torch.tensor(dims, device=flat_index.device, dtype=torch.long)

    multi_index = torch.zeros((len(dims), len(flat_index)), dtype=torch.long, device=flat_index.device)

    for idx in range(len(dims)):
        divisor = float(torch.prod(dims[idx + 1 :]).item()) if idx < len(dims) else 1.0
        multi_index[idx, :] = torch.floor_divide(flat_index, divisor)
        flat_index = flat_index % divisor

    return multi_index
